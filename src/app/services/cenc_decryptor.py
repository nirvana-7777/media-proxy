from typing import List, Dict, Any, Optional
import logging
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class CENCDecryptor:
    """AES-128-CTR decryption with subsample support (CENC scheme)"""

    def __init__(self, data: bytearray, key: bytes, samples: List[Dict[str, Any]], debug: bool = False):
        """
        Args:
            data: Reference to the MP4 data (bytearray for in-place modification)
            key: AES-128 key (16 bytes)
            samples: List of sample dictionaries with IV and subsample info
            debug: Enable debug logging
        """
        if len(key) != 16:
            raise ValueError(f"Key must be exactly 16 bytes, got {len(key)}")

        self.data = data
        self.key = key
        self.samples = samples
        self.debug = debug

    def decrypt(self, mdat_offset: int) -> bool:
        """
        Decrypt MP4 data in-place using AES-128-CTR

        Args:
            mdat_offset: Starting offset of mdat box data

        Returns:
            True if successful, False otherwise
        """
        position = mdat_offset

        for i, sample in enumerate(self.samples):
            # Get IV - keep original size (8 or 16 bytes), expand only when creating counter
            original_iv = sample.get('iv')
            if original_iv is None:
                original_iv = b'\x00' * 16

            # Process subsample information
            subsample_positions = []
            subsample_sizes = []
            total_encrypted_size = 0
            current_position = position

            if 'subsamples' in sample and sample['subsamples']:
                for sub in sample['subsamples']:
                    clear = sub.get('clear', 0)
                    encrypted = sub.get('encrypted', 0)
                    current_position += clear

                    if encrypted > 0:
                        subsample_positions.append(current_position)
                        subsample_sizes.append(encrypted)
                        total_encrypted_size += encrypted
                        current_position += encrypted
            elif 'full_encrypted_size' in sample and sample['full_encrypted_size'] > 0:
                total_encrypted_size = sample['full_encrypted_size']
                subsample_positions.append(position)
                subsample_sizes.append(total_encrypted_size)
                current_position = position + total_encrypted_size

            if total_encrypted_size > 0:
                # Validate data bounds
                if current_position > len(self.data):
                    if self.debug:
                        logger.error(f"Sample {i} exceeds data bounds: {current_position} > {len(self.data)}")
                    return False

                # Generate keystream for the entire encrypted portion
                keystream = self._generate_keystream(original_iv, total_encrypted_size)
                if keystream is None:
                    if self.debug:
                        logger.error(f"Failed to generate keystream for sample {i}")
                    return False

                # Process all subsamples
                keystream_offset = 0
                for j in range(len(subsample_positions)):
                    pos = subsample_positions[j]
                    size = subsample_sizes[j]

                    # Validate bounds
                    if pos + size > len(self.data):
                        if self.debug:
                            logger.error(f"Subsample {j} of sample {i} exceeds data bounds")
                        return False

                    # Skip empty blocks
                    if size == 0:
                        continue

                    # XOR using NumPy (150x faster than Python loop)
                    try:
                        data_view = np.frombuffer(self.data, dtype=np.uint8, offset=pos, count=size)
                        key_view = np.frombuffer(keystream, dtype=np.uint8, offset=keystream_offset, count=size)
                        np.bitwise_xor(data_view, key_view, out=data_view)
                    except Exception as e:
                        if self.debug:
                            logger.error(f"NumPy XOR failed, falling back to Python: {e}")
                        # Fallback to Python loop (rare case)
                        for k in range(size):
                            self.data[pos + k] ^= keystream[keystream_offset + k]

                    keystream_offset += size

            position = current_position

        return True

    def _generate_keystream(self, iv: bytes, size: int) -> Optional[bytes]:
        """
        Generate AES-128-CTR keystream using batched ECB encryption

        Args:
            iv: Initialization vector (8 or 16 bytes)
            size: Required keystream size in bytes

        Returns:
            Keystream bytes or None on error
        """
        try:
            # Expand IV to 16 bytes if needed
            if len(iv) < 16:
                iv_expanded = iv.ljust(16, b'\x00')
            else:
                iv_expanded = iv[:16]

            # Calculate blocks needed
            blocks_needed = (size + 15) // 16

            # Build all counter blocks at once
            all_counter_blocks = bytearray(blocks_needed * 16)
            for block_num in range(blocks_needed):
                counter_block = self._create_counter_block(iv, iv_expanded, block_num)
                offset = block_num * 16
                all_counter_blocks[offset:offset + 16] = counter_block

            # Encrypt all blocks in one operation
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.ECB(),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            keystream = encryptor.update(bytes(all_counter_blocks)) + encryptor.finalize()

            # Return only the needed bytes (last block might be partial)
            return keystream[:size]

        except Exception as e:
            if self.debug:
                logger.error(f"Keystream generation failed: {e}")
            return None

    @staticmethod
    def _create_counter_block(original_iv: bytes, expanded_iv: bytes, block_num: int) -> bytes:
        """
        Create CTR counter block following CENC specification

        Args:
            original_iv: Original IV as received (8 or 16 bytes)
            expanded_iv: IV expanded to 16 bytes
            block_num: Block number to add

        Returns:
            Counter value for the given block (always 16 bytes)
        """
        if len(original_iv) == 8:
            # CENC standard: 8-byte IV + 8-byte block counter
            # The IV stays in the first 8 bytes, counter in the last 8 bytes
            counter = bytearray(16)
            counter[:8] = original_iv
            counter[8:] = block_num.to_bytes(8, 'big')
            return bytes(counter)
        else:
            # For 16-byte IV: treat as 128-bit big-endian integer and increment
            counter_int = int.from_bytes(expanded_iv, 'big')
            counter_int = (counter_int + block_num) & ((1 << 128) - 1)
            return counter_int.to_bytes(16, 'big')