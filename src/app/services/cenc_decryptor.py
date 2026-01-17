import base64
import struct
from typing import List, Dict, Any, Optional
import logging
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
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
            # Get IV or use default (8 or 16 bytes)
            iv = self._expand_iv(sample.get('iv'), append=True)

            # Process subsample information
            subsample_positions = []
            subsample_sizes = []
            total_encrypted_size = 0
            current_position = position

            if 'subsamples' in sample:
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
                # Generate keystream for the entire encrypted portion
                keystream = self._generate_keystream(iv, total_encrypted_size)
                if keystream is None:
                    if self.debug:
                        logger.error(f"Failed to generate keystream for sample {i}")
                    return False

                # Process all subsamples
                keystream_offset = 0
                for j in range(len(subsample_positions)):
                    pos = subsample_positions[j]
                    size = subsample_sizes[j]

                    # XOR data with keystream in-place
                    for k in range(size):
                        self.data[pos + k] ^= keystream[keystream_offset + k]

                    keystream_offset += size

            position = current_position

        return True

    def _expand_iv(self, iv: Optional[bytes], append: bool = True) -> bytes:
        """Expand IV to 16 bytes"""
        if iv is None:
            return b'\x00' * 16

        if len(iv) >= 16:
            return iv[:16]

        if append:
            return iv.ljust(16, b'\x00')
        else:
            return iv.rjust(16, b'\x00')

    def _generate_keystream(self, iv: bytes, size: int) -> Optional[bytes]:
        """
        Generate AES-128-CTR keystream

        Args:
            iv: 16-byte initialization vector
            size: Required keystream size in bytes

        Returns:
            Keystream bytes or None on error
        """
        try:
            # For CTR mode, we need to encrypt a counter
            # The IV is used as the initial counter value
            cipher = Cipher(
                algorithms.AES(self.key),
                # In CTR mode, the entire IV is used as the counter
                # We need to create a custom CTR implementation
                mode=None,  # We'll implement CTR manually
                backend=default_backend()
            )

            # Manually implement AES-CTR
            keystream = bytearray()
            blocks_needed = (size + 15) // 16

            for block_num in range(blocks_needed):
                # Create counter block (IV + block number)
                counter_block = self._increment_ctr(iv, block_num)

                # Encrypt the counter block
                encryptor = cipher.encryptor()
                keystream_block = encryptor.update(counter_block) + encryptor.finalize()
                keystream.extend(keystream_block)

            return bytes(keystream[:size])

        except Exception as e:
            if self.debug:
                logger.error(f"Keystream generation failed: {e}")
            return None

    def _increment_ctr(self, iv: bytes, block_num: int) -> bytes:
        """
        Increment CTR counter

        Args:
            iv: Initial counter value (16 bytes)
            block_num: Block number to add

        Returns:
            Counter value for the given block
        """
        # For 8-byte IVs (common in CENC), we need to handle differently
        if len(iv) == 8:
            # Common CENC pattern: 8-byte IV + 8-byte counter
            counter = bytearray(16)
            counter[:8] = iv
            # Add block_num to the last 8 bytes (big-endian)
            block_bytes = block_num.to_bytes(8, 'big')
            counter[8:] = block_bytes
            return bytes(counter)
        else:
            # For 16-byte IVs, just increment
            counter_int = int.from_bytes(iv, 'big')
            counter_int += block_num
            return counter_int.to_bytes(16, 'big')