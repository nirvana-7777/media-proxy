import logging
from typing import Any, Dict, List, Optional

import numpy as np
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

logger = logging.getLogger(__name__)


class CENCDecryptor:
    """AES-128-CTR decryption with subsample support (CENC scheme)"""

    def __init__(
        self,
        data: bytearray,
        key: bytes,
        samples: List[Dict[str, Any]],
        debug: bool = False,
    ):
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

        if self.debug:
            logger.debug(
                f"CENCDecryptor initialized: key_len={len(key)},"
                f"samples={len(samples)}, data_size={len(data)}"
            )

    def decrypt(self, mdat_offset: int) -> bool:
        """
        Decrypt MP4 data in-place using AES-128-CTR

        Args:
            mdat_offset: Starting offset of mdat box data

        Returns:
            True if successful, False otherwise
        """
        if self.debug:
            logger.debug(
                f"Starting decryption: mdat_offset={mdat_offset}, total_samples={len(self.samples)}"
            )

        position = mdat_offset

        for i, sample in enumerate(self.samples):
            # Get IV - keep original size (8 or 16 bytes),
            # expand only when creating counter
            original_iv = sample.get("iv")
            if original_iv is None:
                original_iv = b"\x00" * 16
                if self.debug:
                    logger.debug(f"Sample {i}: No IV provided, using zero IV")

            if self.debug:
                logger.debug(
                    f"Sample {i}: iv_len={len(original_iv)}, iv_hex={original_iv.hex()[:32]}..."
                )

            # Process subsample information
            subsample_positions = []
            subsample_sizes = []
            total_encrypted_size = 0
            current_position = position

            if "subsamples" in sample and sample["subsamples"]:
                if self.debug:
                    logger.debug(f"Sample {i}: Processing {len(sample['subsamples'])} subsamples")

                for j, sub in enumerate(sample["subsamples"]):
                    clear = sub.get("clear", 0)
                    encrypted = sub.get("encrypted", 0)

                    if self.debug and j < 3:  # Log first 3 subsamples
                        logger.debug(f"  Subsample {j}: clear={clear}, encrypted={encrypted}")

                    current_position += clear

                    if encrypted > 0:
                        subsample_positions.append(current_position)
                        subsample_sizes.append(encrypted)
                        total_encrypted_size += encrypted
                        current_position += encrypted

                if self.debug:
                    logger.debug(f"Sample {i}: total_encrypted_size={total_encrypted_size}")

            elif "full_encrypted_size" in sample and sample["full_encrypted_size"] > 0:
                total_encrypted_size = sample["full_encrypted_size"]
                subsample_positions.append(position)
                subsample_sizes.append(total_encrypted_size)
                current_position = position + total_encrypted_size

                if self.debug:
                    logger.debug(f"Sample {i}: Fully encrypted, size={total_encrypted_size}")

            if total_encrypted_size > 0:
                # Validate data bounds
                if current_position > len(self.data):
                    if self.debug:
                        logger.error(
                            f"Sample {i} exceeds data bounds: {current_position} > {len(self.data)}"
                        )
                    return False

                if self.debug:
                    logger.debug(
                        f"Sample {i}: Generating keystream of {total_encrypted_size} bytes"
                    )

                # Generate keystream for the entire encrypted portion
                keystream = self._generate_keystream(original_iv, total_encrypted_size)
                if keystream is None:
                    if self.debug:
                        logger.error(f"Failed to generate keystream for sample {i}")
                    return False

                if self.debug:
                    logger.debug(
                        f"Sample {i}: Keystream generated, "
                        f"decrypting {len(subsample_positions)} subsample(s)"
                    )

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

                    if self.debug and j < 3:  # Log first 3 subsamples
                        logger.debug(
                            f"  Decrypting subsample {j}: pos={pos}, "
                            f"size={size}, keystream_offset={keystream_offset}"
                        )

                    # XOR using NumPy (150x faster than Python loop)
                    try:
                        data_view = np.frombuffer(self.data, dtype=np.uint8, offset=pos, count=size)
                        key_view = np.frombuffer(
                            keystream,
                            dtype=np.uint8,
                            offset=keystream_offset,
                            count=size,
                        )
                        np.bitwise_xor(data_view, key_view, out=data_view)
                    except Exception as e:
                        if self.debug:
                            logger.error(f"NumPy XOR failed, falling back to Python: {e}")
                        # Fallback to Python loop (rare case)
                        for k in range(size):
                            self.data[pos + k] ^= keystream[keystream_offset + k]

                    keystream_offset += size

                if self.debug:
                    logger.debug(
                        f"Sample {i}: Decryption complete, new position={current_position}"
                    )
            else:
                if self.debug:
                    logger.debug(f"Sample {i}: No encrypted data, skipping")

            position = current_position

        if self.debug:
            logger.debug(f"Decryption complete: processed {len(self.samples)} samples")

        return True

    def _generate_keystream(self, iv: bytes, size: int) -> Optional[bytes]:
        """
        Generate AES-128-CTR keystream using vectorized NumPy counter generation.
        Eliminates the Python loop for ~100x faster setup on 3MB fragments.
        """
        try:
            blocks_needed = (size + 15) // 16

            # 1. Create a buffer for all counter blocks (N blocks, 16 bytes each)
            all_blocks = np.zeros((blocks_needed, 16), dtype=np.uint8)

            if len(iv) == 8:
                # CENC 8-byte IV: [8 bytes IV] [8 bytes block_counter]
                # Set IV in the first 8 columns of all rows
                all_blocks[:, :8] = np.frombuffer(iv, dtype=np.uint8)
                # Generate big-endian 64-bit counters [0, 1, 2...]
                counters = np.arange(blocks_needed, dtype=">u8")
                # Write counters into the last 8 columns using a pointer view
                all_blocks[:, 8:].view(">u8")[:] = counters[:, np.newaxis]
            else:
                # CENC 16-byte IV: Entire 16 bytes incremented as a 128-bit integer
                # We split the 16-byte IV into two 64-bit halves to handle math in NumPy
                iv_high = int.from_bytes(iv[:8], "big")
                iv_low = int.from_bytes(iv[8:], "big")

                # Calculate increments and handle the 64-bit carry manually
                low_counters = np.arange(blocks_needed, dtype=">u8") + iv_low
                # Check for overflow to increment the high 64 bits
                # (Standard uint64 addition in NumPy wraps around, so we detect that)
                carry = (low_counters < iv_low).astype(">u8")
                high_counters = np.full(blocks_needed, iv_high, dtype=">u8") + carry

                # Write both halves into the block buffer
                all_blocks[:, :8].view(">u8")[:] = high_counters[:, np.newaxis]
                all_blocks[:, 8:].view(">u8")[:] = low_counters[:, np.newaxis]

            # 2. Encrypt all blocks in one operation (Fast OpenSSL/C call)
            cipher = Cipher(algorithms.AES(self.key), modes.ECB(), backend=default_backend())
            encryptor = cipher.encryptor()
            # .tobytes() on a contiguous NumPy array is a very fast memory view export
            keystream = encryptor.update(all_blocks.tobytes()) + encryptor.finalize()

            return keystream[:size]

        except Exception as e:
            if self.debug:
                logger.error(f"Vectorized keystream generation failed: {e}")
            return None
