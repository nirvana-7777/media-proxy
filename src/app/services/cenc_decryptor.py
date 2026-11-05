import logging
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

logger = logging.getLogger(__name__)

_ZERO_IV = b"\x00" * 16


class CENCDecryptor:
    """AES-128-CTR decryption with subsample support (CENC 'cenc' scheme).

    One native CTR context per sample.  CTR is stateful across ``update()``
    calls, so the encrypted ranges of a sample (which share one continuous
    keystream, clear bytes excluded) are fed to it one after the other and
    OpenSSL does the counter handling.  No keystream is materialised.
    """

    def __init__(
        self,
        data: bytearray,
        key: bytes,
        samples: List[Dict[str, Any]],
        debug: bool = False,
    ):
        """
        Args:
            data: Reference to the MP4 data (bytearray, modified in place)
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
                f"CENCDecryptor initialized: key_len={len(key)}, "
                f"samples={len(samples)}, data_size={len(data)}"
            )

    @staticmethod
    def _initial_counter_block(iv: bytes) -> Optional[bytes]:
        """Build the 16-byte initial counter block for a CENC IV.

        8-byte IV  -> IV || 64-bit zero block counter
        16-byte IV -> used as the full 128-bit counter
        (OpenSSL increments all 128 bits; for 8-byte IVs the carry into the
        IV half would need 2^64 blocks in one sample, i.e. never.)
        """
        if len(iv) == 8:
            return iv + b"\x00" * 8
        if len(iv) == 16:
            return iv
        return None

    @staticmethod
    def _plan_sample(sample: Dict[str, Any], position: int) -> Tuple[List[Tuple[int, int]], int]:
        """Return ([(pos, size) encrypted ranges], position after the sample)."""
        subsamples = sample.get("subsamples") or []
        if subsamples:
            ranges: List[Tuple[int, int]] = []
            current = position
            for sub in subsamples:
                current += sub.get("clear", 0)
                encrypted = sub.get("encrypted", 0)
                if encrypted > 0:
                    ranges.append((current, encrypted))
                    current += encrypted
            return ranges, current

        full = sample.get("full_encrypted_size") or 0
        if full > 0:
            return [(position, full)], position + full
        return [], position

    def decrypt(self, mdat_offset: int) -> bool:
        """
        Decrypt MP4 data in-place using AES-128-CTR

        Args:
            mdat_offset: Starting offset of mdat box data

        Returns:
            True if successful, False otherwise
        """
        data = self.data
        data_len = len(data)
        view = memoryview(data)
        position = mdat_offset

        try:
            for i, sample in enumerate(self.samples):
                ranges, end = self._plan_sample(sample, position)

                if ranges:
                    if end > data_len:
                        logger.error(f"Sample {i} exceeds data bounds: {end} > {data_len}")
                        return False

                    iv = sample.get("iv")
                    if iv is None:
                        iv = _ZERO_IV
                    counter = self._initial_counter_block(iv)
                    if counter is None:
                        logger.error(f"Sample {i}: unsupported IV length {len(iv)}")
                        return False

                    ctx = Cipher(algorithms.AES(self.key), modes.CTR(counter)).encryptor()
                    for pos, size in ranges:
                        view[pos : pos + size] = ctx.update(view[pos : pos + size])

                    if self.debug:
                        logger.debug(f"Sample {i}: decrypted {len(ranges)} range(s), end={end}")

                position = end
        finally:
            view.release()

        if self.debug:
            logger.debug(f"Decryption complete: processed {len(self.samples)} samples")
        return True
