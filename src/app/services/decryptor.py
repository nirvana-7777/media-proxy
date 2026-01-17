import aiohttp
import asyncio
from typing import Optional
import logging

from .mp4_parser import MP4Parser

logger = logging.getLogger(__name__)


class DecryptorService:
    """Service for downloading and decrypting CENC-encrypted MP4 segments"""

    def __init__(self, max_concurrent_downloads: int = 10):
        """
        Initialize the decryptor service

        Args:
            max_concurrent_downloads: Maximum number of concurrent downloads
        """
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent_downloads)

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'CENC-Decryptor/1.0'}
            )
        return self.session

    async def decrypt_segment(
            self,
            key: str,
            url: str,
            iv: Optional[str] = None,
            kid: Optional[str] = None,
            algorithm: str = "aes-128-ctr"
    ) -> bytes:
        """
        Download and decrypt an MP4 segment

        Args:
            key: Hex-encoded decryption key (32 hex chars = 16 bytes)
            url: URL of the segment to decrypt
            iv: Optional hex-encoded initialization vector (for testing)
            kid: Optional hex-encoded Key ID
            algorithm: Encryption algorithm (default: aes-128-ctr)

        Returns:
            Decrypted MP4 segment as bytes

        Raises:
            ValueError: If key format is invalid
            Exception: If download or decryption fails
        """
        # Validate key format
        if not key or len(key) != 32:
            raise ValueError(f"Key must be 32 hex characters (16 bytes), got {len(key) if key else 0}")

        try:
            bytes.fromhex(key)
        except ValueError:
            raise ValueError("Key must be valid hexadecimal")

        async with self.semaphore:
            try:
                # Download the segment
                encrypted_data = await self._download_segment(url)

                if not encrypted_data:
                    raise Exception("Downloaded segment is empty")

                # Convert to bytearray for in-place modification
                data = bytearray(encrypted_data)

                # Parse MP4 structure
                parser = MP4Parser(data, kid=kid, key=key, debug=False)

                if not parser.parse():
                    raise Exception("Failed to parse MP4 structure")

                # The parser already applies decryption in-place when it encounters mdat boxes
                # Return the decrypted data
                return bytes(data)

            except aiohttp.ClientError as e:
                logger.error(f"Network error downloading segment from {url}: {str(e)}")
                raise Exception(f"Failed to download segment: {str(e)}")
            except ValueError as e:
                logger.error(f"Validation error: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Failed to decrypt segment from {url}: {str(e)}")
                raise

    async def _download_segment(self, url: str) -> bytes:
        """
        Download segment with retry logic

        Args:
            url: URL to download from

        Returns:
            Downloaded data as bytes

        Raises:
            aiohttp.ClientError: If all retry attempts fail
        """
        session = await self.get_session()

        last_error = None
        for attempt in range(3):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    data = await response.read()

                    if self._is_valid_mp4(data):
                        return data
                    else:
                        logger.warning(f"Downloaded data doesn't appear to be valid MP4")
                        return data  # Return anyway, parser will handle errors

            except aiohttp.ClientError as e:
                last_error = e
                if attempt < 2:
                    wait_time = 1 * (attempt + 1)
                    logger.warning(f"Download attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All download attempts failed for {url}")
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < 2:
                    wait_time = 2 * (attempt + 1)
                    logger.warning(f"Download timeout on attempt {attempt + 1}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All download attempts timed out for {url}")

        raise last_error if last_error else Exception("Download failed")

    @staticmethod
    def _is_valid_mp4(data: bytes) -> bool:
        """
        Quick check if data looks like valid MP4

        Args:
            data: Data to check

        Returns:
            True if data appears to be MP4 format
        """
        if len(data) < 8:
            return False

        # Check for common MP4 box types at start
        common_types = [b'ftyp', b'styp', b'moof', b'moov', b'mdat']
        box_type = data[4:8]

        return box_type in common_types

    async def decrypt_batch(
            self,
            segments: list[dict],
            max_concurrent: Optional[int] = None
    ) -> list[bytes]:
        """
        Decrypt multiple segments concurrently

        Args:
            segments: List of dicts with 'url', 'key', and optional 'kid', 'iv'
            max_concurrent: Override default concurrency limit

        Returns:
            List of decrypted segment data in same order as input
        """
        if max_concurrent:
            original_limit = self.semaphore._value
            self.semaphore = asyncio.Semaphore(max_concurrent)

        try:
            tasks = []
            for seg in segments:
                task = self.decrypt_segment(
                    key=seg['key'],
                    url=seg['url'],
                    kid=seg.get('kid'),
                    iv=seg.get('iv')
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for errors
            decrypted_segments = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Segment {i} failed: {str(result)}")
                    raise result
                decrypted_segments.append(result)

            return decrypted_segments

        finally:
            if max_concurrent:
                self.semaphore = asyncio.Semaphore(original_limit)

    async def close(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            # Wait a bit for connections to close
            await asyncio.sleep(0.1)

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()