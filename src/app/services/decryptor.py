import aiohttp
import asyncio
import base64
import binascii
from typing import Optional, Dict, Any
import logging
import io

from .mp4_parser import MP4Parser
from .cenc_decryptor import CENCDecryptor

logger = logging.getLogger(__name__)


class DecryptorService:
    def __init__(self, max_concurrent_downloads: int = 10):
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent_downloads)

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def decrypt_segment(
            self,
            key: str,
            url: str,
            iv: Optional[str] = None,
            algorithm: str = "aes-128-ctr"
    ) -> bytes:
        """
        Download and decrypt an MP4 segment

        Args:
            key: Hex-encoded decryption key
            url: URL of the segment to decrypt
            iv: Optional base64 encoded initialization vector
            algorithm: Encryption algorithm (default: aes-128-ctr)

        Returns:
            Decrypted MP4 segment as bytes
        """
        async with self.semaphore:
            try:
                # Download the segment
                encrypted_data = await self._download_segment(url)

                # Convert to bytearray for in-place modification
                data = bytearray(encrypted_data)

                # Parse MP4 structure
                parser = MP4Parser(data, key=key, debug=False)

                if not parser.parse():
                    raise Exception("Failed to parse MP4 structure")

                # The parser already applies decryption in-place when it encounters mdat boxes
                # So our data is now decrypted

                return bytes(data)

            except Exception as e:
                logger.error(f"Failed to decrypt segment from {url}: {str(e)}")
                raise

    async def _download_segment(self, url: str) -> bytes:
        """Download segment with retry logic"""
        session = await self.get_session()

        for attempt in range(3):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.read()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(1 * (attempt + 1))

    async def close(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()