import asyncio
import logging
from typing import Dict, List, Optional, Union, cast

import aiohttp

from .mp4_parser import MP4Parser

logger = logging.getLogger(__name__)

# Default Chrome User-Agent for Windows
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


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
        self.max_concurrent = max_concurrent_downloads

    async def get_session(
        self, proxy: Optional[str] = None, user_agent: Optional[str] = None
    ) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp session

        Args:
            proxy: Optional proxy URL
            user_agent: Optional user agent string

        Returns:
            Configured ClientSession
        """
        # If proxy or custom user agent is specified, create a new session for this request
        if proxy or user_agent:
            return await self._create_session(proxy, user_agent)

        # Otherwise use the default session
        if self.session is None or self.session.closed:
            self.session = await self._create_session(None, None)
        return self.session

    async def _create_session(
        self, proxy: Optional[str] = None, user_agent: Optional[str] = None
    ) -> aiohttp.ClientSession:
        """
        Create a new aiohttp session with specified configuration

        Args:
            proxy: Optional proxy URL
            user_agent: Optional user agent string

        Returns:
            Configured ClientSession
        """
        timeout = aiohttp.ClientTimeout(total=30, connect=10)

        # Set user agent
        ua = user_agent if user_agent else DEFAULT_USER_AGENT
        headers = {"User-Agent": ua}

        # Configure connector and proxy
        connector: Union[aiohttp.TCPConnector, "ProxyConnector", None] = None

        if proxy:
            # Check if it's a SOCKS proxy
            if proxy.startswith("socks"):
                try:
                    # Use aiohttp-socks for SOCKS proxies
                    from aiohttp_socks import ProxyConnector

                    connector = ProxyConnector.from_url(proxy)
                except ImportError:
                    raise Exception(
                        "SOCKS proxy support requires aiohttp-socks. "
                        "Install with: pip install aiohttp-socks"
                    )
            else:
                # HTTP/HTTPS proxy
                connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        else:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)

        return aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers,
            trust_env=False,  # Don't use environment proxy settings
        )

    async def decrypt_segment(
        self,
        key: str,
        url: str,
        iv: Optional[str] = None,
        kid: Optional[str] = None,
        algorithm: str = "aes-128-ctr",
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> bytes:
        """
        Download and decrypt an MP4 segment

        Args:
            key: Hex-encoded decryption key (32 hex chars = 16 bytes)
            url: URL of the segment to decrypt
            iv: Optional hex-encoded initialization vector (for testing)
            kid: Optional hex-encoded Key ID
            algorithm: Encryption algorithm (default: aes-128-ctr)
            proxy: Optional proxy URL
            user_agent: Optional user agent string

        Returns:
            Decrypted MP4 segment as bytes

        Raises:
            ValueError: If key format is invalid
            Exception: If download or decryption fails
        """
        # Validate key format
        if not key or len(key) != 32:
            raise ValueError(
                f"Key must be 32 hex characters (16 bytes), got {len(key) if key else 0}"
            )

        try:
            bytes.fromhex(key)
        except ValueError:
            raise ValueError("Key must be valid hexadecimal")

        # Create session for this request if proxy/UA specified
        session = await self.get_session(proxy, user_agent)
        should_close_session = proxy is not None or user_agent is not None

        async with self.semaphore:
            try:
                # Download the segment
                encrypted_data = await self._download_segment(url, session, proxy)

                if not encrypted_data:
                    raise Exception("Downloaded segment is empty")

                # Convert to bytearray for in-place modification
                data = bytearray(encrypted_data)

                # Parse MP4 structure (this also decrypts in-place)
                parser = MP4Parser(data, kid=kid, key=key, debug=False)

                if not parser.parse():
                    raise Exception("Failed to parse MP4 structure")

                # Return the decrypted data
                return bytes(data)

            except aiohttp.ClientError as e:
                logger.error(f"Network error downloading segment from {url}: {str(e)}")
                if proxy:
                    raise Exception(f"Failed to download segment via proxy {proxy}: {str(e)}")
                raise Exception(f"Failed to download segment: {str(e)}")
            except ValueError as e:
                logger.error(f"Validation error: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Failed to decrypt segment from {url}: {str(e)}")
                raise
            finally:
                # Close session if it was created for this request
                if should_close_session and not session.closed:
                    await session.close()

    async def _download_segment(
        self, url: str, session: aiohttp.ClientSession, proxy: Optional[str] = None
    ) -> bytes:
        """
        Download segment with retry logic

        Args:
            url: URL to download from
            session: ClientSession to use
            proxy: Optional proxy URL (for HTTP/HTTPS proxies)

        Returns:
            Downloaded data as bytes

        Raises:
            aiohttp.ClientError: If all retry attempts fail
        """
        # If proxy is HTTP/HTTPS (not SOCKS), pass it to the request
        proxy_url = None
        if proxy and not proxy.startswith("socks"):
            proxy_url = proxy

        last_error: Optional[Exception] = None
        retry_count = 3 if not proxy else 1  # Don't retry with proxy to avoid confusion

        for attempt in range(retry_count):
            try:
                async with session.get(url, proxy=proxy_url) as response:
                    response.raise_for_status()
                    data = await response.read()

                    if self._is_valid_mp4(data):
                        return data
                    else:
                        logger.warning("Downloaded data doesn't appear to be valid MP4")
                        return data  # Return anyway, parser will handle errors

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if proxy:
                    # Don't retry with proxy - fail immediately
                    logger.error(f"Proxy request failed: {str(e)}")
                    raise Exception(f"Failed to download via proxy {proxy}: {str(e)}")

                if attempt < retry_count - 1:
                    wait_time = (
                        1 * (attempt + 1)
                        if isinstance(e, aiohttp.ClientError)
                        else 2 * (attempt + 1)
                    )
                    logger.warning(
                        f"Download attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All download attempts failed for {url}")
                    if last_error:
                        raise last_error
                    else:
                        raise Exception("Download failed")

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
        common_types = [b"ftyp", b"styp", b"moof", b"moov", b"mdat"]
        box_type = data[4:8]

        return box_type in common_types

    async def decrypt_batch(
        self, segments: List[Dict], max_concurrent: Optional[int] = None
    ) -> List[bytes]:
        """
        Decrypt multiple segments concurrently

        Args:
            segments: List of dicts with 'url', 'key', and
            optional 'kid', 'iv', 'proxy', 'user_agent'
            max_concurrent: Override default concurrency limit

        Returns:
            List of decrypted segment data in same order as input
        """
        original_limit = self.semaphore._value
        if max_concurrent and max_concurrent != self.max_concurrent:
            self.semaphore = asyncio.Semaphore(max_concurrent)

        try:
            tasks = []
            for seg in segments:
                task = self.decrypt_segment(
                    key=seg["key"],
                    url=seg["url"],
                    kid=seg.get("kid"),
                    iv=seg.get("iv"),
                    proxy=seg.get("proxy"),
                    user_agent=seg.get("user_agent"),
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for errors
            decrypted_segments: List[bytes] = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Segment {i} failed: {str(result)}")
                    raise result
                decrypted_segments.append(cast(bytes, result))

            return decrypted_segments

        finally:
            if max_concurrent and max_concurrent != self.max_concurrent:
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
