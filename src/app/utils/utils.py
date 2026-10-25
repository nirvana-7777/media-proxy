import base64


def decode_base64_url(encoded: str) -> str:
    """
    Decode base64-encoded URL, handling missing padding

    Args:
        encoded: Base64 URL-safe encoded string (with or without padding)

    Returns:
        Decoded URL string

    Raises:
        ValueError: If base64 decoding fails
    """
    try:
        # Add padding if needed
        padding_needed = 4 - len(encoded) % 4
        if padding_needed != 4:  # Only add padding if needed
            encoded += "=" * padding_needed

        return base64.urlsafe_b64decode(encoded.encode("utf-8")).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Failed to decode base64 URL: {e}")


def compose_url_from_template(base64_part: str, template_suffix: str = "") -> str:
    """
    Compose full URL from base64-encoded base URL and optional template suffix

    Args:
        base64_part: Base64 URL-safe encoded base URL (with or without padding)
        template_suffix: Optional template path to append (e.g., "segment-123.m4s")

    Returns:
        Complete URL

    Raises:
        ValueError: If base64 decoding fails
    """
    try:
        base_url = decode_base64_url(base64_part)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 URL: {e}")

    if not template_suffix:
        return base_url

    # Ensure proper joining (base_url might or might not end with /)
    if base_url.endswith("/"):
        return base_url + template_suffix
    else:
        return base_url + "/" + template_suffix


def socks_error_types() -> tuple[type[BaseException], ...]:
    """
    Exception classes raised by ``aiohttp-socks`` on connect failure.

    aiohttp-socks 0.12+ wraps SOCKS handshake failures in its own
    ``ProxyConnectionError``/``ProxyTimeoutError``/``ProxyError`` classes,
    which are **not** ``OSError`` subclasses.  Without listing them
    explicitly, a dead SOCKS endpoint surfaces as a 500 instead of a 502 in
    ``stream_proxy``'s ``except (aiohttp.ClientError, OSError, ...)`` arm.

    Returns an empty tuple when the library is not installed, so callers can
    always write::

        except (*socks_error_types()):
            ...

    safely, whether or not ``aiohttp-socks`` is a runtime dependency.
    """
    try:
        from aiohttp_socks import (
            ProxyConnectionError,
            ProxyError,
            ProxyTimeoutError,
        )
    except ImportError:
        return ()
    return (ProxyConnectionError, ProxyTimeoutError, ProxyError)
