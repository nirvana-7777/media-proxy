import base64


def decode_base64_url(encoded: str) -> str:
    """
    Decode base64-encoded URL

    Args:
        encoded: Base64 URL-safe encoded string

    Returns:
        Decoded URL string
    """
    return base64.urlsafe_b64decode(encoded.encode("utf-8")).decode("utf-8")


def compose_url_from_template(base64_part: str, template_suffix: str = "") -> str:
    """
    Compose full URL from base64-encoded base URL and optional template suffix

    Args:
        base64_part: Base64 URL-safe encoded base URL
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
