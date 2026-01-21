import base64
import binascii
import logging
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SampleInfo:
    """Container for sample information"""

    index: int
    iv: Optional[bytes] = None
    subsamples: List[Dict[str, int]] = field(default_factory=list)
    full_encrypted_size: Optional[int] = None
    duration: Optional[int] = None
    flags: Optional[int] = None
    composition_offset: Optional[int] = None


class MP4Parser:
    """High-performance MP4 parser for fragmented MP4 with CENC encryption"""

    def __init__(
        self,
        data: bytearray,
        key: Optional[str] = None,  # Changed to Optional
        kid: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Args:
            data: MP4 data as bytearray (mutable)
            key: Decryption key in hex format (optional)
            kid: Key ID in hex format
            debug: Enable debug logging
        """
        self.data = data
        self.data_size = len(data)
        self.offset = 0
        self.debug = debug

        # Decryption info
        self.kid: Optional[bytes] = bytes.fromhex(kid) if kid else None

        # Store key (optional)
        self.key: Optional[bytes] = None
        if key:
            key_bytes = bytes.fromhex(key)
            if len(key_bytes) != 16:
                raise ValueError(f"Key must be exactly 16 bytes, got {len(key_bytes)}")
            self.key = key_bytes

        # Sample tracking - reset for each fragment (not cumulative)
        self.samples: Dict[int, SampleInfo] = {}
        self.default_sample_size = 0
        self.default_iv_size = 8
        self.former_type: Optional[str] = None
        self.pssh_boxes: List[str] = []

        # Track container boxes and their data sizes
        self.container_boxes = {
            "moov",
            "trak",
            "mdia",
            "minf",
            "stbl",
            "moof",
            "traf",
            "mvex",
            "mfra",
            "stsd",
            "encv",
            "enca",
            "avc3",
            "sinf",
            "schi",
            "dinf",
        }

        self.replace_types = {
            "pssh",
            "senc",
            "sinf",
            "enca",
            "encv",
            "schm",
            "schi",
            "tenc",
            "frma",
            "saiz",
            "saio",
            "sbgp",
            "sgpd",
        }

    def parse(self) -> bool:
        """Parse MP4 boxes and return success status"""
        try:
            while self.offset < self.data_size:
                if not self._parse_box():
                    return False
            return True
        except Exception as e:
            if self.debug:
                logger.error(f"Parse error at offset {self.offset}: {e}")
            return False

    def _parse_box(self) -> bool:
        """Parse a single MP4 box - with more lenient validation"""
        if self.offset + 8 > self.data_size:
            return False

        # Read box header
        size_bytes = self.data[self.offset : self.offset + 4]
        type_bytes = self.data[self.offset + 4 : self.offset + 8]

        if not size_bytes or not type_bytes:
            return False

        size = struct.unpack(">I", size_bytes)[0]

        # Validate box type - be more lenient
        try:
            box_type = type_bytes.decode("ascii")
            # Check if it's printable ASCII, but don't fail on some special chars
            if not all(32 <= ord(c) <= 126 for c in box_type):
                # Instead of failing, just treat it as unknown box
                if self.debug:
                    logger.debug(
                        f"Non-printable box type at offset {self.offset}: {type_bytes.hex()}"
                    )
                # Skip this box
                if size > 0 and size != 1:
                    self.offset += size
                else:
                    self.offset += 8
                return True  # Continue parsing, don't fail
        except UnicodeDecodeError:
            if self.debug:
                logger.debug(f"Non-ASCII box type at offset {self.offset}: {type_bytes.hex()}")
            # Skip this box
            if size > 0 and size != 1:
                self.offset += size
            else:
                self.offset += 8
            return True  # Continue parsing

        box_start = self.offset
        self.offset += 8

        # Handle large size
        if size == 1:
            if self.offset + 8 > self.data_size:
                return False
            size_bytes = self.data[self.offset : self.offset + 8]
            size = struct.unpack(">Q", size_bytes)[0]
            self.offset += 8

        if size == 0:
            # Box extends to end of file
            size = self.data_size - box_start

        # Validate box size
        box_end = box_start + size
        if box_end > self.data_size:
            if self.debug:
                logger.warning(f"Box {box_type} extends beyond data: {box_end} > {self.data_size}")
            # Don't fail - just skip to end
            self.offset = self.data_size
            return True

        if self.debug:
            logger.debug(f"Parsing box: {box_type} at {box_start}, size: {size}")

        # Handle specific box types
        handler = getattr(self, f"_parse_{box_type}", None)
        if handler:
            try:
                success = handler(box_start, size)
                if not success:
                    if self.debug:
                        logger.warning(f"Handler for {box_type} returned False, continuing anyway")
                    # Don't fail the entire parse - just skip to end of box
                    self.offset = box_end
            except Exception as e:
                if self.debug:
                    logger.error(f"Error parsing {box_type}: {e}")
                # Don't fail - skip to end of box
                self.offset = box_end
        elif box_type in self.container_boxes:
            # Container boxes contain other boxes - continue parsing inside
            pass
        else:
            # Skip unknown box
            self.offset = box_end

        # Replace box type if needed (after parsing to preserve data)
        if box_type in self.replace_types:
            self._replace_box_type(box_start, box_type)

        # Ensure we're at the right position
        if self.offset > box_end:
            if self.debug:
                logger.warning(f"Overshot box {box_type}, correcting offset")
            self.offset = box_end
        elif self.offset < box_end:
            # This is normal for container boxes
            pass

        return True

    def _parse_senc(self, box_start: int, box_size: int) -> bool:
        """Parse Sample Encryption (senc) box"""
        if self.offset + 4 > self.data_size:
            return False

        # Read version and flags
        version_flags = struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
        version = (version_flags >> 24) & 0xFF
        flags = version_flags & 0xFFFFFF
        self.offset += 4

        # Read sample count
        if self.offset + 4 > self.data_size:
            return False
        sample_count = struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
        self.offset += 4

        if self.debug:
            logger.debug(f"SENC: version={version}, flags={flags:#x}, samples={sample_count}")

        # Determine IV size
        iv_size = self.default_iv_size
        if version > 0:
            if self.offset + 1 > self.data_size:
                return False
            iv_size = self.data[self.offset]
            self.offset += 1

        # Validate IV size
        if iv_size not in (0, 8, 16):
            if self.debug:
                logger.warning(f"Unusual IV size: {iv_size}")

        # Process samples
        for i in range(sample_count):
            # Get or create sample
            if i not in self.samples:
                self.samples[i] = SampleInfo(index=i)
            sample = self.samples[i]

            # Read IV if not already present and size > 0
            if sample.iv is None and iv_size > 0:
                if self.offset + iv_size > self.data_size:
                    return False
                # Store IV as-is (8 or 16 bytes) - don't expand here
                sample.iv = bytes(self.data[self.offset : self.offset + iv_size])
                self.offset += iv_size

            # Process subsamples if present
            if flags & 0x02:
                if self.offset + 2 > self.data_size:
                    return False
                subsample_count = struct.unpack(">H", self.data[self.offset : self.offset + 2])[0]
                self.offset += 2

                if not sample.subsamples:
                    for j in range(subsample_count):
                        if self.offset + 6 > self.data_size:
                            return False
                        clear, encrypted = struct.unpack(
                            ">HI", self.data[self.offset : self.offset + 6]
                        )
                        self.offset += 6
                        sample.subsamples.append({"clear": clear, "encrypted": encrypted})
                else:
                    # Skip if already processed
                    self.offset += subsample_count * 6

        return True

    def _parse_trun(self, box_start: int, box_size: int) -> bool:
        """Parse Track Fragment Run (trun) box"""
        if self.offset + 4 > self.data_size:
            return False

        # Read version and flags
        version_flags = struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
        version = (version_flags >> 24) & 0xFF
        flags = version_flags & 0xFFFFFF
        self.offset += 4

        # Read sample count
        if self.offset + 4 > self.data_size:
            return False
        sample_count = struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
        self.offset += 4

        if self.debug:
            logger.debug(f"TRUN: version={version}, flags={flags:#x}, samples={sample_count}")

        # Parse optional fields
        if flags & 0x000001:  # data-offset-present
            if self.offset + 4 > self.data_size:
                return False
            struct.unpack(">i", self.data[self.offset : self.offset + 4])[0]
            self.offset += 4

        if flags & 0x000004:  # first-sample-flags-present
            if self.offset + 4 > self.data_size:
                return False
            self.offset += 4

        # Determine which sample fields are present
        has_duration = bool(flags & 0x000100)
        has_size = bool(flags & 0x000200)
        has_flags = bool(flags & 0x000400)
        has_composition = bool(flags & 0x000800)

        # Parse samples - use simple indexing (0, 1, 2, ...) not cumulative
        for i in range(sample_count):
            if i not in self.samples:
                self.samples[i] = SampleInfo(index=i)
            sample = self.samples[i]

            # Read sample fields
            if has_duration:
                if self.offset + 4 > self.data_size:
                    return False
                sample.duration = struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
                self.offset += 4

            if has_size:
                if self.offset + 4 > self.data_size:
                    return False
                sample.full_encrypted_size = struct.unpack(
                    ">I", self.data[self.offset : self.offset + 4]
                )[0]
                self.offset += 4

            if has_flags:
                if self.offset + 4 > self.data_size:
                    return False
                sample.flags = struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
                self.offset += 4

            if has_composition:
                if self.offset + 4 > self.data_size:
                    return False
                composition = struct.unpack(">i", self.data[self.offset : self.offset + 4])[0]
                sample.composition_offset = composition
                self.offset += 4

        # Apply default sample size to any samples without size
        if self.default_sample_size > 0:
            for i in range(sample_count):
                if i in self.samples and self.samples[i].full_encrypted_size is None:
                    self.samples[i].full_encrypted_size = self.default_sample_size

        return True

    def _parse_tfhd(self, box_start: int, box_size: int) -> bool:
        """Parse Track Fragment Header (tfhd) box"""
        if self.offset + 8 > self.data_size:
            return False

        # Read version, flags, and track ID
        version_flags = struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
        track_id = struct.unpack(">I", self.data[self.offset + 4 : self.offset + 8])[0]
        version = (version_flags >> 24) & 0xFF
        flags = version_flags & 0xFFFFFF
        self.offset += 8

        if self.debug:
            logger.debug(f"TFHD: version={version}, flags={flags:#x}, track_id={track_id}")

        # Parse optional fields
        if flags & 0x000001:  # base-data-offset-present
            if self.offset + 8 > self.data_size:
                return False
            self.offset += 8

        if flags & 0x000002:  # sample-description-index-present
            if self.offset + 4 > self.data_size:
                return False
            self.offset += 4

        if flags & 0x000008:  # default-sample-duration-present
            if self.offset + 4 > self.data_size:
                return False
            self.offset += 4

        if flags & 0x000010:  # default-sample-size-present
            if self.offset + 4 > self.data_size:
                return False
            self.default_sample_size = struct.unpack(
                ">I", self.data[self.offset : self.offset + 4]
            )[0]
            self.offset += 4

        if flags & 0x000020:  # default-sample-flags-present
            if self.offset + 4 > self.data_size:
                return False
            self.offset += 4

        return True

    def _parse_mdat(self, box_start: int, box_size: int) -> bool:
        """Process Media Data (mdat) box - apply decryption if needed"""
        # mdat data starts after the 8-byte header
        mdat_data_offset = box_start + 8

        if self.debug:
            logger.debug(
                f"Processing MDAT: offset={mdat_data_offset}, "
                f"size={box_size - 8}, has_key={bool(self.key)}, "
                f"samples={len(self.samples)}"
            )

        # Only decrypt if we have key and samples
        if self.key and self.samples:
            # Convert samples to dict format for decryptor
            samples_dict: List[Dict[str, Any]] = []
            for i in sorted(self.samples.keys()):
                sample = self.samples[i]
                sample_dict: Dict[str, Any] = {"index": i}
                if sample.iv:
                    sample_dict["iv"] = sample.iv  # Pass IV as-is (8 or 16 bytes)
                if sample.subsamples:
                    sample_dict["subsamples"] = sample.subsamples
                if sample.full_encrypted_size:
                    sample_dict["full_encrypted_size"] = sample.full_encrypted_size
                samples_dict.append(sample_dict)

            # Create and use decryptor
            from .cenc_decryptor import CENCDecryptor

            try:
                decryptor = CENCDecryptor(self.data, self.key, samples_dict, self.debug)
                success = decryptor.decrypt(mdat_data_offset)

                if self.debug:
                    logger.debug(f"Decryption {'succeeded' if success else 'failed'}")

                if not success:
                    return False
            except ValueError as e:
                if self.debug:
                    logger.error(f"Decryptor initialization failed: {e}")
                return False

        elif self.debug:
            if self.samples and not self.key:
                logger.debug("Skipping MDAT decryption - no key provided")
            elif not self.samples and self.key:
                logger.debug("Skipping MDAT decryption - no samples available")
            elif not self.samples and not self.key:
                logger.debug("Skipping MDAT processing - no key and no samples")

        # Clear processed samples to free memory (like PHP does)
        self.samples.clear()

        return True

    def _parse_tenc(self, box_start: int, box_size: int) -> bool:
        """Parse Track Encryption (tenc) box - matches CENC specification"""

        # tenc box structure (ISO/IEC 23001-7):
        # - 4 bytes: version (1 byte) + flags (3 bytes)
        # - 1 byte: reserved (must be 0)
        # - 1 byte: default_crypt_byte_block (version >= 1) or reserved (version 0)
        # - 1 byte: default_skip_byte_block (version >= 1) or reserved (version 0)
        # - 1 byte: default_isProtected (1 = encrypted, 0 = not encrypted)
        # - 1 byte: default_Per_Sample_IV_Size
        # - 16 bytes: default_KID
        # Total: 24 bytes (minimum)

        data_size = box_size - 8  # Subtract box header
        if self.offset + data_size > self.data_size:
            return False

        # For tenc, we expect at least 24 bytes
        if data_size < 24:
            if self.debug:
                logger.warning(f"TENC box too small: {data_size} bytes")
            self.offset += data_size
            return True  # Don't fail, just skip

        tenc_data = self.data[self.offset : self.offset + data_size]

        # Parse version and flags (4 bytes)
        version_flags = struct.unpack(">I", tenc_data[:4])[0]
        version = (version_flags >> 24) & 0xFF
        flags = version_flags & 0xFFFFFF

        # Skip reserved byte at offset 4
        reserved = tenc_data[4]

        # Version-dependent fields at offsets 5-6
        if version >= 1:
            crypt_byte_block = tenc_data[5]
            skip_byte_block = tenc_data[6]
        else:
            # Version 0: these are reserved
            crypt_byte_block = 0
            skip_byte_block = 0

        # Common fields
        is_protected = tenc_data[7]
        iv_size = tenc_data[8]
        default_kid = tenc_data[9:25]  # 16 bytes from offset 9-24

        # Store IV size for SENC parsing
        self.default_iv_size = iv_size

        if self.debug:
            logger.debug(
                f"TENC: version={version}, flags={flags:#x}, "
                f"protected={is_protected}, iv_size={iv_size}, "
                f"kid={binascii.hexlify(default_kid).decode()}"
            )

        # Store KID if not already set
        if self.kid is None:
            self.kid = default_kid

        self.offset += data_size
        return True

    def _parse_pssh(self, box_start: int, box_size: int) -> bool:
        """Parse Protection System Specific Header (pssh) box"""
        if box_start + box_size > self.data_size:
            return False

        pssh_data = self.data[box_start : box_start + box_size]
        pssh_base64 = base64.b64encode(pssh_data).decode("ascii")
        self.pssh_boxes.append(pssh_base64)

        if self.debug:
            logger.debug(f"PSSH box found: {pssh_base64[:50]}...")

        return True

    def _replace_box_type(self, box_start: int, original_type: str):
        """Replace box type with 'free' or former type"""
        if box_start + 8 > self.data_size:
            return

        # For encv/enca, use the former type; otherwise use 'free'
        if original_type in ("encv", "enca") and self.former_type:
            new_type = self.former_type
        else:
            new_type = "free"

        # Replace type (4 bytes at offset +4)
        self.data[box_start + 4 : box_start + 8] = new_type.encode("ascii")

        if self.debug:
            logger.debug(f"Replaced '{original_type}' box with '{new_type}' at offset {box_start}")

    def _parse_frma(self, box_start: int, box_size: int) -> bool:
        """Parse Original Format (frma) box"""
        if self.offset + 4 > self.data_size:
            return False

        original_format = self.data[self.offset : self.offset + 4].decode("ascii", errors="replace")
        self.offset += 4
        self.former_type = original_format

        if self.debug:
            logger.debug(f"FRMA: original format = {original_format}")

        return True

    def _parse_schm(self, box_start: int, box_size: int) -> bool:
        """Parse Scheme Type (schm) box"""
        if self.offset + 8 > self.data_size:
            return False

        struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
        scheme_type = self.data[self.offset + 4 : self.offset + 8].decode("ascii", errors="replace")

        if self.debug:
            logger.debug(f"SCHM: scheme type = {scheme_type}")

        return True

    def _parse_saiz(self, box_start: int, box_size: int) -> bool:
        """Parse Sample Auxiliary Information Sizes (saiz) box"""
        return True

    def _parse_saio(self, box_start: int, box_size: int) -> bool:
        """Parse Sample Auxiliary Information Offsets (saio) box"""
        return True

    def _parse_sbgp(self, box_start: int, box_size: int) -> bool:
        """Parse Sample to Group (sbgp) box"""
        return True

    def _parse_mvhd(self, box_start: int, box_size: int) -> bool:
        """Parse Movie Header box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_tkhd(self, box_start: int, box_size: int) -> bool:
        """Parse Track Header box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_mdhd(self, box_start: int, box_size: int) -> bool:
        """Parse Media Header box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_hdlr(self, box_start: int, box_size: int) -> bool:
        """Parse Handler Reference box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_vmhd(self, box_start: int, box_size: int) -> bool:
        """Parse Video Media Header box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_dref(self, box_start: int, box_size: int) -> bool:
        """Parse Data Reference box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_stts(self, box_start: int, box_size: int) -> bool:
        """Parse Time-to-Sample box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_stsc(self, box_start: int, box_size: int) -> bool:
        """Parse Sample-to-Chunk box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_stsz(self, box_start: int, box_size: int) -> bool:
        """Parse Sample Size box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_stco(self, box_start: int, box_size: int) -> bool:
        """Parse Chunk Offset box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_trex(self, box_start: int, box_size: int) -> bool:
        """Parse Track Extends box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_avcC(self, box_start: int, box_size: int) -> bool:
        """Parse AVC Configuration box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_btrt(self, box_start: int, box_size: int) -> bool:
        """Parse Bit Rate box - just skip it"""
        self.offset = box_start + box_size
        return True

    def _parse_sgpd(self, box_start: int, box_size: int) -> bool:
        """Parse Sample Group Description (sgpd) box"""
        return True

    def get_samples(self) -> List[Dict[str, Any]]:
        """Get samples in dict format for decryptor"""
        samples: List[Dict[str, Any]] = []
        for i in sorted(self.samples.keys()):
            sample = self.samples[i]
            sample_dict: Dict[str, Any] = {"index": i}
            if sample.iv:
                sample_dict["iv"] = sample.iv
            if sample.subsamples:
                sample_dict["subsamples"] = sample.subsamples
            if sample.full_encrypted_size:
                sample_dict["full_encrypted_size"] = sample.full_encrypted_size
            samples.append(sample_dict)
        return samples

    def get_pssh_boxes(self) -> List[str]:
        """Get PSSH boxes as base64 strings"""
        return self.pssh_boxes

    def get_kid(self) -> Optional[str]:
        """Get KID as hex string"""
        if self.kid:
            return binascii.hexlify(self.kid).decode()
        return None
