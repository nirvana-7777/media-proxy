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
        key: Optional[str] = None,
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
        self.pssh_boxes: List[str] = []

        # Track enca/encv boxes waiting for frma
        self.pending_enc_boxes: List[int] = []  # List of box_start offsets

        # Track container boxes (boxes that contain other boxes)
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
            "sinf",
            "schi",
            "dinf",
        }

        # Sample entry boxes need special handling (skip fixed fields, then parse children)
        self.sample_entry_boxes = {
            "encv",
            "enca",
            "avc1",
            "hvc1",
            "mp4a",
            "ac-3",
            "ec-3",
            "opus",
            "avc3",
        }

        # Boxes to replace with 'free'
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

            # Any remaining enca/encv boxes without frma become 'free'
            for box_start in self.pending_enc_boxes:
                self._write_box_type(box_start, "free")
                if self.debug:
                    logger.debug(f"Replaced pending enc box at {box_start} with 'free'")

            return True
        except Exception as e:
            if self.debug:
                logger.error(f"Parse error at offset {self.offset}: {e}")
            return False

    def _parse_box(self) -> bool:
        """Parse a single MP4 box"""
        if self.offset + 8 > self.data_size:
            return False

        # Read box header
        size_bytes = self.data[self.offset : self.offset + 4]
        type_bytes = self.data[self.offset + 4 : self.offset + 8]

        if not size_bytes or not type_bytes:
            return False

        size = struct.unpack(">I", size_bytes)[0]

        # Validate box type
        try:
            box_type = type_bytes.decode("ascii")
            if not all(32 <= ord(c) <= 126 for c in box_type):
                if self.debug:
                    logger.debug(
                        f"Non-printable box type at offset {self.offset}: {type_bytes.hex()}"
                    )
                if size > 0 and size != 1:
                    self.offset += size
                else:
                    self.offset += 8
                return True
        except UnicodeDecodeError:
            if self.debug:
                logger.debug(f"Non-ASCII box type at offset {self.offset}: {type_bytes.hex()}")
            if size > 0 and size != 1:
                self.offset += size
            else:
                self.offset += 8
            return True

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
            size = self.data_size - box_start

        # Validate box size
        box_end = box_start + size
        if box_end > self.data_size:
            if self.debug:
                logger.warning(f"Box {box_type} extends beyond data: {box_end} > {self.data_size}")
            self.offset = self.data_size
            return True

        if self.debug:
            logger.debug(f">>> Found box: '{box_type}' at offset {box_start}, size={size}")

        # Determine if we should skip parsing (replaced boxes)
        should_replace = box_type in self.replace_types
        skip_parsing = False

        # Handle replacement logic
        if should_replace:
            if self.debug:
                logger.debug(f"Box {box_type} is in replace_types")

            if box_type in ("enca", "encv"):
                # Remember this box location for later frma replacement
                self.pending_enc_boxes.append(box_start)
                # Don't skip parsing - we need to parse children
            elif box_type == "frma":
                # Parse frma and replace pending enc boxes
                success = self._parse_frma_with_replacement(box_start, size)
                if not success:
                    self.offset = box_end
                skip_parsing = True  # Already handled
            elif box_type == "senc":
                # Parse senc first to extract sample data, then replace
                # Handler will parse the data, then we replace and skip
                pass  # Let handler run, then replace after
            else:
                # Immediate replacement for other boxes (pssh, tenc, saiz, etc.)
                self._write_box_type(box_start, "free")
                if self.debug:
                    logger.debug(f"Replaced {box_type} with 'free' - skipping to end")
                self.offset = box_end
                skip_parsing = True

        # Parse the box if not skipped
        if not skip_parsing:
            # Check for specific handlers first
            handler = getattr(self, f"_parse_{box_type}", None)
            if handler:
                if self.debug:
                    logger.debug(f"Found handler for {box_type}: {handler.__name__}")
                try:
                    success = handler(box_start, size)
                    if not success:
                        if self.debug:
                            logger.warning(f"Handler for {box_type} returned False")
                        self.offset = box_end
                except Exception as e:
                    if self.debug:
                        logger.error(f"Error parsing {box_type}: {e}")
                    self.offset = box_end

                # After parsing senc, replace it with 'free'
                if box_type == "senc" and box_type in self.replace_types:
                    self._write_box_type(box_start, "free")
                    if self.debug:
                        logger.debug(f"Replaced {box_type} with 'free' after extracting data")

            elif box_type in self.sample_entry_boxes:
                # Sample entry boxes: skip fixed fields, then parse children
                if self.debug:
                    logger.debug(
                        f"Box {box_type} is a sample entry - parsing with special handling"
                    )
                try:
                    success = self._parse_sample_entry(box_type, box_start, size)
                    if not success:
                        self.offset = box_end
                except Exception as e:
                    if self.debug:
                        logger.error(f"Error parsing sample entry {box_type}: {e}")
                    self.offset = box_end
            elif box_type in self.container_boxes:
                if self.debug:
                    logger.debug(f"Box {box_type} is a container - parsing children")
                # Container boxes: continue parsing inside
                pass
            else:
                if self.debug:
                    logger.debug(f"Unknown box {box_type} - skipping to offset {box_end}")
                self.offset = box_end

        # Ensure we're at the right position
        if self.offset > box_end:
            if self.debug:
                logger.warning(f"Overshot box {box_type}, correcting offset")
            self.offset = box_end
        elif (
            self.offset < box_end
            and box_type not in self.container_boxes
            and box_type not in self.sample_entry_boxes
        ):
            if self.debug:
                logger.debug(f"Offset {self.offset} < box_end {box_end} for leaf box {box_type}")

        return True

    def _parse_sample_entry(self, box_type: str, box_start: int, box_size: int) -> bool:
        """Parse sample entry boxes (encv, enca, avc1, mp4a, etc.)"""
        # Skip standard sample entry header (6 reserved + data_reference_index)
        header_size = (6 * 4) + 2  # 24 + 2 = 26 bytes
        self.offset += header_size

        if self.debug:
            logger.debug(f"Sample entry {box_type}: skipped {header_size} bytes of header")

        # For ENCRYPTED entries, skip any additional fields and go straight to children
        if box_type in ("enca", "encv"):
            if self.debug:
                logger.debug(f"Encrypted entry {box_type} - skipping type-specific fields")
            # Don't try to parse audio/video fields
            # Go straight to parsing child boxes
            box_end = box_start + box_size
            while self.offset < box_end:
                if not self._parse_box():
                    return False
            return True

        # For non-encrypted entries, use existing logic
        if box_type in ("encv", "avc1", "avc3", "hvc1"):
            return self._parse_visual_sample_entry(box_start, box_size)
        elif box_type in ("enca", "mp4a", "ac-3", "ec-3", "opus"):
            return self._parse_audio_sample_entry(box_start, box_size)
        else:
            self.offset = box_start + box_size
            return True

    def _parse_visual_sample_entry(self, box_start: int, box_size: int) -> bool:
        """Parse visual sample entry fields and then child boxes"""
        # Visual sample entry structure after standard header:
        # - pre_defined (2) + reserved (2) = 4 bytes
        # - width (2) + height (2) = 4 bytes
        # - horiz resolution (4) + vert resolution (4) = 8 bytes
        # - reserved (4)
        # - frame_count (2)
        # - compressor name (32 bytes: 1 length + 31 data)
        # - depth (2) + pre_defined (2) = 4 bytes
        # Total: 58 bytes

        if self.offset + 58 > self.data_size:
            return False

        self.offset += 4  # pre_defined + reserved
        width = struct.unpack(">H", self.data[self.offset : self.offset + 2])[0]
        height = struct.unpack(">H", self.data[self.offset + 2 : self.offset + 4])[0]
        self.offset += 4

        if self.debug:
            logger.debug(f"Visual sample entry: width={width}, height={height}")

        self.offset += 8  # resolutions
        self.offset += 4  # reserved
        self.offset += 2  # frame_count
        self.offset += 32  # compressor name (1 + 31)
        self.offset += 4  # depth + pre_defined

        # Now parse child boxes (avcC, btrt, sinf, etc.)
        box_end = box_start + box_size
        while self.offset < box_end:
            if not self._parse_box():
                return False

        return True

    def _parse_audio_sample_entry(self, box_start: int, box_size: int) -> bool:
        """Parse audio sample entry fields and then child boxes"""
        # Audio sample entry structure after standard header:
        # - reserved (8 bytes)
        # - channel count (2)
        # - sample size (2)
        # - pre_defined (2) + reserved (2) = 4 bytes
        # - sample rate (4) - stored as 16.16 fixed point
        # Total: 20 bytes (for version 0)

        if self.offset + 20 > self.data_size:
            return False

        self.offset += 8  # reserved
        channel_count = struct.unpack(">H", self.data[self.offset : self.offset + 2])[0]
        sample_size = struct.unpack(">H", self.data[self.offset + 2 : self.offset + 4])[0]
        self.offset += 4
        self.offset += 4  # pre_defined + reserved
        sample_rate = struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
        self.offset += 4

        if self.debug:
            logger.debug(
                f"Audio sample entry: channels={channel_count}, "
                f"sample_size={sample_size}, rate={sample_rate >> 16}"
            )

        # Now parse child boxes (esds, sinf, etc.)
        box_end = box_start + box_size
        while self.offset < box_end:
            if not self._parse_box():
                return False

        return True

    def _write_box_type(self, box_start: int, new_type: str):
        """Write new box type to data buffer"""
        if box_start + 8 <= self.data_size:
            self.data[box_start + 4 : box_start + 8] = new_type.encode("ascii")
            if self.debug:
                logger.debug(f"Wrote box type '{new_type}' at offset {box_start}")

    def _parse_frma_with_replacement(self, frma_box_start: int, frma_box_size: int) -> bool:
        """Parse frma box and replace any pending enca/encv boxes with its content"""
        if self.offset + 4 > self.data_size:
            return False

        # Read the original format from frma box
        original_format = self.data[self.offset : self.offset + 4].decode("ascii", errors="replace")
        self.offset += 4

        if self.debug:
            logger.debug(f"FRMA: original format = {original_format}")

        # Replace ALL pending enca/encv boxes with this format
        for box_start in self.pending_enc_boxes:
            self._write_box_type(box_start, original_format)

        # Clear the pending list
        self.pending_enc_boxes.clear()

        # Replace frma box itself with 'free'
        self._write_box_type(frma_box_start, "free")

        return True

    def _parse_senc(self, box_start: int, box_size: int) -> bool:
        """Parse Sample Encryption (senc) box - extract data then it will be replaced"""
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

        if self.debug:
            logger.debug(f"SENC: Extracted data from {sample_count} samples")

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

        # Parse samples
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
                    sample_dict["iv"] = sample.iv
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

        # Clear processed samples to free memory
        self.samples.clear()

        return True

    def _parse_tenc(self, box_start: int, box_size: int) -> bool:
        """Parse Track Encryption (tenc) box"""
        data_size = box_size - 8
        if self.offset + data_size > self.data_size:
            return False

        if data_size < 24:
            if self.debug:
                logger.warning(f"TENC box too small: {data_size} bytes")
            self.offset += data_size
            return True

        tenc_data = self.data[self.offset : self.offset + data_size]

        # Parse version and flags
        version_flags = struct.unpack(">I", tenc_data[:4])[0]
        version = (version_flags >> 24) & 0xFF
        flags = version_flags & 0xFFFFFF

        # Extract fields
        is_protected = tenc_data[7]
        iv_size = tenc_data[8]
        default_kid = tenc_data[9:25]

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

    def _parse_schm(self, box_start: int, box_size: int) -> bool:
        """Parse Scheme Type (schm) box"""
        if self.offset + 8 > self.data_size:
            return False

        struct.unpack(">I", self.data[self.offset : self.offset + 4])[0]
        scheme_type = self.data[self.offset + 4 : self.offset + 8].decode("ascii", errors="replace")

        if self.debug:
            logger.debug(f"SCHM: scheme type = {scheme_type}")

        return True

    # Simple skip handlers for boxes that don't need processing
    def _parse_saiz(self, box_start: int, box_size: int) -> bool:
        """Parse Sample Auxiliary Information Sizes (saiz) box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_saio(self, box_start: int, box_size: int) -> bool:
        """Parse Sample Auxiliary Information Offsets (saio) box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_sbgp(self, box_start: int, box_size: int) -> bool:
        """Parse Sample to Group (sbgp) box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_sgpd(self, box_start: int, box_size: int) -> bool:
        """Parse Sample Group Description (sgpd) box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_mvhd(self, box_start: int, box_size: int) -> bool:
        """Parse Movie Header box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_tkhd(self, box_start: int, box_size: int) -> bool:
        """Parse Track Header box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_mdhd(self, box_start: int, box_size: int) -> bool:
        """Parse Media Header box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_hdlr(self, box_start: int, box_size: int) -> bool:
        """Parse Handler Reference box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_vmhd(self, box_start: int, box_size: int) -> bool:
        """Parse Video Media Header box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_smhd(self, box_start: int, box_size: int) -> bool:
        """Parse Sound Media Header box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_dref(self, box_start: int, box_size: int) -> bool:
        """Parse Data Reference box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_stts(self, box_start: int, box_size: int) -> bool:
        """Parse Time-to-Sample box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_stsc(self, box_start: int, box_size: int) -> bool:
        """Parse Sample-to-Chunk box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_stsz(self, box_start: int, box_size: int) -> bool:
        """Parse Sample Size box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_stco(self, box_start: int, box_size: int) -> bool:
        """Parse Chunk Offset box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_trex(self, box_start: int, box_size: int) -> bool:
        """Parse Track Extends box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_avcC(self, box_start: int, box_size: int) -> bool:
        """Parse AVC Configuration box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_hvcC(self, box_start: int, box_size: int) -> bool:
        """Parse HEVC Configuration box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_btrt(self, box_start: int, box_size: int) -> bool:
        """Parse Bit Rate box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_esds(self, box_start: int, box_size: int) -> bool:
        """Parse Elementary Stream Descriptor box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_stss(self, box_start: int, box_size: int) -> bool:
        """Parse Sync Sample Box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_mfhd(self, box_start: int, box_size: int) -> bool:
        """Parse Movie Fragment Header Box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_emsg(self, box_start: int, box_size: int) -> bool:
        """Parse Event Message box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_sidx(self, box_start: int, box_size: int) -> bool:
        """Parse Segment Index box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_tfdt(self, box_start: int, box_size: int) -> bool:
        """Parse Track Fragment Decode Time box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_uuid(self, box_start: int, box_size: int) -> bool:
        """Parse UUID box - skip to end"""
        self.offset = box_start + box_size
        return True

    def _parse_dec3(self, box_start: int, box_size: int) -> bool:
        """Parse EC-3 Specific Configuration box - skip to end"""
        self.offset = box_start + box_size
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
