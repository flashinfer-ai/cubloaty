"""Parser for the CUDA fatbinary container format.

The layout below was worked out from fatbins produced by the public CUDA
toolkit (nvcc --compress-mode=none/speed/size/..., -lineinfo, -G, -dlto) and
cross-checked against cuobjdump; FATBIN_MAGIC and the host section names are
in the public fatbinary_section.h.

A fatbin section (.nv_fatbin / __nv_relfatbin) holds one or more fatbins,
typically one per translation unit, each aligned and zero-padded:

    header (16 bytes): u32 magic (0xBA55ED50), u16 version, u16 header size,
                       u64 size of the entries that follow

    entry header (header size bytes, then the payload):
        0x00 u16 kind (1 = PTX, 2 = ELF/cubin, 8 = LTO-IR)
        0x04 u32 header size (the payload starts right after it)
        0x08 u64 payload size, padded to 8 bytes
        0x10 u32 compressed payload size (0 if stored uncompressed)
        0x1c u32 SM number (e.g. 90, 100)
        0x20 u32 offset / 0x24 u32 length of an optional identifier string
        0x28 u64 flags
        0x38 u64 uncompressed payload size (0 if stored uncompressed)

The next entry starts at header size + padded payload size. Compressed
payloads are recognized by their content (ZSTD and LZ4 frames and zlib
streams carry magic numbers; anything else is tried as a raw LZ4 block).
"""

import functools
import struct
import zlib
from dataclasses import dataclass

FATBIN_MAGIC = 0xBA55ED50
FATBIN_MAGIC_BYTES = struct.pack("<I", FATBIN_MAGIC)
FATBIN_HEADER = struct.Struct("<IHHQ")
ENTRY_HEADER = struct.Struct("<HHIQIIIIIIQQQ")

KIND_PTX = 1
KIND_ELF = 2
KIND_LTOIR = 8
KIND_NAMES = {KIND_PTX: "ptx", KIND_ELF: "elf", KIND_LTOIR: "ltoir"}

FLAG_ARCH_SPECIFIC = 0x100000  # "a" targets, e.g. sm_90a
FLAG_FAMILY_SPECIFIC = 0x200000  # "f" targets, e.g. sm_100f

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
LZ4_FRAME_MAGIC = b"\x04\x22\x4d\x18"


class FatbinError(Exception):
    pass


# --------------------------------------------------------------------------
# Decompression (public formats)


def _lz4_block_into(src, dst):
    """Decode one LZ4 block from `src`, appending to bytearray `dst`.
    Matches may reach back into earlier blocks already in `dst`."""
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        dst += src[i : i + lit]
        i += lit
        if i >= n:
            break
        offset = src[i] | (src[i + 1] << 8)
        i += 2
        if offset == 0 or offset > len(dst):
            raise FatbinError("corrupt LZ4 stream")
        match = token & 15
        if match == 15:
            while True:
                b = src[i]
                i += 1
                match += b
                if b != 255:
                    break
        match += 4
        start = len(dst) - offset
        if offset >= match:
            dst += dst[start : start + match]
        else:  # overlapping copy repeats the last `offset` bytes
            chunk = bytes(dst[start:])
            dst += (chunk * (match // offset + 1))[:match]


def _lz4_frame_into(src, dst):
    """Pure-Python LZ4 frame decoder (fallback when lz4 is not installed)"""
    if struct.unpack_from("<I", src, 0)[0] != 0x184D2204:
        raise FatbinError("bad LZ4 frame magic")
    flg = src[4]
    i = 7 + (8 if flg & 0x08 else 0) + (4 if flg & 0x01 else 0)
    while True:
        size = struct.unpack_from("<I", src, i)[0]
        i += 4
        if size == 0:
            break
        block = src[i : i + (size & 0x7FFFFFFF)]
        i += len(block)
        if size & 0x80000000:
            dst += block
        else:
            _lz4_block_into(block, dst)
        if flg & 0x10:
            i += 4  # block checksum


def _lz4_block_py(data, size):
    dst = bytearray()
    _lz4_block_into(data, dst)
    return bytes(dst)


def _lz4_frame_py(data, size):
    dst = bytearray()
    _lz4_frame_into(data, dst)
    return bytes(dst)


def _zlib(data, size):
    return zlib.decompressobj().decompress(data)


@functools.lru_cache(maxsize=None)
def get_decompressor(name):
    """Return f(data: bytes, size: int) -> bytes, or None if unavailable.
    Resolved once: Python does not cache failed imports."""
    if name == "zstd":
        try:
            from compression import zstd  # Python 3.14+

            return lambda data, size: zstd.decompress(data)
        except ImportError:
            pass
        try:
            import zstandard
        except ImportError:
            return None
        dctx = zstandard.ZstdDecompressor()
        return lambda data, size: dctx.decompress(data, max_output_size=size)
    if name in ("lz4", "lz4frame"):
        try:
            import lz4.block
            import lz4.frame
        except ImportError:
            return _lz4_block_py if name == "lz4" else _lz4_frame_py
        if name == "lz4":
            return lambda data, size: lz4.block.decompress(data, uncompressed_size=size)
        return lambda data, size: lz4.frame.decompress(data)
    if name == "zlib":
        return _zlib
    return None


# --------------------------------------------------------------------------
# Container


def sniff_compression(data):
    if data[:4] == ZSTD_MAGIC:
        return "zstd"
    if data[:4] == LZ4_FRAME_MAGIC:
        return "lz4frame"
    if len(data) >= 2 and data[0] & 0x0F == 8 and (data[0] << 8 | data[1]) % 31 == 0:
        return "zlib"
    return "lz4"


@dataclass
class FatbinEntry:
    kind: int
    offset: int  # absolute offset of the entry header in the backing buffer
    header_size: int
    padded_size: int  # stored payload size, padded to 8
    compressed_size: int
    uncompressed_size: int
    arch_number: int
    flags: int
    identifier: str
    _blob: object  # backing buffer the payload is sliced from

    @property
    def kind_name(self):
        return KIND_NAMES.get(self.kind, f"kind{self.kind:#x}")

    @property
    def file_size(self):
        """Bytes this entry occupies in the file: header + padded payload"""
        return self.header_size + self.padded_size

    @property
    def compressed(self):
        return self.uncompressed_size != 0

    @property
    def stored_size(self):
        """Payload bytes as stored (compressed length if compressed)"""
        return (self.compressed and self.compressed_size) or self.padded_size

    @functools.cached_property
    def compression(self):
        if not self.compressed:
            return None
        start = self.offset + self.header_size
        return sniff_compression(bytes(self._blob[start : start + 4]))

    @property
    def arch(self):
        suffix = ""
        if self.flags & FLAG_ARCH_SPECIFIC:
            suffix = "a"
        elif self.flags & FLAG_FAMILY_SPECIFIC:
            suffix = "f"
        prefix = {KIND_PTX: "compute", KIND_LTOIR: "lto"}.get(self.kind, "sm")
        return f"{prefix}_{self.arch_number}{suffix}"

    def payload(self):
        """Return the (decompressed) payload bytes"""
        start = self.offset + self.header_size
        if not self.compressed:
            return bytes(self._blob[start : start + self.padded_size])
        compression = self.compression
        decompress = get_decompressor(compression)
        if decompress is None:
            raise FatbinError(
                f"{compression}-compressed entry; install the "
                f"'{'zstandard' if compression == 'zstd' else 'lz4'}' package"
            )
        data = bytes(self._blob[start : start + self.stored_size])
        try:
            out = decompress(data, self.uncompressed_size)
        except FatbinError:
            raise
        except Exception as e:  # zstandard / lz4 / zlib raise their own types
            raise FatbinError(f"{compression} decompression failed: {e}") from e
        if len(out) != self.uncompressed_size:
            raise FatbinError(
                f"decompressed {len(out)} bytes, expected {self.uncompressed_size}"
            )
        return out


def _read_str(buf, start, size):
    raw = bytes(buf[start : start + size])
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace").strip()


@dataclass
class Fatbin:
    offset: int  # absolute offset in the backing buffer
    header_size: int
    entries_size: int
    entries: list

    @property
    def file_size(self):
        return self.header_size + self.entries_size

    @property
    def end(self):
        return self.offset + self.file_size


def _parse_entries(buf, start, end, strict):
    entries = []
    off = start
    while off + ENTRY_HEADER.size <= end:
        (
            kind,
            version,
            header_size,
            padded_size,
            compressed_size,
            _,
            _,
            arch_number,
            ident_offset,
            ident_size,
            flags,
            _,
            uncompressed_size,
        ) = ENTRY_HEADER.unpack_from(buf, off)
        entry = FatbinEntry(
            kind=kind,
            offset=off,
            header_size=header_size,
            padded_size=padded_size,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            arch_number=arch_number,
            flags=flags,
            identifier=(
                _read_str(buf, off + ident_offset, ident_size)
                if ident_offset and ident_offset + ident_size <= header_size
                else ""
            ),
            _blob=buf,
        )
        if (
            header_size < ENTRY_HEADER.size
            or off + entry.file_size > end
            or off + header_size + entry.stored_size > end
        ):
            raise FatbinError(f"corrupt fatbin entry at offset {off:#x}")
        if strict and (version != 0x0101 or header_size > 0x10000 or not kind):
            raise FatbinError(f"implausible fatbin entry at offset {off:#x}")
        entries.append(entry)
        off += entry.file_size
    if strict and (off != end or not entries):
        raise FatbinError(f"fatbin entries do not fill the fatbin at {start:#x}")
    return entries


def parse_fatbin(buf, off, end, strict=False):
    """Parse the fatbin starting at buf[off]; raise FatbinError if invalid"""
    if off + FATBIN_HEADER.size > end:
        raise FatbinError(f"truncated fatbin header at offset {off:#x}")
    magic, version, header_size, entries_size = FATBIN_HEADER.unpack_from(buf, off)
    if magic != FATBIN_MAGIC or header_size < FATBIN_HEADER.size:
        raise FatbinError(f"bad fatbin header at offset {off:#x}")
    if strict and (version != 1 or header_size != FATBIN_HEADER.size):
        raise FatbinError(f"implausible fatbin header at offset {off:#x}")
    fend = off + header_size + entries_size
    if fend > end:
        raise FatbinError(f"truncated fatbin at offset {off:#x}")
    entries = _parse_entries(buf, off + header_size, fend, strict)
    return Fatbin(off, header_size, entries_size, entries)


def find_fatbins(buf, start=0, end=None, strict=False):
    """Find the fatbins in buf[start:end], using absolute offsets.

    In a dedicated fatbin section (.nv_fatbin, __nv_relfatbin) fatbins are
    laid out back to back with alignment padding. With strict=True, fatbins
    may sit anywhere among unrelated data (e.g. in .rodata for
    cuModuleLoadData), so every magic hit is validated and false positives
    are skipped.
    """
    end = len(buf) if end is None else end
    fatbins = []
    off = start
    while True:
        off = buf.find(FATBIN_MAGIC_BYTES, off, end)
        if off < 0:
            return fatbins
        try:
            fatbin = parse_fatbin(buf, off, end, strict)
        except (FatbinError, struct.error):
            if not strict:
                raise
            off += 1
            continue
        fatbins.append(fatbin)
        off = fatbin.end


def is_fatbin(buf, off=0):
    return buf[off : off + 4] == FATBIN_MAGIC_BYTES
