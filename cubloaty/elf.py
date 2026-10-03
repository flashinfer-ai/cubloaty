"""Minimal read-only ELF parser.

Only what cubloaty needs: section headers, symbol tables and string tables of
host binaries (.so / executables / .o) and CUDA device binaries (cubins).
Supports ELF32/ELF64, both byte orders, and extended section numbering
(e_shnum == 0 / e_shstrndx == SHN_XINDEX / SHT_SYMTAB_SHNDX), which large
cubins with thousands of kernels can hit.
"""

import struct
from dataclasses import dataclass

ELF_MAGIC = b"\x7fELF"

EM_CUDA = 190

ET_REL = 1
ET_EXEC = 2
ET_DYN = 3

SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_NOBITS = 8
SHT_REL = 9
SHT_DYNSYM = 11
SHT_SYMTAB_SHNDX = 18

SHF_WRITE = 0x1
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4
SHF_INFO_LINK = 0x40

SHN_UNDEF = 0
SHN_LORESERVE = 0xFF00
SHN_XINDEX = 0xFFFF

STT_FUNC = 2

# In cubins, these sections reserve memory but have no file contents. Fully
# linked cubins mark them SHT_NOBITS; relocatable (-rdc) ones use
# processor-specific types instead, so match them by name.
CUDA_NOBITS_PREFIXES = (".nv.shared.", ".nv.local.", ".nv.merc.nv.shared.")
CUDA_NOBITS_NAMES = (".nv.shared", ".nv.global", ".nv.local")


def is_cuda_nobits(name):
    return name in CUDA_NOBITS_NAMES or name.startswith(CUDA_NOBITS_PREFIXES)


class ElfError(Exception):
    pass


@dataclass
class Section:
    index: int
    name: str
    name_offset: int
    type: int
    flags: int
    offset: int
    size: int
    link: int
    info: int
    entsize: int
    file_size: int  # bytes occupied in the file (0 for NOBITS-like sections)


@dataclass
class Symbol:
    index: int
    name: str
    name_offset: int
    value: int
    size: int
    info: int
    other: int
    shndx: int

    @property
    def type(self):
        return self.info & 0xF

    @property
    def bind(self):
        return self.info >> 4


class ElfFile:
    """Parsed view over an in-memory ELF image (bytes, bytearray or mmap)"""

    def __init__(self, data):
        if len(data) < 52 or data[:4] != ELF_MAGIC:
            raise ElfError("not an ELF file")
        self.data = data
        ei_class, ei_data = data[4], data[5]
        if ei_class not in (1, 2) or ei_data not in (1, 2):
            raise ElfError("unsupported ELF class/encoding")
        self.is64 = ei_class == 2
        self.endian = "<" if ei_data == 1 else ">"
        self.osabi = data[7]
        self.abiversion = data[8]

        if self.is64:
            fmt = self.endian + "HHIQQQIHHHHHH"
        else:
            fmt = self.endian + "HHIIIIIHHHHHH"
        (
            self.type,
            self.machine,
            _,
            _,
            self.phoff,
            self.shoff,
            self.flags,
            self.ehsize,
            self.phentsize,
            self.phnum,
            self.shentsize,
            shnum,
            shstrndx,
        ) = struct.unpack_from(fmt, data, 16)

        self.sections = []
        self.shnum = 0
        self.shstrndx = 0
        if self.shoff == 0:
            return

        sh_fmt = self.endian + ("IIQQQQIIQQ" if self.is64 else "IIIIIIIIII")
        sh_size = struct.calcsize(sh_fmt)
        if self.shentsize != sh_size:
            raise ElfError(f"unexpected section header size {self.shentsize}")

        # Extended numbering: real counts live in section header 0
        if self.shoff + sh_size > len(data):
            raise ElfError("section header table out of bounds")
        sh0 = struct.unpack_from(sh_fmt, data, self.shoff)
        if shnum == 0:
            shnum = sh0[5]
        if shstrndx == SHN_XINDEX:
            shstrndx = sh0[6]
        if self.phnum == 0xFFFF:
            self.phnum = sh0[7]
        self.shnum = shnum

        table_end = self.shoff + shnum * sh_size
        if table_end > len(data):
            raise ElfError("section header table out of bounds")
        raw = struct.iter_unpack(sh_fmt, data[self.shoff : table_end])

        headers = list(raw)
        names_off = headers[shstrndx][4] if shstrndx < shnum else None
        cuda = self.machine == EM_CUDA
        for i, h in enumerate(headers):
            name_off, typ, flags, _, off, size, link, info, _, entsize = h
            name = self._cstr(names_off + name_off) if names_off is not None else ""
            nobits = typ in (SHT_NULL, SHT_NOBITS) or (cuda and is_cuda_nobits(name))
            fsize = 0 if nobits else size
            self.sections.append(
                Section(
                    i, name, name_off, typ, flags, off, size, link, info, entsize, fsize
                )
            )
        self.shstrndx = shstrndx

    @property
    def section_header_range(self):
        return (self.shoff, self.shoff + self.shnum * self.shentsize)

    @property
    def program_header_range(self):
        return (self.phoff, self.phoff + self.phnum * self.phentsize)

    def _cstr(self, offset):
        end = self.data.find(b"\0", offset)
        if end < 0:
            end = len(self.data)
        return bytes(self.data[offset:end]).decode("utf-8", "replace")

    def section_data(self, section):
        if not section.file_size:
            return b""
        end = section.offset + section.size
        if end > len(self.data):
            raise ElfError(f"section {section.name} out of bounds")
        return self.data[section.offset : end]

    def section_by_name(self, name):
        for s in self.sections:
            if s.name == name:
                return s
        return None

    def symbols(self, symtab):
        """Parse a symbol table section (SYMTAB/DYNSYM or a vendor alias)"""
        strtab = self.sections[symtab.link]
        sym_fmt = self.endian + ("IBBHQQ" if self.is64 else "IIIBBH")
        ent = struct.calcsize(sym_fmt)
        raw = self.section_data(symtab)
        count = len(raw) // ent

        # Section indices >= SHN_LORESERVE spill into SHT_SYMTAB_SHNDX
        xindex = None
        for s in self.sections:
            if s.type == SHT_SYMTAB_SHNDX and s.link == symtab.index:
                xindex = struct.unpack_from(
                    f"{self.endian}{count}I", self.section_data(s)
                )

        result = []
        for i, fields in enumerate(struct.iter_unpack(sym_fmt, raw[: count * ent])):
            if self.is64:
                name_off, info, other, shndx, value, size = fields
            else:
                name_off, value, size, info, other, shndx = fields
            if shndx == SHN_XINDEX and xindex is not None:
                shndx = xindex[i]
            name = self._cstr(strtab.offset + name_off) if name_off else ""
            result.append(Symbol(i, name, name_off, value, size, info, other, shndx))
        return result


def elf_extent(buf, off, end):
    """Size of the ELF64 image starting at buf[off], or 0 if the bytes there
    are not a plausible ELF that fits in buf[off:end]. Used to find cubins
    embedded in host data sections."""
    if off + 64 > end or buf[off : off + 4] != ELF_MAGIC:
        return 0
    hdr = bytes(buf[off : off + 64])
    if hdr[4] != 2 or hdr[5] != 1:  # ELF64 little-endian
        return 0
    phoff, shoff = struct.unpack_from("<QQ", hdr, 32)
    ehsize, phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHHH", hdr, 52)
    if ehsize != 64 or shentsize != 64 or shoff == 0 or off + shoff + 64 > end:
        return 0
    sh_fmt = struct.Struct("<IIQQQQIIQQ")
    if shnum == 0:
        shnum = sh_fmt.unpack_from(buf, off + shoff)[5]
    extent = max(64, shoff + shnum * 64, phoff + phnum * phentsize)
    if off + extent > end:
        return 0
    for i in range(shnum):
        sh = sh_fmt.unpack_from(buf, off + shoff + i * 64)
        # Only standard types: processor-specific memory-only sections
        # (relocatable shared/local/global) may claim sizes they don't store
        if SHT_PROGBITS <= sh[1] <= SHT_SYMTAB_SHNDX and sh[1] != SHT_NOBITS:
            extent = max(extent, sh[4] + sh[5])
    return extent if off + extent <= end else 0


def elf_machine(buf, off):
    return struct.unpack_from("<H", buf, off + 18)[0]
