"""Builders for synthetic ELF / fatbin / archive inputs used by the tests"""

import shutil
import struct
import zlib

import lz4.block
import lz4.frame
import pytest

from cubloaty import elf as E
from cubloaty import fatbin as F

try:
    from compression import zstd as _zstd

    def zstd_compress(data):
        return _zstd.compress(data)

except ImportError:
    import zstandard

    def zstd_compress(data):
        return zstandard.ZstdCompressor().compress(data)


EM_AARCH64 = 183

requires_demangler = pytest.mark.skipif(
    not (shutil.which("c++filt") or shutil.which("cu++filt")),
    reason="c++filt not found",
)


def align(n, a):
    return (n + a - 1) // a * a


class Sec:
    def __init__(
        self,
        name,
        type=E.SHT_PROGBITS,
        flags=0,
        data=b"",
        size=None,
        link=0,
        info=0,
        entsize=0,
        align=1,
        alias=None,
    ):
        self.name = name
        self.type = type
        self.flags = flags
        self.data = data
        self.size = len(data) if size is None else size
        self.link = link  # int index or section name
        self.info = info  # int index or section name
        self.entsize = entsize
        self.align = align
        self.alias = alias  # name of a section whose bytes this one shares


class Sym:
    def __init__(
        self, name, section, value=0, size=0, type=E.STT_FUNC, bind=1, other=0
    ):
        self.name = name
        self.section = section
        self.value = value
        self.size = size
        self.type = type
        self.bind = bind
        self.other = other


def build_elf(
    sections,
    symbols=(),
    machine=E.EM_CUDA,
    etype=E.ET_EXEC,
    osabi=0x41,
    abiversion=8,
    flags=0x06005A04,
):
    """Build an ELF64 LE image. Section 0 is NULL; .shstrtab, .strtab and
    .symtab come first, followed by `sections` in order."""
    sections = list(sections)
    names = [".shstrtab", ".strtab", ".symtab"] + [s.name for s in sections]
    index = {n: i + 1 for i, n in enumerate(names)}

    shstrtab = b"\0"
    name_off = {}
    for n in names:
        name_off[n] = len(shstrtab)
        shstrtab += n.encode() + b"\0"

    strtab = b"\0"
    symtab = b"\0" * 24
    for sym in symbols:
        off = len(strtab)
        strtab += sym.name.encode() + b"\0"
        shndx = index[sym.section] if sym.section else 0
        info = (sym.bind << 4) | sym.type
        symtab += struct.pack(
            "<IBBHQQ", off, info, sym.other, shndx, sym.value, sym.size
        )

    all_secs = [
        Sec(".shstrtab", E.SHT_STRTAB, data=shstrtab),
        Sec(".strtab", E.SHT_STRTAB, data=strtab),
        Sec(".symtab", E.SHT_SYMTAB, data=symtab, link=".strtab", entsize=24, align=8),
    ] + sections

    def resolve(v):
        return index[v] if isinstance(v, str) else v

    out = bytearray(64)
    placed = {}
    for s in all_secs:
        if s.alias:
            placed[s.name] = placed[s.alias]
            continue
        off = align(len(out), s.align)
        if s.type == E.SHT_NOBITS or (
            machine == E.EM_CUDA and E.is_cuda_nobits(s.name)
        ):
            placed[s.name] = (off, s.size)
            continue
        out += b"\0" * (off - len(out))
        out += s.data
        placed[s.name] = (off, len(s.data))

    shoff = align(len(out), 8)
    out += b"\0" * (shoff - len(out))
    out += b"\0" * 64  # NULL section header
    for s in all_secs:
        off, size = placed[s.name]
        out += struct.pack(
            "<IIQQQQIIQQ",
            name_off[s.name],
            s.type,
            s.flags,
            0,
            off,
            s.size if s.alias is None else size,
            resolve(s.link),
            resolve(s.info),
            s.align,
            s.entsize,
        )
    ident = b"\x7fELF" + bytes([2, 1, 1, osabi, abiversion]) + b"\0" * 7
    shnum = len(all_secs) + 1
    out[:64] = ident + struct.pack(
        "<HHIQQQIHHHHHH", etype, machine, 1, 0, 0, shoff, flags, 64, 56, 0, 64, shnum, 1
    )
    return bytes(out)


def compress(data, method):
    # flag bits nvcc sets for ZSTD (--compress-mode=size) and LZ4 (speed)
    if method == "zstd":
        return zstd_compress(data), 0x8000
    if method == "lz4":
        return lz4.block.compress(data, store_size=False), 0x2000
    if method == "lz4frame":
        return lz4.frame.compress(data), 0
    if method == "zlib":
        return zlib.compress(data), 0
    return data, 0


def entry_header(
    kind, arch, header_size, padded, compressed, flags, uncompressed, _=0, ident=(0, 0)
):
    return struct.pack(
        "<HHIQIIIIIIQQQ",
        kind,
        0x0101,
        header_size,
        padded,
        compressed,
        0,
        0x10008,
        arch,
        ident[0],
        ident[1],
        flags,
        0,
        uncompressed,
    )


def build_entry(
    payload,
    kind=F.KIND_ELF,
    arch=90,
    method=None,
    flags=0x11,  # observed on every nvcc entry (64-bit, Linux)
    identifier=None,
):
    code, cflag = compress(payload, method)
    ident = b""
    ident_ref = (0, 0)
    if identifier:
        ident = identifier.encode() + b"\0"
        ident = ident + b"\0" * (align(len(ident), 8) - len(ident))
        ident_ref = (64, len(identifier))
    header_size = 64 + len(ident)
    padded = align(len(code), 8)
    hdr = entry_header(
        kind,
        arch,
        header_size,
        padded,
        len(code) if method else 0,
        flags | cflag,
        len(payload) if method else 0,
        0,
        ident_ref,
    )
    return hdr + ident + code + b"\0" * (padded - len(code))


def build_fatbin(entries):
    body = b"".join(entries)
    return struct.pack("<IHHQ", F.FATBIN_MAGIC, 1, 16, len(body)) + body


def pad_to(data, alignment):
    return data + b"\0" * (align(len(data), alignment) - len(data))


def build_host_elf(sections, etype=E.ET_DYN):
    """Host ELF from (name, data) pairs"""
    secs = [
        Sec(
            n,
            E.SHT_PROGBITS,
            flags=E.SHF_ALLOC,
            data=d,
            align=256 if "fatbin" in n else 8,
        )
        for n, d in sections
    ]
    return build_elf(
        secs, machine=EM_AARCH64, etype=etype, osabi=0, abiversion=0, flags=0
    )


def build_archive(members):
    """GNU ar archive from (name, data) pairs; long names use the // table"""
    longnames = b""
    headers = []
    for name, _ in members:
        if len(name) > 15:
            headers.append(f"/{len(longnames)}".encode())
            longnames += name.encode() + b"/\n"
        else:
            headers.append(name.encode() + b"/")

    def member(name, data):
        hdr = name.ljust(16) + b"0".ljust(12) + b"0".ljust(6) + b"0".ljust(6)
        hdr += b"644".ljust(8) + str(len(data)).encode().ljust(10) + b"`\n"
        return hdr + data + (b"\n" if len(data) % 2 else b"")

    out = b"!<arch>\n"
    if longnames:
        out += member(b"//", longnames)
    for h, (_, data) in zip(headers, members):
        out += member(h, data)
    return out


# --------------------------------------------------------------------------
# A small but realistic cubin


def kernel_cubin(kernels=("_Z6kernelPf",), device_fns=(), tu=None, flags=0x06005A04):
    """Cubin with, per kernel K: .text.K, .nv.info.K, .nv.constant0.K,
    .nv.shared.K (NOBITS), .rela.text.K and an inlined `$K$helper` symbol;
    per device function D: .text.D. Plus shared .nv.constant3 aliased by
    .nv.merc.nv.constant.pic and a NOBITS .nv.global.

    `tu` adds an _INTERNAL_ symbol naming that TU (like libcu++ objects)."""
    secs = [
        Sec(".nv.info", 0x70000000, data=b"\x04\x2f\x08\x00" + b"\0" * 8, align=4),
        Sec(".nv.constant3", E.SHT_PROGBITS, E.SHF_ALLOC, data=b"\x01" * 0x30, align=8),
        Sec(".nv.merc.nv.constant.pic", 0x7000007D, alias=".nv.constant3"),
        Sec(".nv.global", E.SHT_NOBITS, E.SHF_ALLOC | E.SHF_WRITE, size=0x41),
    ]
    syms = []
    if tu:
        stem, ext = tu.rsplit(".", 1)
        mangled = f"{stem}_{ext}"
        name = f"_INTERNAL_0123abcd_{len(mangled)}_{mangled}_89abcdef"
        syms.append(Sym(f"_ZN{len(name)}{name}3fooE", ".nv.global", type=1, bind=0))
    for k in kernels:
        text = f".text.{k}"
        secs += [
            Sec(
                text,
                E.SHT_PROGBITS,
                E.SHF_ALLOC | E.SHF_EXECINSTR,
                data=b"\xaa" * 0x100,
                align=128,
            ),
            Sec(
                f".nv.info.{k}",
                0x70000000,
                E.SHF_INFO_LINK,
                data=b"\x04" * 0x20,
                info=text,
                align=4,
            ),
            Sec(
                f".nv.constant0.{k}",
                E.SHT_PROGBITS,
                E.SHF_ALLOC | E.SHF_INFO_LINK,
                data=b"\0" * 0x40,
                info=text,
                align=4,
            ),
            Sec(
                f".nv.shared.{k}",
                E.SHT_NOBITS,
                E.SHF_ALLOC | E.SHF_WRITE,
                size=0x1000,
                info=text,
            ),
            Sec(
                f".rela.text.{k}",
                E.SHT_RELA,
                E.SHF_INFO_LINK,
                data=b"\x05" * 48,
                link=".symtab",
                info=text,
                entsize=24,
                align=8,
            ),
        ]
        syms.append(Sym(k, text, size=0x100, other=0x10))
        syms.append(Sym(f"${k}$helper", text, value=0x40, size=0x20, bind=0))
    for d in device_fns:
        secs.append(
            Sec(
                f".text.{d}",
                E.SHT_PROGBITS,
                E.SHF_ALLOC | E.SHF_EXECINSTR,
                data=b"\xbb" * 0x80,
                align=128,
            )
        )
        syms.append(Sym(d, f".text.{d}", size=0x80))
    return build_elf(secs, syms, flags=flags)


PTX = b"""//
// Generated by NVIDIA NVVM Compiler
//
.version 9.3
.target sm_90
.address_size 64

.extern .func (.param .b32 func_retval0) vprintf
(
\t.param .b64 vprintf_param_0
)
;
.global .align 4 .b8 _ZN34_INTERNAL_0123abcd_4_a_cu_89abcdef3fooE[4];

.func (.param .b32 func_retval0) _Z6helperf(
\t.param .b32 _Z6helperf_param_0
)
{
\t.reg .b32 \t%f<3>;
\tret;

}
\t// .globl\t_Z6kernelPf
.visible .entry _Z6kernelPf(
\t.param .u64 .ptr .align 1 _Z6kernelPf_param_0
)
.maxntid 128, 1, 1
{
\t.reg .b32 \t%r<4>;
\t{ // callseq 0, 0
\t.param .b32 retval0;
\t} // callseq 0
\tret;

}
\t.section\t.debug_str
\t{
.b8 95,90,0
\t}
"""
