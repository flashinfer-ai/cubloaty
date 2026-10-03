"""Byte-accurate size attribution for CUDA binaries.

Every byte of the input is attributed exactly once, so all totals add up:

    input file
    ├── host bytes (everything outside fatbin sections)
    └── fatbin sections (.nv_fatbin, __nv_relfatbin, ...)
        ├── container overhead (fatbin headers, alignment padding)
        └── entries (one cubin / PTX / LTO-IR image each)
            ├── entry header + payload padding        -> "fatbin" category
            └── payload, attributed per function and category

Inside a cubin, sections are owned by the kernel/function whose name they
carry (.text.<fn>, .nv.info.<fn>, .nv.constant0.<fn>, .rela.text.<fn>, ...)
or that they link to via sh_info. Symbol-table entries and string-table bytes
are owned by the function they name. Overlapping sections (some .nv.merc.*
sections alias other sections' bytes) are counted once, and NOBITS sections
(.nv.shared.*, .nv.global, ...) occupy no file space so they count as zero.

"size" is the uncompressed size; "file size" is what an item occupies in the
input file. For compressed entries, the compressed payload is apportioned to
functions/categories in proportion to their uncompressed bytes (an estimate,
since compression is applied to the image as a whole).
"""

import bisect
import logging
import mmap
import os
import re
import shutil
import struct
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from . import elf as E
from .fatbin import (
    KIND_ELF,
    KIND_LTOIR,
    KIND_PTX,
    FatbinError,
    find_fatbins,
    is_fatbin,
)

logger = logging.getLogger("cubloaty")

STO_CUDA_ENTRY = 0x10  # st_other bit marking a __global__ function

# category -> description, in display order
CATEGORIES = {
    "code": "SASS code (.text)",
    "capmerc": ".nv.capmerc sections",
    "merc": ".nv.merc sections",
    "constant": "Constant banks (.nv.constant*)",
    "data": "Initialized globals",
    "info": "Kernel attributes (.nv.info, notes)",
    "reloc": "Relocations",
    "symbols": "Symbol & string tables",
    "debug": "Debug info",
    "ptx": "PTX",
    "ltoir": "LTO IR",
    "elf": "ELF headers & padding",
    "fatbin": "Fatbin headers & padding",
    "other": "Other",
}


def section_category(sec):
    n = sec.name
    if n.startswith(
        (".debug", ".nv_debug", ".nv.debug", ".nv.merc.debug", ".nv.merc.nv_debug")
    ):
        return "debug"
    if n.startswith(".nv.capmerc"):
        return "capmerc"
    if n.startswith(".nv.merc"):
        return "merc"
    if sec.type in (E.SHT_REL, E.SHT_RELA) or n.startswith(
        (".nv.resolvedrela", ".nv.rel.action")
    ):
        return "reloc"
    if sec.flags & E.SHF_EXECINSTR or n.startswith(".text"):
        return "code"
    if n.startswith(".nv.constant"):
        return "constant"
    if sec.type in (E.SHT_SYMTAB, E.SHT_STRTAB, E.SHT_SYMTAB_SHNDX):
        return "symbols"
    if n.startswith((".nv.info", ".note", ".nv.compat", ".nv.callgraph")) or (
        n.startswith(".nv.prototype")
    ):
        return "info"
    if sec.flags & E.SHF_ALLOC:
        return "data"
    return "other"


@dataclass
class Function:
    """A kernel or device function within one image"""

    name: str  # mangled
    is_kernel: bool
    size: Counter = field(default_factory=Counter)  # category -> bytes
    file_size: Counter = field(default_factory=Counter)


@dataclass
class Image:
    """One device image: a cubin, PTX or LTO-IR fatbin entry (or bare cubin)"""

    kind: str  # "sass", "ptx", "ltoir"
    arch: str
    section: str  # host section holding it ("" for a bare cubin/fatbin file)
    member: str  # archive member ("" if not from an archive)
    fatbin: int  # index of the fatbin within its section
    entry: int  # index of the entry within its fatbin
    file_size: int  # bytes in the input file (header + stored payload + padding)
    size: int = 0  # uncompressed payload size
    offset: int = None  # file offset, for cubins embedded outside fatbins
    compression: str = None
    relocatable: bool = False
    opaque: bool = False  # contents not analyzable (LTO-IR, unknown, error)
    identifier: str = ""
    tu: str = None
    tu_hints: set = field(default_factory=set)
    functions: dict = field(default_factory=dict)  # mangled name -> Function
    shared: Counter = field(default_factory=Counter)  # unowned bytes by category
    shared_file: Counter = field(default_factory=Counter)
    error: str = None

    @property
    def location(self):
        parts = [p for p in (self.member, self.section) if p]
        if self.fatbin is not None:
            parts.append(f"fatbin {self.fatbin}")
        elif self.offset is not None:
            parts.append(f"{self.offset:#x}")
        return ":".join(parts) or "file"

    def add(self, owner, category, nbytes, is_kernel=None):
        if nbytes <= 0:
            return
        if owner is None:
            self.shared[category] += nbytes
            return
        fn = self.functions.get(owner)
        if fn is None:
            fn = self.functions[owner] = Function(owner, bool(is_kernel))
        fn.size[category] += nbytes


@dataclass
class HostSection:
    name: str
    member: str
    file_size: int
    fatbins: int
    container_overhead: int  # fatbin headers + inter-fatbin padding
    embedded: bool = False  # device code found inside a regular host section


@dataclass
class Report:
    path: str
    file_size: int
    file_format: str
    images: list = field(default_factory=list)
    sections: list = field(default_factory=list)  # HostSection
    names: dict = field(default_factory=dict)  # mangled -> display name

    @property
    def device_file_size(self):
        if self.file_format == "cubin":
            return self.file_size
        return sum(s.file_size for s in self.sections)

    @property
    def host_file_size(self):
        return self.file_size - self.device_file_size

    @property
    def container_overhead(self):
        return sum(s.container_overhead for s in self.sections)


# --------------------------------------------------------------------------
# Cubin attribution


def _find_owner(name, funcs):
    """Function whose name is the longest '.'-separated suffix of `name`"""
    i = name.find(".", 1)
    while i != -1:
        if name[i + 1 :] in funcs:
            return name[i + 1 :]
        i = name.find(".", i + 1)
    return None


def _is_symtab(sec, is64):
    if sec.type in (E.SHT_SYMTAB, E.SHT_DYNSYM):
        return True
    # vendor aliases such as .nv.merc.symtab use a processor-specific type
    return sec.name.endswith("symtab") and sec.entsize == (24 if is64 else 16)


ELFOSABI_CUDA_V2 = 0x41  # OS/ABI of cubins from current toolkits


def decode_cubin_arch(elf):
    """SM architecture of a bare cubin.

    Current cubins (OS/ABI 0x41) keep the SM number in bits 8-15 of e_flags,
    older ones in bits 0-7. An "a" target is marked in .nv.compat by a
    {u8 format=2, u8 attribute=0x09, u8 value, u8 pad} record with value 1
    (found by comparing sm_XX and sm_XXa cubins); records with format 4 are
    followed by `u16 value` bytes of data. "f" targets cannot be told apart
    from plain ones."""
    if elf.osabi == ELFOSABI_CUDA_V2:
        sm = (elf.flags >> 8) & 0xFF
    else:
        sm = elf.flags & 0xFF
    accel = False
    compat = elf.section_by_name(".nv.compat")
    if compat is not None:
        raw = bytes(elf.section_data(compat))
        i = 0
        while i + 4 <= len(raw):
            fmt, attr = raw[i], raw[i + 1]
            if fmt == 0x02 and attr == 0x09:
                accel = bool(raw[i + 2])
            i += 4 + (int.from_bytes(raw[i + 2 : i + 4], "little") if fmt == 4 else 0)
    return f"sm_{sm}{'a' if accel else ''}"


def attribute_cubin(img, data):
    elf = E.ElfFile(data)
    img.relocatable = elf.type == E.ET_REL
    secs = elf.sections
    nsec = len(secs)

    symtabs = [s for s in secs if s.file_size and _is_symtab(s, elf.is64)]
    symbols = {}
    for tab in symtabs:
        if tab.link < nsec:
            symbols[tab.index] = elf.symbols(tab)
    main = next((s for s in symtabs if s.type == E.SHT_SYMTAB), None)
    funcs = {}
    for sym in symbols.get(main.index, []) if main else []:
        if sym.type == E.STT_FUNC and sym.name:
            funcs[sym.name] = sym

    owner = [_find_owner(s.name, funcs) for s in secs]
    for s in secs:
        if (
            owner[s.index] is None
            and 0 < s.info < nsec
            and (s.flags & E.SHF_INFO_LINK or s.type in (E.SHT_REL, E.SHT_RELA))
        ):
            owner[s.index] = owner[s.info]

    # Sub-section ownership of string/symbol table bytes: (start, end, owner)
    claims = defaultdict(list)
    if 0 < elf.shstrndx < nsec:
        for s in secs[1:]:
            start = s.name_offset
            claims[elf.shstrndx].append(
                (start, start + len(s.name.encode()) + 1, owner[s.index])
            )
    for tab_index, syms in symbols.items():
        tab = secs[tab_index]
        ent = tab.entsize or (24 if elf.is64 else 16)
        for sym in syms[1:]:
            o = owner[sym.shndx] if 0 < sym.shndx < nsec else None
            claims[tab_index].append((sym.index * ent, (sym.index + 1) * ent, o))
            if sym.name_offset:
                start = sym.name_offset
                claims[tab.link].append((start, start + len(sym.name.encode()) + 1, o))

    # Sweep all file ranges in offset order; earlier ranges win overlaps
    ranges = [(0, elf.ehsize, -3, None, "elf")]
    if elf.phnum:
        ranges.append((*elf.program_header_range, -2, None, "elf"))
    ranges.append((*elf.section_header_range, -1, None, "elf"))
    for s in secs:
        if s.file_size:
            cat = section_category(s)
            ranges.append((s.offset, s.offset + s.size, s.index, owner[s.index], cat))
    ranges.sort(key=lambda r: (r[0], r[2]))

    acc = defaultdict(int)  # (owner, category) -> bytes
    total = len(data)
    cursor = 0
    for start, end, index, own, cat in ranges:
        end = min(end, total)
        if start > cursor:
            acc[(None, "elf")] += start - cursor  # alignment padding
            cursor = start
        if end <= cursor:
            continue
        pos = cursor
        for cs, ce, co in sorted(claims.get(index, ()), key=lambda c: c[0]):
            cs, ce = max(cs + start, pos), min(ce + start, end)
            if ce > cs:
                acc[(own, cat)] += cs - pos
                acc[(co, cat)] += ce - cs
                pos = ce
        acc[(own, cat)] += end - pos
        cursor = end
    acc[(None, "elf")] += total - cursor

    for (o, cat), n in acc.items():
        sym = funcs.get(o) if o else None
        img.add(o, cat, n, sym is not None and bool(sym.other & STO_CUDA_ENTRY))

    strtab = secs[main.link] if main is not None and main.link < nsec else None
    if strtab is not None:
        img.tu_hints |= tu_names(elf.section_data(strtab))


# --------------------------------------------------------------------------
# PTX attribution

_PTX_DECL_RE = re.compile(
    rb"[ \t]*(?:\.(?:visible|weak|extern|common)[ \t]+)*\.(entry|func)[ \t]+"
    rb"(?:\([^)]*\)[ \t]*)?([A-Za-z_$%][\w$]*)"
)
_PTX_DEBUG_RE = re.compile(rb"[ \t]*\.section[ \t]+\.(?:nv_)?debug")
_PTX_CLOSE_RE = re.compile(rb"\n[ \t]*\}[ \t]*\n?")


def _ptx_line_matches(text, keyword, regex):
    """Matches of `regex` anchored at the start of lines containing `keyword`
    (the match must cover it). A literal search is much faster than a
    MULTILINE regex over multi-megabyte PTX."""
    found = []
    pos = text.find(keyword)
    while pos >= 0:
        m = regex.match(text, text.rfind(b"\n", 0, pos) + 1)
        if m and m.start() <= pos < m.end():
            found.append(m)
        pos = text.find(keyword, pos + len(keyword))
    return found


def attribute_ptx(img, data):
    text = data.rstrip(b"\0")
    img.add(None, "ptx", len(data) - len(text))  # NUL terminator / padding

    debug = []
    for m in _ptx_line_matches(text, b".section", _PTX_DEBUG_RE):
        close = _PTX_CLOSE_RE.search(text, m.end())
        debug.append((m.start(), close.end() if close else len(text)))
    debug.sort()

    def add_shared(a, b):
        for ds, de in debug:
            lo, hi = max(a, ds), min(b, de)
            if lo < hi:
                img.add(None, "ptx", lo - a)
                img.add(None, "debug", hi - lo)
                a = hi
        img.add(None, "ptx", b - a)

    decls = _ptx_line_matches(text, b".entry", _PTX_DECL_RE)
    decls += _ptx_line_matches(text, b".func", _PTX_DECL_RE)
    decls.sort(key=lambda m: m.start())
    pos = 0
    for m in decls:
        if m.start() < pos:
            continue
        brace = text.find(b"{", m.end())
        semi = text.find(b";", m.end())
        if brace < 0 or 0 <= semi < brace:
            continue  # prototype/declaration only
        close = text.find(b"\n}", brace)
        end = len(text) if close < 0 else close + 2
        if text[end : end + 1] == b"\n":
            end += 1
        add_shared(pos, m.start())
        name = m.group(2).decode("ascii", "replace")
        img.add(name, "ptx", end - m.start(), is_kernel=m.group(1) == b"entry")
        pos = end
    add_shared(pos, len(text))
    # A PTX image comes from a single TU, so the first hit is enough
    img.tu_hints |= tu_names(text, first=True)


# --------------------------------------------------------------------------
# Translation unit inference

# nvcc mangles internal-linkage entities with the TU file name:
#   _INTERNAL_57686b55_4_a_cu_45d778d9::helper, _GLOBAL__N__9a26021a_4_p_cu_...
# and, in device-linked (-rdc) images, renames static kernels to
#   __nv_static_25__45ddc4bb_4_b_cu_848acf13__Z13header_kernelPf
_TU_RE = re.compile(rb"(?:_INTERNAL_|_GLOBAL__N__|__nv_static_\d+__)[0-9a-f]{8}_(\d+)_")
_TU_EXTS = ("cu", "cuh", "cpp", "cxx", "cc", "c")


def tu_names(blob, first=False):
    names = set()
    blob = bytes(blob)
    for prefix in (b"_INTERNAL_", b"_GLOBAL__N__", b"__nv_static_"):
        pos = blob.find(prefix)
        while pos >= 0:
            m = _TU_RE.match(blob, pos)
            pos = blob.find(prefix, pos + 1)
            if not m:
                continue
            n = int(m.group(1))
            raw = blob[m.end() : m.end() + n]
            if len(raw) != n or not re.fullmatch(rb"\w+", raw):
                continue
            name = raw.decode()
            for ext in _TU_EXTS:
                if name.endswith("_" + ext):
                    name = name[: -len(ext) - 1] + "." + ext
                    break
            names.add(name)
            if first:
                return names
    return names


def function_tu(img, fn):
    """TU of one function copy: from its own mangled name when it embeds one
    (static/anonymous-namespace entities), else from its image"""
    names = tu_names(fn.name.encode())
    return next(iter(names)) if len(names) == 1 else img.tu


# --------------------------------------------------------------------------
# File-size apportioning


def apportion(total, weights):
    """Split integer `total` proportionally to `weights` (largest remainder)"""
    wsum = sum(weights)
    if wsum <= 0:
        return [0] * len(weights)
    raw = [w * total for w in weights]
    out = [r // wsum for r in raw]
    short = total - sum(out)
    if short:
        order = sorted(range(len(raw)), key=lambda i: raw[i] % wsum, reverse=True)
        for i in order[:short]:
            out[i] += 1
    return out


def finalize_file_sizes(img, stored, overhead):
    """Distribute `stored` payload file bytes over the image's buckets"""
    buckets = [(None, cat, n) for cat, n in img.shared.items()]
    for fn in img.functions.values():
        buckets.extend((fn, cat, n) for cat, n in fn.size.items())
    shares = apportion(stored, [b[2] for b in buckets])
    if not buckets or sum(b[2] for b in buckets) == 0:
        overhead += stored
    for (fn, cat, _), share in zip(buckets, shares):
        if fn is None:
            img.shared_file[cat] += share
        else:
            fn.file_size[cat] += share
    if overhead:
        img.shared_file["fatbin"] += overhead


# --------------------------------------------------------------------------
# Input handling


IMAGE_KINDS = {KIND_ELF: "sass", KIND_PTX: "ptx", KIND_LTOIR: "ltoir"}


def _looks_like_ptx(data):
    head = data[:256]
    return head.isascii() and (b".version" in data[:4096] or b"//" in head)


def analyze_entry(entry, section, member, fatbin_index, entry_index):
    """Attribute one fatbin entry. Payloads that cannot be decoded (LTO-IR,
    unknown kinds, unrecognized contents) are kept as one opaque block whose
    sizes come from the entry header."""
    kind = IMAGE_KINDS.get(entry.kind, entry.kind_name)
    img = Image(
        kind=kind,
        arch=entry.arch,
        section=section,
        member=member,
        fatbin=fatbin_index,
        entry=entry_index,
        file_size=entry.file_size,
        compression=entry.compression,
        identifier=entry.identifier,
    )
    opaque = {"ptx": "ptx", "ltoir": "ltoir"}.get(kind, "other")

    def make_opaque():
        img.opaque = True
        img.functions.clear()
        img.shared.clear()
        img.size = entry.uncompressed_size or entry.stored_size
        img.add(None, opaque, img.size)

    try:
        if entry.kind not in (KIND_ELF, KIND_PTX):
            make_opaque()
        else:
            payload = entry.payload()
            img.size = len(payload)
            if payload[:4] == E.ELF_MAGIC:
                attribute_cubin(img, payload)
            elif kind == "ptx" and _looks_like_ptx(payload):
                attribute_ptx(img, payload)
            else:
                make_opaque()
    except (FatbinError, E.ElfError, struct.error, IndexError, ValueError) as e:
        logger.debug(f"Could not analyze {kind} image in {img.location}: {e}")
        img.error = str(e)
        make_opaque()
    finalize_file_sizes(img, entry.stored_size, entry.file_size - entry.stored_size)
    return img


def analyze_fatbins(fatbins, section, member):
    images = []
    for fi, fb in enumerate(fatbins):
        imgs = [
            analyze_entry(entry, section, member, fi, ei)
            for ei, entry in enumerate(fb.entries)
        ]
        # All entries of a fatbin come from the same TU
        hints = set().union(*(i.tu_hints for i in imgs)) if imgs else set()
        idents = {
            i.identifier for i in imgs if i.identifier and " " not in i.identifier
        }
        if member:
            tu = member
        elif len(idents) == 1:
            tu = idents.pop()
        elif len(hints) == 1:
            tu = next(iter(hints))
        else:
            tu = None
        for img in imgs:
            img.tu = tu
        images.extend(imgs)
    return images


def analyze_bare_cubin(data, section="", member="", offset=None):
    elf = E.ElfFile(data)
    img = Image("sass", decode_cubin_arch(elf), section, member, None, None, len(data))
    img.offset = offset
    img.size = len(data)
    attribute_cubin(img, data)
    finalize_file_sizes(img, len(data), 0)
    img.tu = member or (next(iter(img.tu_hints)) if len(img.tu_hints) == 1 else None)
    return img


def _find_cubins(buf, start, end, fatbins):
    """Raw CUDA ELF images embedded in buf[start:end] outside any fatbin"""
    found = []
    starts = [fb.offset for fb in fatbins]  # sorted, non-overlapping
    off = start
    while True:
        off = buf.find(E.ELF_MAGIC, off, end)
        if off < 0:
            return found
        i = bisect.bisect_right(starts, off) - 1
        if i >= 0 and off < fatbins[i].end:
            off = fatbins[i].end  # ELF entries inside a fatbin are handled there
            continue
        size = E.elf_extent(buf, off, end)
        if size and E.elf_machine(buf, off) == E.EM_CUDA:
            found.append((off, size))
            off += size
        else:
            off += 4


def analyze_region(report, buf, start, end, section, member="", dedicated=True):
    """Analyze device code in buf[start:end].

    A dedicated region (.nv_fatbin, __nv_relfatbin, a .fatbin file) is all
    device code: bytes outside fatbin entries are container overhead. Any
    other host section is scanned for embedded fatbins and cubins (libraries
    such as cuFFT/cuBLASLt keep runtime-loaded kernels in .data/.rodata);
    bytes around them stay host bytes.
    """
    if dedicated:
        try:
            fatbins = find_fatbins(buf, start, end)
        except (FatbinError, struct.error) as e:
            logger.warning(f"{section or 'fatbin'}: {e}; skipping invalid data")
            fatbins = find_fatbins(buf, start, end, strict=True)
        cubins = []
    else:
        fatbins = find_fatbins(buf, start, end, strict=True)
        cubins = _find_cubins(buf, start, end, fatbins)
    if not fatbins and not cubins:
        return

    images = analyze_fatbins(fatbins, section, member)
    for off, size in cubins:
        try:
            images.append(
                analyze_bare_cubin(buf[off : off + size], section, member, off)
            )
        except (E.ElfError, struct.error, IndexError, ValueError) as e:
            logger.debug(f"Skipping embedded ELF at {off:#x} in {section}: {e}")

    entry_bytes = sum(e.file_size for fb in fatbins for e in fb.entries)
    if dedicated:
        device = end - start
        overhead = device - entry_bytes
    else:
        overhead = sum(fb.header_size for fb in fatbins)
        device = (
            entry_bytes
            + overhead
            + sum(i.file_size for i in images if i.fatbin is None)
        )
    logger.debug(
        f"{member + ':' if member else ''}{section}: {len(fatbins)} fatbin(s), "
        f"{len(cubins)} embedded cubin(s), {device} device bytes"
    )
    report.images.extend(images)
    report.sections.append(
        HostSection(section, member, device, len(fatbins), overhead, not dedicated)
    )


FATBIN_SECTIONS = (".nv_fatbin", "__nv_relfatbin", "__nv_fatbin")


def _analyze_host_elf(report, elf, member=""):
    buf = elf.data
    for sec in elf.sections:
        if not sec.file_size or sec.offset + sec.size > len(buf):
            continue
        start, end = sec.offset, sec.offset + sec.size
        dedicated = sec.name in FATBIN_SECTIONS or (
            "fatbin" in sec.name.lower() and is_fatbin(buf, start)
        )
        analyze_region(report, buf, start, end, sec.name, member, dedicated)


def _iter_archive(data):
    """Yield (member_name, offset, size) for each member of an ar archive"""
    off = 8
    longnames = b""
    while off + 60 <= len(data):
        hdr = bytes(data[off : off + 60])
        name = hdr[:16].decode("utf-8", "replace").rstrip()
        size = int(hdr[48:58].decode().strip() or 0)
        start = off + 60
        if name == "//":
            longnames = bytes(data[start : start + size])
        elif name not in ("/", "/SYM64/", "__.SYMDEF", "__.SYMDEF SORTED"):
            if name.startswith("/") and name[1:].isdigit():
                i = int(name[1:])
                end = longnames.find(b"/\n", i)
                name = longnames[i : end if end >= 0 else None].decode()
            elif name.startswith("#1/"):  # BSD long name stored inline
                n = int(name[3:])
                name = bytes(data[start : start + n]).decode().rstrip("\0")
                start, size = start + n, size - n
            yield name.rstrip("/"), start, size
        off = start + size + (size & 1)


def analyze_file(path):
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) if file_size else b""

    head = bytes(data[:8])
    if head == b"!<arch>\n":
        report = Report(path, file_size, "static library")
        for name, start, size in _iter_archive(data):
            member = data[start : start + size]
            if bytes(member[:4]) != E.ELF_MAGIC:
                continue
            try:
                _analyze_host_elf(report, E.ElfFile(member), member=name)
            except E.ElfError as e:
                logger.debug(f"Skipping archive member {name}: {e}")
    elif head[:4] == E.ELF_MAGIC:
        elf = E.ElfFile(data)
        if elf.machine == E.EM_CUDA:
            report = Report(path, file_size, "cubin")
            report.images.append(analyze_bare_cubin(data))
        else:
            kinds = {E.ET_REL: "object file", E.ET_EXEC: "executable"}
            report = Report(path, file_size, kinds.get(elf.type, "shared library"))
            _analyze_host_elf(report, elf)
    elif is_fatbin(data):
        report = Report(path, file_size, "fatbin")
        analyze_region(report, data, 0, file_size, "")
    else:
        raise ValueError("unrecognized file format (expected ELF, fatbin or .a)")

    names = {fn.name for img in report.images for fn in img.functions.values()}
    report.names = demangle_symbols(names)
    return report


# --------------------------------------------------------------------------
# Names


def _find_tool(name):
    path = shutil.which(name)
    if path is None:
        cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        for root in filter(None, (cuda, "/usr/local/cuda")):
            candidate = os.path.join(root, "bin", name)
            if os.access(candidate, os.X_OK):
                return candidate
    return path


def _demangle_with(tool, symbols):
    try:
        result = subprocess.run(
            [tool],
            input="\n".join(symbols),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        logger.debug(f"{tool} failed: {e}")
        return None
    demangled = result.stdout.splitlines()
    if len(demangled) != len(symbols):
        logger.debug(f"{tool} returned {len(demangled)} lines for {len(symbols)}")
        return None
    return demangled


def demangle_symbols(symbols):
    """Batch-demangle C++ symbol names.

    All symbols go through a single c++filt process (one process per symbol
    is prohibitively slow for large libraries). GNU c++filt gives up on some
    long CUTLASS/CuTe names, so leftovers are retried with NVIDIA's cu++filt.
    Symbols that fail to demangle are kept as-is.
    """
    # Demangle the original name inside per-TU renamed static kernels
    inner = {s: _strip_nv_static(s) for s in symbols}
    result = {s: s for s in set(inner.values())}
    pending = sorted(result)
    for name in ("c++filt", "cu++filt"):
        tool = _find_tool(name)
        if not pending or tool is None:
            continue
        demangled = _demangle_with(tool, pending)
        if demangled is not None:
            result.update(zip(pending, demangled))
            pending = [s for s in pending if result[s] == s and s.startswith("_Z")]
    if pending and len(pending) == len(result):
        logger.warning("c++filt not found; showing mangled names")
    return {s: result[i] for s, i in inner.items()}


_NV_STATIC_RE = re.compile(r"__nv_static_(\d+)__")


def _strip_nv_static(name):
    m = _NV_STATIC_RE.match(name)
    if m is None:
        return name
    rest = name[m.end() + int(m.group(1)) :]
    return rest if rest.startswith("_Z") else name


_INTERNAL_NS_RE = re.compile(r"_INTERNAL_[0-9a-f]{8}_\d+_\w+?::")


def canonical_name(demangled):
    """Name with TU-specific internal-linkage prefixes removed, so copies of
    the same header-defined function from different TUs compare equal"""
    return _INTERNAL_NS_RE.sub("", demangled)


# --------------------------------------------------------------------------
# Aggregated views


@dataclass
class FunctionSummary:
    """A function aggregated across images (archs, TUs)"""

    name: str  # canonical demangled name
    mangled: set = field(default_factory=set)
    is_kernel: bool = False
    size: int = 0
    file_size: int = 0
    by_arch: dict = field(default_factory=dict)  # arch -> [size, file_size, copies]
    by_category: Counter = field(default_factory=Counter)


def summarize_functions(report, images):
    rows = {}
    for img in images:
        for fn in img.functions.values():
            name = canonical_name(report.names.get(fn.name, fn.name))
            row = rows.get(name)
            if row is None:
                row = rows[name] = FunctionSummary(name)
            row.mangled.add(fn.name)
            row.is_kernel |= fn.is_kernel
            size, fsize = sum(fn.size.values()), sum(fn.file_size.values())
            row.size += size
            row.file_size += fsize
            arch = row.by_arch.setdefault(img.arch, [0, 0, 0])
            arch[0] += size
            arch[1] += fsize
            arch[2] += 1
            row.by_category.update(fn.size)
    return rows


@dataclass
class Duplicate:
    """The same function compiled into several images of one arch (issue #1:
    header-defined kernels get a full copy in every TU that launches them)"""

    name: str
    arch: str
    copies: list  # [(Image, Function)]

    @property
    def sizes(self):
        return [sum(fn.size.values()) for _, fn in self.copies]

    @property
    def file_sizes(self):
        return [sum(fn.file_size.values()) for _, fn in self.copies]

    @property
    def wasted_size(self):
        return sum(self.sizes) - max(self.sizes)

    @property
    def wasted_file_size(self):
        return sum(self.file_sizes) - max(self.file_sizes)


def find_duplicates(report, images):
    """Group functions by (arch, canonical name) across loadable images.

    Relocatable images (-rdc objects in __nv_relfatbin) are excluded: they are
    inputs to device linking rather than code that gets loaded.
    """
    groups = defaultdict(list)
    for img in images:
        if img.relocatable or img.section == "__nv_relfatbin":
            continue
        for fn in img.functions.values():
            name = canonical_name(report.names.get(fn.name, fn.name))
            groups[(img.arch, name)].append((img, fn))
    dups = [
        Duplicate(name, arch, copies)
        for (arch, name), copies in groups.items()
        if len(copies) > 1
    ]
    dups.sort(key=lambda d: (d.wasted_file_size, d.wasted_size), reverse=True)
    return dups
