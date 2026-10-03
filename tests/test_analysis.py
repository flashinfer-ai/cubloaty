import pytest

from cubloaty import analysis as A
from cubloaty import elf as E
from cubloaty import fatbin as F

from .builders import (
    PTX,
    build_archive,
    build_entry,
    build_fatbin,
    build_host_elf,
    kernel_cubin,
    pad_to,
    requires_demangler,
)

K = "_Z6kernelPf"


def image_total(img, attr="size"):
    shared = img.shared if attr == "size" else img.shared_file
    return sum(shared.values()) + sum(
        sum(getattr(fn, attr).values()) for fn in img.functions.values()
    )


def check_exact(report):
    """Every byte is attributed exactly once at every level"""
    for img in report.images:
        assert image_total(img, "size") == img.size, img.location
        assert image_total(img, "file_size") == img.file_size, img.location
    assert (
        sum(i.file_size for i in report.images) + report.container_overhead
        == report.device_file_size
    )
    assert report.host_file_size >= 0


def analyze_bytes(tmp_path, data, name="lib.so"):
    path = tmp_path / name
    path.write_bytes(data)
    report = A.analyze_file(str(path))
    check_exact(report)
    return report


# --------------------------------------------------------------------------
# Cubins


def test_cubin_attribution():
    data = kernel_cubin(kernels=[K], device_fns=["_Z3devv"])
    img = A.Image("sass", "sm_90", "", "", None, None, len(data))
    img.size = len(data)
    A.attribute_cubin(img, data)
    assert image_total(img) == len(data)

    # The helper symbol inside the kernel's .text is not a separate function
    assert set(img.functions) == {K, "_Z3devv"}
    kern, dev = img.functions[K], img.functions["_Z3devv"]
    assert kern.is_kernel and not dev.is_kernel
    assert kern.size["code"] == 0x100
    assert kern.size["info"] == 0x20
    assert kern.size["constant"] == 0x40
    assert kern.size["reloc"] == 48
    assert dev.size["code"] == 0x80

    # Section names, symbol entries and symbol names owned by the kernel
    prefixes = [".text.", ".nv.info.", ".nv.constant0.", ".nv.shared.", ".rela.text."]
    names = sum(len(p + K) + 1 for p in prefixes)
    syms = 2 * 24 + len(K) + 1 + len(f"${K}$helper") + 1
    assert kern.size["symbols"] == names + syms

    # NOBITS (.nv.shared.K, .nv.global) take no space; the .nv.merc alias of
    # .nv.constant3 is counted once
    assert img.shared["constant"] == 0x30
    assert "data" not in img.shared and "merc" not in img.shared


def test_relocatable_cuda_nobits_types():
    data = E.ElfFile(kernel_cubin())
    assert data.section_by_name(f".nv.shared.{K}").file_size == 0
    from .builders import Sec, build_elf

    elf = E.ElfFile(
        build_elf([Sec(".nv.shared.foo", 0x7000000A, size=0x400)], etype=E.ET_REL)
    )
    assert elf.section_by_name(".nv.shared.foo").file_size == 0
    host = E.ElfFile(
        build_elf(
            [Sec(".nv.shared.attrs", E.SHT_PROGBITS, data=b"x" * 16)],
            machine=183,
        )
    )
    assert host.section_by_name(".nv.shared.attrs").file_size == 16


@pytest.mark.parametrize(
    "osabi,abiversion,flags,expected",
    [
        (0x41, 8, 0x06005A04, "sm_90"),
        (0x41, 8, 0x06006402, "sm_100"),
        (0x33, 7, 0x00000550, "sm_80"),
    ],
)
def test_decode_cubin_arch(osabi, abiversion, flags, expected):
    from .builders import build_elf

    elf = E.ElfFile(build_elf([], osabi=osabi, abiversion=abiversion, flags=flags))
    assert A.decode_cubin_arch(elf) == expected


def test_decode_cubin_arch_from_compat():
    from .builders import Sec, build_elf

    compat = b"\x02\x09\x01\x00\x03\x0d\x01\x01\x04\x0b\x08\x00" + b"\0" * 8
    elf = E.ElfFile(build_elf([Sec(".nv.compat", 0x70000086, data=compat)]))
    assert A.decode_cubin_arch(elf) == "sm_90a"


# --------------------------------------------------------------------------
# PTX


def test_ptx_attribution():
    img = A.Image("ptx", "compute_90", "", "", 0, 0, 0)
    data = PTX + b"\0"
    img.size = len(data)
    A.attribute_ptx(img, data)
    assert image_total(img) == len(data)
    assert set(img.functions) == {"_Z6helperf", "_Z6kernelPf"}
    assert img.functions["_Z6kernelPf"].is_kernel
    assert not img.functions["_Z6helperf"].is_kernel
    kernel_text = PTX[PTX.index(b".visible .entry") : PTX.index(b"\t.section")]
    assert img.functions["_Z6kernelPf"].size["ptx"] == len(kernel_text)
    assert img.shared["debug"] == len(PTX) - PTX.index(b"\t.section")
    assert img.tu_hints == {"a.cu"}


# --------------------------------------------------------------------------
# Helpers


def test_apportion_is_exact_and_proportional():
    assert A.apportion(10, [1, 1, 1]) == [4, 3, 3]
    assert A.apportion(0, [5, 5]) == [0, 0]
    assert A.apportion(7, [0, 0]) == [0, 0]
    shares = A.apportion(1000, [1, 2, 3, 994])
    assert sum(shares) == 1000 and shares[3] == 994


def test_tu_names_and_canonical_name():
    blob = b"xx_ZN34_INTERNAL_57686b55_4_a_cu_45d778d96helperEf\0" + (
        b"_ZN36_GLOBAL__N__9a26021a_4_p_cu_2446ca4211anon_kernelEPf"
    )
    assert A.tu_names(blob) == {"a.cu", "p.cu"}
    static = "__nv_static_25__45ddc4bb_4_b_cu_848acf13__Z13header_kernelPf"
    assert A.tu_names(static.encode()) == {"b.cu"}
    assert A._strip_nv_static(static) == "_Z13header_kernelPf"


@requires_demangler
def test_demangle_nv_static():
    static = "__nv_static_25__45ddc4bb_4_b_cu_848acf13__Z13header_kernelPf"
    assert A.demangle_symbols([static, "_Z13header_kernelPf"]) == {
        static: "header_kernel(float*)",
        "_Z13header_kernelPf": "header_kernel(float*)",
    }
    assert (
        A.canonical_name("_INTERNAL_57686b55_4_a_cu_45d778d9::helper(float)")
        == "helper(float)"
    )


# --------------------------------------------------------------------------
# Whole files


@requires_demangler
@pytest.mark.parametrize("method", [None, "zstd", "lz4"])
def test_host_library_with_embedded_device_code(tmp_path, method):
    nv_fatbin = pad_to(build_fatbin([build_entry(kernel_cubin(), method=method)]), 256)
    nv_fatbin += build_fatbin(
        [
            build_entry(kernel_cubin(kernels=["_Z1bv"]), method=method),
            build_entry(PTX, kind=F.KIND_PTX, method="zstd"),
        ]
    )
    embedded_fb = build_fatbin(
        [build_entry(kernel_cubin(kernels=["_Z1cv"]), method="lz4")]
    )
    embedded_cubin = kernel_cubin(kernels=["_Z1dv"], flags=0x06006402)
    rodata = b"r" * 100 + embedded_fb + b"r" * 37 + embedded_cubin + b"r" * 11
    data = build_host_elf(
        [(".text", b"\x90" * 1000), (".nv_fatbin", nv_fatbin), (".rodata", rodata)]
    )
    report = analyze_bytes(tmp_path, data)

    assert report.file_format == "shared library"
    sections = {s.name: s for s in report.sections}
    assert sections[".nv_fatbin"].file_size == len(nv_fatbin)
    assert sections[".rodata"].embedded
    assert sections[".rodata"].file_size == len(embedded_fb) + len(embedded_cubin)
    assert report.device_file_size == len(nv_fatbin) + len(embedded_fb) + len(
        embedded_cubin
    )
    kinds = sorted((i.section, i.kind, i.arch) for i in report.images)
    assert kinds == [
        (".nv_fatbin", "ptx", "compute_90"),
        (".nv_fatbin", "sass", "sm_90"),
        (".nv_fatbin", "sass", "sm_90"),
        (".rodata", "sass", "sm_100"),
        (".rodata", "sass", "sm_90"),
    ]
    assert report.names[K] == "kernel(float*)"


@requires_demangler
def test_duplicate_kernels_across_tus(tmp_path):
    fatbins = b"".join(
        pad_to(
            build_fatbin([build_entry(kernel_cubin([K, f"_Z{len(u)}{u}v"], tu=tu))]),
            256,
        )
        for tu, u in [("x.cu", "ux"), ("y.cu", "uy"), ("z.cu", "uz")]
    )
    relfatbin = build_fatbin([build_entry(kernel_cubin([K], tu="r.cu"))])
    data = build_host_elf([(".nv_fatbin", fatbins), ("__nv_relfatbin", relfatbin)])
    report = analyze_bytes(tmp_path, data)

    (dup,) = A.find_duplicates(report, report.images)
    assert dup.name == "kernel(float*)" and dup.arch == "sm_90"
    assert len(dup.copies) == 3  # the relocatable copy is not counted
    assert sorted(A.function_tu(img, fn) for img, fn in dup.copies) == [
        "x.cu",
        "y.cu",
        "z.cu",
    ]
    assert dup.wasted_size == 2 * max(dup.sizes)
    assert dup.wasted_file_size == 2 * max(dup.file_sizes)

    rows = A.summarize_functions(report, report.images)
    assert rows["kernel(float*)"].by_arch["sm_90"][2] == 4


def test_static_archive(tmp_path):
    def obj(kernels):
        fb = build_fatbin([build_entry(kernel_cubin(kernels), method="zstd")])
        return build_host_elf([(".text", b"\0" * 64), (".nv_fatbin", fb)], E.ET_REL)

    long_name = "a_very_long_object_file_name.cu.o"
    data = build_archive(
        [("a.o", obj([K])), (long_name, obj([K, "_Z1fv"])), ("notes.txt", b"hi")]
    )
    report = analyze_bytes(tmp_path, data, "libx.a")
    assert report.file_format == "static library"
    assert sorted(i.member for i in report.images) == ["a.o", long_name]
    (dup,) = A.find_duplicates(report, report.images)
    assert sorted(img.tu for img, _ in dup.copies) == ["a.o", long_name]


def test_opaque_entries(tmp_path):
    entries = [
        build_entry(b"\x06\x39" * 64, kind=F.KIND_LTOIR, method="zstd"),
        build_entry(bytes(range(256)) * 2, kind=F.KIND_PTX, method="zstd"),
        build_entry(b"garbage!" * 8, kind=F.KIND_ELF),
        build_entry(b"x" * 24, kind=0x400),
    ]
    report = analyze_bytes(tmp_path, build_fatbin(entries), "x.fatbin")
    lto, ptx, bad, unknown = report.images
    assert lto.kind == "ltoir" and lto.opaque and lto.shared["ltoir"] == 128
    assert ptx.opaque and ptx.shared["ptx"] == 512 and not ptx.error
    assert bad.opaque and bad.functions == {}
    assert unknown.opaque and unknown.shared["other"] == 24


def test_bare_cubin(tmp_path):
    data = kernel_cubin(tu="t.cu", flags=0x06006402)
    report = analyze_bytes(tmp_path, data, "k.cubin")
    (img,) = report.images
    assert report.file_format == "cubin"
    assert img.arch == "sm_100" and img.tu == "t.cu"
    assert report.host_file_size == 0


def test_unrecognized_file(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"hello world")
    with pytest.raises(ValueError):
        A.analyze_file(str(path))
