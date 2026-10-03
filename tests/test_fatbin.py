import os

import pytest

from cubloaty import fatbin as F

from .builders import (
    build_entry,
    build_fatbin,
    compress,
    kernel_cubin,
    pad_to,
)

METHODS = [None, "zstd", "lz4", "lz4frame", "zlib"]


@pytest.mark.parametrize("method", METHODS)
def test_entry_roundtrip(method):
    payload = kernel_cubin()
    blob = build_fatbin([build_entry(payload, method=method)])
    (fb,) = F.find_fatbins(blob)
    (entry,) = fb.entries
    assert entry.compression == method
    assert entry.arch == "sm_90"
    assert entry.payload()[: len(payload)] == payload
    assert fb.file_size == len(blob)
    assert entry.file_size == len(blob) - 16
    if method not in (None, "zlib"):
        assert entry.stored_size < entry.padded_size + 8


@pytest.mark.parametrize("method", ["lz4", "lz4frame"])
def test_pure_python_lz4_matches_library(method):
    data = os.urandom(3000) + b"abc" * 5000 + kernel_cubin() * 3
    code, _ = compress(data, method)
    decode = F._lz4_block_py if method == "lz4" else F._lz4_frame_py
    assert decode(code, len(data)) == data


def test_arch_names():
    def arch(**kw):
        (fb,) = F.find_fatbins(build_fatbin([build_entry(b"x" * 8, **kw)]))
        return fb.entries[0].arch

    assert arch(arch=90, flags=F.FLAG_ARCH_SPECIFIC) == "sm_90a"
    assert arch(arch=100, flags=F.FLAG_FAMILY_SPECIFIC) == "sm_100f"
    assert arch(arch=90, kind=F.KIND_PTX) == "compute_90"
    assert arch(arch=90, kind=F.KIND_LTOIR) == "lto_90"


def test_identifier():
    (fb,) = F.find_fatbins(build_fatbin([build_entry(b"x" * 8, identifier="a.cu")]))
    assert fb.entries[0].identifier == "a.cu"
    assert fb.entries[0].payload() == b"x" * 8


def test_compression_is_sniffed_from_content():
    for method in METHODS[1:]:
        (fb,) = F.find_fatbins(build_fatbin([build_entry(b"x" * 64, method=method)]))
        assert fb.entries[0].compression == method


def test_dedicated_section_with_padding():
    a = pad_to(build_fatbin([build_entry(kernel_cubin())]), 256)
    b = build_fatbin(
        [
            build_entry(kernel_cubin(), method="zstd"),
            build_entry(b"p" * 40, kind=F.KIND_PTX),
        ]
    )
    blob = a + b
    fatbins = F.find_fatbins(blob)
    assert [fb.offset for fb in fatbins] == [0, len(a)]
    assert [len(fb.entries) for fb in fatbins] == [1, 2]


def test_strict_scan_skips_false_positives():
    fb = build_fatbin([build_entry(kernel_cubin(), method="lz4")])
    junk = b"junk" + F.FATBIN_MAGIC_BYTES + b"\xff" * 64
    blob = junk + fb + junk + fb[:-8] + b"tail"
    found = F.find_fatbins(blob, strict=True)
    assert [f.offset for f in found] == [len(junk)]
    with pytest.raises(F.FatbinError):
        F.find_fatbins(junk + fb)  # non-strict: junk is an error
