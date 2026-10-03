import json
import sys

import pytest

from cubloaty import cli
from cubloaty import fatbin as F

from .builders import (
    PTX,
    build_entry,
    build_fatbin,
    build_host_elf,
    kernel_cubin,
    pad_to,
    requires_demangler,
)

K = "_Z6kernelPf"
pytestmark = requires_demangler


@pytest.fixture
def library(tmp_path):
    fatbins = b"".join(
        pad_to(
            build_fatbin(
                [
                    build_entry(
                        kernel_cubin([K, f"_Z2u{i}v"], tu=f"t{i}.cu"), method="zstd"
                    ),
                    build_entry(
                        kernel_cubin([K], flags=0x0600640A),
                        arch=100,
                        flags=F.FLAG_ARCH_SPECIFIC | 0x11,
                    ),
                    build_entry(
                        PTX.replace(b"_4_a_cu_", f"_5_t{i}_cu_".encode()),
                        kind=F.KIND_PTX,
                        method="lz4",
                    ),
                ]
            ),
            256,
        )
        for i in range(2)
    )
    path = tmp_path / "libdemo.so"
    path.write_bytes(
        build_host_elf([(".text", b"\x90" * 4096), (".nv_fatbin", fatbins)])
    )
    return path


def run(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["cubloaty", *map(str, args)])
    monkeypatch.setenv("COLUMNS", "160")
    cli.main()
    return capsys.readouterr()


def test_json(monkeypatch, capsys, library):
    out = json.loads(run(monkeypatch, capsys, library, "--format", "json").out)
    assert out["file_size"] == library.stat().st_size
    assert out["host_file_size"] + out["device_file_size"] == out["file_size"]
    assert set(out["architectures"]) == {"sm_90", "sm_100a", "compute_90"}
    assert sum(c["file_size"] for c in out["categories"]) == out["device_file_size"]

    kernels = {k["name"]: k for k in out["kernels"]}
    kern = kernels["kernel(float*)"]
    assert kern["kind"] == "kernel"
    assert kern["mangled"] == [K]
    assert kern["by_arch"]["sm_90"]["copies"] == 2
    assert kernels["helper(float)"]["kind"] == "device_function"

    dups = {(d["name"], d["arch"]): d for d in out["duplicates"]}
    assert set(dups) == {
        ("kernel(float*)", "sm_90"),
        ("kernel(float*)", "sm_100a"),
        ("kernel(float*)", "compute_90"),
        ("helper(float)", "compute_90"),
    }
    assert sorted(
        loc["tu"] for loc in dups[("kernel(float*)", "sm_90")]["locations"]
    ) == [
        "t0.cu",
        "t1.cu",
    ]


def test_json_filters(monkeypatch, capsys, library):
    out = json.loads(
        run(monkeypatch, capsys, library, "-f", "json", "--arch", "sm_100a").out
    )
    assert list(out["architectures"]) == ["sm_100a"]
    out = json.loads(run(monkeypatch, capsys, library, "-f", "json", "-r", "^u0").out)
    assert [k["name"] for k in out["kernels"]] == ["u0()"]
    assert out["duplicates"] == []


def test_tables(monkeypatch, capsys, library):
    out = run(monkeypatch, capsys, library, "--top", "3").out
    for title in (
        "File Composition",
        "Device Code Breakdown",
        "Top Kernels (all architectures)",
        "Top Kernels (sm_100a)",
        "Duplicate Kernels",
    ):
        assert title in out
    assert "kernel(float*)" in out
    assert "t0.cu, t1.cu" in out


def test_top_zero(monkeypatch, capsys, library):
    out = run(monkeypatch, capsys, library, "--top", "0").out
    assert "TOTAL WASTED" in out


def test_no_color_is_ascii(monkeypatch, capsys, library):
    out = run(monkeypatch, capsys, library, "--no-color", "--mangled").out
    assert out.isascii()
    assert K in out


def test_errors(monkeypatch, capsys, library, tmp_path):
    with pytest.raises(SystemExit):
        run(monkeypatch, capsys, library, "--arch", "sm_75")
    assert "sm_75" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run(monkeypatch, capsys, tmp_path / "missing.so")
    empty = tmp_path / "host.so"
    empty.write_bytes(build_host_elf([(".text", b"\0" * 64)]))
    with pytest.raises(SystemExit):
        run(monkeypatch, capsys, empty)
    assert "No CUDA device code" in capsys.readouterr().err


def test_shorten_name():
    name = "void ns::detail::kern<ns::P<float, 4>, (ns::Algo)0>(ns::P<float, 4>, int)"
    assert cli.shorten_name(name, 200) == name[5:]
    assert (
        cli.shorten_name(name, 60)
        == "ns::detail::kern<P<float, 4>, (Algo)0>(P<float, 4>, int)"
    )
    assert cli.shorten_name(name, 40) == "ns::detail::kern<P<float, 4>, (Algo)0>"
    assert cli.shorten_name(name, 30) == "kern<P<float, 4>, (Algo)0>"
    assert cli.shorten_name("(anonymous namespace)::k(float*)", 12) == "k(float*)"
    assert cli.shorten_name("plain(int)", 5) == "plain(int)"
