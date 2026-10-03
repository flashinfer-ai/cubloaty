"""End-to-end tests against real nvcc output (skipped without a CUDA toolkit)"""

import glob
import hashlib
import mmap
import os
import shutil
import subprocess

import pytest

from cubloaty import analysis as A
from cubloaty import elf as E
from cubloaty.fatbin import KIND_ELF, find_fatbins

from .test_analysis import check_exact


def _find_nvcc():
    """An nvcc that can actually compile ($CUDA_HOME first, then PATH)"""
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    for nvcc in (os.path.join(cuda, "bin", "nvcc"), shutil.which("nvcc")):
        if nvcc is None:
            continue
        try:
            subprocess.run(
                [nvcc, "-cubin", "-x", "cu", "-", "-o", "/dev/null"],
                input=b"__global__ void k() {}",
                check=True,
                capture_output=True,
                timeout=120,
            )
            return nvcc
        except (OSError, subprocess.SubprocessError):
            continue
    return None


NVCC = _find_nvcc()
pytestmark = pytest.mark.skipif(NVCC is None, reason="no working nvcc")

HEADER = """
#pragma once
template <typename T, int N>
__global__ void scale_kernel(T* x, T a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  #pragma unroll
  for (int k = 0; k < N; ++k) if (i * N + k < n) x[i * N + k] *= a;
}
static __device__ __noinline__ float helper(float v) { return v * v + 1.0f; }
static __global__ void header_kernel(float* x) { x[threadIdx.x] = helper(x[threadIdx.x]); }
namespace { __global__ void anon_kernel(float* x) { x[threadIdx.x] += 1.0f; } }
"""

SOURCES = {
    "a.cu": """#include "kern.cuh"
__global__ void only_in_a(float* x) { x[threadIdx.x] = 1.0f; }
void launch_a(float* x) {
  scale_kernel<float, 4><<<1, 32>>>(x, 2.0f, 32);
  header_kernel<<<1, 32>>>(x); anon_kernel<<<1, 32>>>(x); only_in_a<<<1, 1>>>(x);
}""",
    "b.cu": """#include "kern.cuh"
void launch_b(float* x) {
  scale_kernel<float, 4><<<1, 32>>>(x, 3.0f, 32);
  header_kernel<<<1, 32>>>(x); anon_kernel<<<1, 32>>>(x);
}""",
}


def build(tmp_path, *flags):
    (tmp_path / "kern.cuh").write_text(HEADER)
    for name, src in SOURCES.items():
        (tmp_path / name).write_text(src)
    out = tmp_path / "lib.so"
    subprocess.run(
        [NVCC, "-shared", "-Xcompiler", "-fPIC", "-diag-suppress", "20050", *flags]
        + list(SOURCES)
        + ["-o", str(out)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return out


@pytest.mark.parametrize("compress", ["none", "size", "speed"])
def test_duplicate_header_kernels(tmp_path, compress):
    lib = build(tmp_path, "-arch=sm_90", f"--compress-mode={compress}")
    report = A.analyze_file(str(lib))
    check_exact(report)

    dups = {(d.arch, d.name): d for d in A.find_duplicates(report, report.images)}
    for name in (
        "void scale_kernel<float, 4>(float*, float, int)",
        "header_kernel(float*)",
        "(anonymous namespace)::anon_kernel(float*)",
    ):
        dup = dups[("sm_90", name)]
        assert sorted(A.function_tu(i, f) for i, f in dup.copies) == ["a.cu", "b.cu"]
        assert dup.wasted_size > 0
    assert not any("only_in_a" in name for _, name in dups)


def test_rdc_static_kernels_are_duplicates(tmp_path):
    lib = build(tmp_path, "-arch=sm_90", "-rdc=true")
    report = A.analyze_file(str(lib))
    check_exact(report)
    dups = {(d.arch, d.name): d for d in A.find_duplicates(report, report.images)}
    # nvlink merges template instantiations but keeps per-TU static kernels
    assert ("sm_90", "void scale_kernel<float, 4>(float*, float, int)") not in dups
    dup = dups[("sm_90", "header_kernel(float*)")]
    assert sorted(A.function_tu(i, f) for i, f in dup.copies) == ["a.cu", "b.cu"]


@pytest.mark.skipif(shutil.which("cuobjdump") is None, reason="cuobjdump not found")
def test_cubins_match_cuobjdump(tmp_path):
    lib = build(tmp_path, "-arch=sm_90", "-Xfatbin", "-compress-all")
    report = A.analyze_file(str(lib))
    check_exact(report)

    extract = tmp_path / "x"
    extract.mkdir()
    subprocess.run(
        ["cuobjdump", "-xelf", "all", str(lib)],
        cwd=extract,
        check=True,
        capture_output=True,
    )
    theirs = sorted(
        hashlib.sha256(open(p, "rb").read()).hexdigest()
        for p in glob.glob(str(extract / "*.cubin"))
    )

    with open(lib, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    sec = E.ElfFile(buf).section_by_name(".nv_fatbin")
    ours = []
    for fb in find_fatbins(buf, sec.offset, sec.offset + sec.size):
        for entry in fb.entries:
            if entry.kind == KIND_ELF:
                data = entry.payload()
                ours.append(
                    hashlib.sha256(data[: E.elf_extent(data, 0, len(data))]).hexdigest()
                )
    assert sorted(ours) == theirs

    # Every kernel owns its .text section
    sass = [i for i in report.images if i.kind == "sass"]
    kernels = [fn for img in sass for fn in img.functions.values() if fn.is_kernel]
    assert kernels and all(fn.size["code"] > 0 for fn in kernels)
