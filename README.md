# cubloaty

**Ever wondered what's making your CUDA binary big?**

Cubloaty is a size profiler for CUDA binaries. It analyzes `.so` files and `.cubin` files to show you the size of each kernel, broken down by architecture (sm_70, sm_80, sm_90, etc.).

Think of it as [bloaty](https://github.com/google/bloaty), but for CUDA kernels.

## Quick Example

```bash
$ cubloaty fused_moe_trtllm_sm100.so --top 5

fused_moe_trtllm_sm100.so: shared library, 18.6MB (19,528,232 bytes)

                                   File Composition
╭───────────────────────────┬─────────┬────────┬──────────────┬───────────┬───────────╮
│ Component                 │ Kernels │ Images │ Uncompressed │ File Size │ % of File │
├───────────────────────────┼─────────┼────────┼──────────────┼───────────┼───────────┤
│ sm_103a SASS              │    1952 │     10 │      115.8MB │     6.3MB │     33.7% │
│ Fatbin container overhead │         │        │              │     1.5KB │      0.0% │
│ Host code & data          │         │        │              │    12.4MB │     66.3% │
├───────────────────────────┼─────────┼────────┼──────────────┼───────────┼───────────┤
│ TOTAL                     │         │     10 │              │    18.6MB │    100.0% │
╰───────────────────────────┴─────────┴────────┴──────────────┴───────────┴───────────╯

                             Device Code Breakdown
╭─────────────────────────────────────┬──────────────┬───────────┬─────────────╮
│ Category                            │ Uncompressed │ File Size │ % of Device │
├─────────────────────────────────────┼──────────────┼───────────┼─────────────┤
│ SASS code (.text)                   │       77.1MB │     4.2MB │       66.5% │
│ .nv.capmerc sections                │       25.2MB │     1.3MB │       21.3% │
│ Symbol & string tables              │        5.5MB │   303.2KB │        4.7% │
│ .nv.merc sections                   │        2.8MB │   147.5KB │        2.3% │
│ Constant banks (.nv.constant*)      │        2.0MB │   109.1KB │        1.7% │
│ Kernel attributes (.nv.info, notes) │        1.4MB │    84.2KB │        1.3% │
│ ELF headers & padding               │      943.1KB │    51.9KB │        0.8% │
│ Debug info                          │      489.3KB │    27.7KB │        0.4% │
│ Initialized globals                 │      425.9KB │    50.4KB │        0.8% │
│ Relocations                         │       70.9KB │     4.1KB │        0.1% │
│ Fatbin headers & padding            │              │     2.6KB │        0.0% │
├─────────────────────────────────────┼──────────────┼───────────┼─────────────┤
│ TOTAL                               │      115.8MB │     6.3MB │      100.0% │
╰─────────────────────────────────────┴──────────────┴───────────┴─────────────╯

                                       Top Kernels (all architectures)
╭──────┬────────────────────────────────────────────────────────────┬──────────────┬───────────┬─────────────╮
│ Rank │ Kernel                                                     │ Uncompressed │ File Size │ % of Device │
├──────┼────────────────────────────────────────────────────────────┼──────────────┼───────────┼─────────────┤
│    1 │ routingIndicesClusterKernel<KernelParams<float, __nv_bflo… │      796.5KB │    46.6KB │        0.7% │
│    2 │ routingIndicesClusterKernel<KernelParams<__nv_bfloat16, _… │      796.4KB │    46.6KB │        0.7% │
│    3 │ routingIndicesClusterKernel<KernelParams<float, float, 20… │      771.7KB │    45.2KB │        0.7% │
│    4 │ routingIndicesClusterKernel<KernelParams<float, __nv_bflo… │      655.8KB │    38.4KB │        0.6% │
│    5 │ routingIndicesClusterKernel<KernelParams<float, float, 20… │      631.0KB │    36.9KB │        0.6% │
│  ... │ (1947 more)                                                │              │           │             │
├──────┼────────────────────────────────────────────────────────────┼──────────────┼───────────┼─────────────┤
│      │ TOTAL (1952 functions)                                     │      112.3MB │     6.0MB │       96.4% │
╰──────┴────────────────────────────────────────────────────────────┴──────────────┴───────────┴─────────────╯
```

Note how the 6.3MB of device code in the file expands to 115.8MB once
decompressed: cubloaty reports both, so you can see what actually ships in
your wheel and what the driver loads.

## Features

- 📏 **Accurate, exhaustive accounting** - Every byte of the file is attributed
  exactly once: host code, fatbin container overhead, and each kernel's code,
  constants, metadata, relocations, symbol names and debug info. Totals always
  add up to the file size.
- 🗜️ **Compression-aware** - Reports both the on-disk (compressed) size and the
  uncompressed size of fatbin entries (LZ4, ZSTD; LZ4 frame and zlib streams are
  recognized too).
- 🔁 **Duplicate kernel detection** - Finds kernels compiled into several
  translation units (e.g. header-defined kernels/templates) and how many bytes
  the extra copies waste, with the TUs they came from.
- 📊 **Multi-architecture analysis** - SASS per `sm_XX` (including `a`/`f`
  variants), PTX per `compute_XX`, and LTO-IR.
- 🔍 **Finds embedded device code** - Fatbins and cubins stored outside
  `.nv_fatbin` (e.g. in `.rodata`/`.data` for `cuModuleLoadData`, as cuFFT and
  cuBLASLt do) are found and analyzed.
- 📦 **Many formats** - Shared libraries, executables, object files, static
  libraries (`.a`), standalone `.cubin` and `.fatbin` files.
- 🎨 **Rich output** - Tables or JSON for scripting.
- ⚡ **Fast and self-contained** - Parses ELF and fatbin formats natively, no
  CUDA toolkit required; a 600MB library is analyzed in a few seconds.

## Understanding the Numbers

- **File Size** is what an item occupies in the analyzed file, i.e. what you
  pay in package/wheel size. **Uncompressed** is the size after decompressing
  fatbin entries (what the CUDA driver loads). They are equal for entries that
  are stored uncompressed.
- Compression is applied to whole images (a cubin with many kernels), so a
  single kernel's file size is an estimate: the image's compressed bytes are
  split across kernels in proportion to their uncompressed bytes.
- A kernel's size includes everything that exists only because of it: its
  `.text` (SASS), `.nv.capmerc`/`.nv.merc` sections (sm_100+), its parameter
  constant bank, `.nv.info` attributes, relocations, and its symbol and section
  names. Bytes shared by all kernels of an image (global constants, ELF headers,
  debug line tables, ...) are shown in the *Device Code Breakdown*.
- Shared memory (`.nv.shared.*`) and uninitialized globals occupy no space in
  the file and are not counted.
- Entries whose contents cannot be decoded (e.g. LTO-IR) are counted as a
  single opaque block per image, using the sizes from the fatbin entry header.
- *Kernels* are `__global__` functions. Non-inlined `__device__` functions that
  have their own sections (e.g. with `-rdc=true`) are listed as `[device fn]`.

## Duplicate Kernels

In whole-program compilation (the default), every translation unit that
launches a kernel defined in a header (templates, `static`/`inline` kernels,
anonymous namespaces) gets its own copy of the compiled kernel, and the linker
does not deduplicate them. cubloaty groups identical kernels per architecture
across TUs and reports how many bytes the extra copies cost:

```
                          Duplicate Kernels - 7 compiled into multiple TUs
╭──────┬────────────────────────┬─────────┬────────┬────────────────┬─────────────┬─────────────────╮
│ Rank │ Kernel                 │ Arch    │ Copies │ Wasted Uncomp. │ Wasted File │ TUs             │
├──────┼────────────────────────┼─────────┼────────┼────────────────┼─────────────┼─────────────────┤
│    1 │ scale_kernel<float, 4> │ sm_100a │      3 │          5.9KB │       5.9KB │ a.cu, b.cu, c.… │
│    2 │ header_kernel(float*)  │ sm_100a │      3 │          4.3KB │       4.3KB │ a.cu, b.cu, c.… │
│    3 │ scale_kernel<float, 4> │ sm_90   │      3 │          3.7KB │       3.7KB │ a.cu, b.cu, c.… │
│    4 │ header_kernel(float*)  │ sm_90   │      3 │          2.7KB │       2.7KB │ a.cu, b.cu, c.… │
│  ... │ (3 more)               │         │        │                │             │                 │
├──────┼────────────────────────┼─────────┼────────┼────────────────┼─────────────┼─────────────────┤
│      │ TOTAL WASTED           │         │        │         20.8KB │      17.6KB │                 │
╰──────┴────────────────────────┴─────────┴────────┴────────────────┴─────────────┴─────────────────╯
 Every TU that instantiates a header-defined kernel embeds its own copy; define it in a single TU to
                                              keep one.
```

Typical fixes are to define (explicitly instantiate) the kernel in a single
`.cu` file and launch it through a host function declared in the header, or to
build with `-rdc=true` so the device linker merges template instantiations
(kernels with internal linkage, i.e. `static` or in an anonymous namespace,
still get one copy per TU). Relocatable images in `__nv_relfatbin` (inputs to
device linking) are not counted as duplicates.

## Dependencies

Python dependencies (`rich`, `lz4`, and `zstandard` on Python < 3.14) are
installed automatically. No CUDA toolkit is needed: ELF and fatbin formats are
parsed natively.

For demangled kernel names, `c++filt` (from binutils) should be on your `PATH`.
NVIDIA's `cu++filt` (CUDA toolkit, also looked up in `$CUDA_HOME/bin`) is used
for names GNU `c++filt` cannot handle, such as long CUTLASS instantiations.

## Installation

Install the package from pypi:

```
pip install cubloaty
```

Or git clone the repo and install from source:
```bash
git clone https://github.com/flashinfer-ai/cubloaty.git
pip install -e . -v  # editable mode
```

## Usage

### Analyze a shared library, object file or static library

```bash
cubloaty libmykernel.so
cubloaty kernels.o
cubloaty libmykernels.a       # per-object TUs in the duplicates report
```

### Analyze a cubin or fatbin file

```bash
cubloaty kernel.sm_90.cubin
cubloaty kernels.fatbin
```

### Show top 50 kernels

```bash
cubloaty libmykernel.so --top 50
```

### Filter by architecture

```bash
cubloaty libmykernel.so --arch sm_90
cubloaty libmykernel.so --arch compute_90   # PTX
```

### Filter kernels by name (regex)

```bash
# Find all GEMM kernels
cubloaty libmykernel.so --filter "gemm"

# Find attention-related kernels
cubloaty libmykernel.so --filter "attention|flash"
```

### Rank by uncompressed size

```bash
cubloaty libmykernel.so --sort size
```

### Output as JSON

```bash
cubloaty libmykernel.so --format json > analysis.json
```

### Show full kernel names without truncation

```bash
cubloaty libmykernel.so --full-names
cubloaty libmykernel.so --mangled     # mangled names, e.g. to match nm output
```

By default, long names are shortened to fit the terminal by dropping
namespaces inside template arguments and, if needed, redundant parameter lists.

### Combine filters

```bash
# Show top 20 GEMM kernels for sm_90 in JSON format
cubloaty lib.so --arch sm_90 --filter "gemm" --top 20 --format json
```

## Advanced Examples

### Find the largest kernels

```bash
# Show just the top 10
cubloaty libmykernel.so --top 10
```

### Export for further analysis

```bash
# Kernels larger than 100KB in the file
cubloaty lib.so --format json | jq '.kernels[] | select(.file_size > 100000)'

# Bytes wasted by duplicate kernels
cubloaty lib.so --format json | jq '[.duplicates[].wasted_file_size] | add'
```

## Options

```
  file                    Path to a .so/.o/.a/executable, .cubin or .fatbin file
  --top N, -n N          Show top N kernels (default: 30)
  --arch ARCH, -a ARCH   Filter by architecture (e.g., sm_90, sm_100a, compute_90)
  --filter REGEX, -r     Filter kernel names by regex (case-insensitive)
  --sort {file,size}     Rank by bytes in the file (default) or uncompressed size
  --format {table,json}  Output format (default: table)
  --full-names           Show full kernel names without truncation
  --mangled              Show mangled instead of demangled names
  --no-color             Disable colored output (plain ASCII tables)
  --verbose, -v          Show detailed processing information
  --version              Show version number
```

## JSON Output

`--format json` emits all data (not limited by `--top`):

- `file_size`, `host_file_size`, `device_file_size`: top-level split of the file
- `components`: per (section, kind, arch) image totals, e.g. `sm_90` SASS in `.nv_fatbin`
- `architectures`: per-arch `size` (uncompressed), `file_size`, `images`, `kernel_count`
- `categories`: device bytes by category (code, capmerc, constant, symbols, ptx, ...)
- `kernels`: per function `name`, `mangled`, `kind` (`kernel`/`device_function`),
  `size`, `file_size`, `by_arch` (with `copies`), `by_category`
- `duplicates`: per (kernel, arch) `copies`, `wasted_size`, `wasted_file_size`
  and `locations` (TU and image of each copy)

## How It Works

Cubloaty reads the input with its own ELF parser, locates fatbin data in
`.nv_fatbin`/`__nv_relfatbin` (and validated fatbins/cubins embedded in any
other section), and walks the fatbin container format. Each entry is
decompressed and parsed: cubins are attributed section by section (sections
named after a kernel, or linked to one via `sh_info`, belong to it;
overlapping sections are counted once), PTX is split at `.entry`/`.func`
boundaries. Translation units are inferred from archive member names, fatbin
identifiers, or the TU name nvcc embeds in internal-linkage symbols. Names are
demangled in one batch with `c++filt`/`cu++filt`.

## Contributing

Issues and pull requests are welcome! Run the tests with:

```bash
pip install -e ".[test]"
pytest tests          # tests using nvcc are skipped if it is not on PATH
```
