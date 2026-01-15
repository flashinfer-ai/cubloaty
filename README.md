# cubloaty

**Ever wondered what's making your CUDA binary big?**

Cubloaty is a size profiler for CUDA binaries. It analyzes `.so` files and `.cubin` files to show you the size of each kernel, broken down by architecture (sm_70, sm_80, sm_90, etc.).

Think of it as [bloaty](https://github.com/google/bloaty), but for CUDA kernels.

## Quick Example

```bash
$ cubloaty sampling.so

╭─────────────────────────────────────╮
│ 📊 CUDA Kernel Size Analysis Report │
╰─────────────────────────────────────╯
                Architecture Summary
╭─────────────────┬────────────┬─────────────────┬──────────────╮
│ Architecture    │    Kernels │      Total Size │   Percentage │
├─────────────────┼────────────┼─────────────────┼──────────────┤
│ SM_89           │        361 │           5.5MB │       100.0% │
├─────────────────┼────────────┼─────────────────┼──────────────┤
│ TOTAL           │        361 │           5.5MB │       100.0% │
╰─────────────────┴────────────┴─────────────────┴──────────────╯

                      Section Breakdown                       
╭───────────────────────────┬─────────────────┬──────────────╮
│ Section Type              │      Total Size │   % of Total │
├───────────────────────────┼─────────────────┼──────────────┤
│ Code Sections             │           4.3MB │        78.9% │
│ Metadata                  │         567.5KB │        10.1% │
│ Data Sections             │         510.4KB │         9.1% │
│ Debug Info                │          39.8KB │         0.7% │
├───────────────────────────┼─────────────────┼──────────────┤
│ TOTAL                     │           5.5MB │       100.0% │
╰───────────────────────────┴─────────────────┴──────────────╯

                              Top CUDA Kernels (All Architectures) - 361 Total
╭────────┬────────────────────────────────────────────────────────────────────────┬──────────────┬────────────╮
│   Rank │ Kernel Name                                                            │    Code Size │  % of Code │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼────────────┤
│      1 │ void flashinfer::sampling::TopKTopPSamplingFromProbKernel<1024u, (c... │       55.8KB │       1.2% │
│      2 │ void flashinfer::sampling::TopKSamplingFromProbKernel<1024u, (cub::... │       55.5KB │       1.2% │
│      3 │ void flashinfer::sampling::TopKTopPSamplingFromProbKernel<1024u, (c... │       52.9KB │       1.2% │
│      4 │ void flashinfer::sampling::TopKSamplingFromProbKernel<1024u, (cub::... │       52.6KB │       1.2% │
│      5 │ void flashinfer::sampling::TopKTopPSamplingFromProbKernel<512u, (cu... │       51.5KB │       1.1% │
│      6 │ void flashinfer::sampling::TopKSamplingFromProbKernel<512u, (cub::C... │       51.2KB │       1.1% │
│      7 │ void flashinfer::sampling::TopKTopPSamplingFromProbKernel<512u, (cu... │       46.4KB │       1.0% │
│      8 │ void flashinfer::sampling::TopKSamplingFromProbKernel<512u, (cub::C... │       46.2KB │       1.0% │
│      9 │ void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::... │       46.0KB │       1.0% │
│     10 │ void flashinfer::sampling::ChainSpeculativeSampling<1024u, (cub::CU... │       45.5KB │       1.0% │
│     11 │ void flashinfer::sampling::ChainSpeculativeSampling<512u, (cub::CUB... │       43.0KB │       1.0% │
│     12 │ void flashinfer::sampling::TopPSamplingFromProbKernel<1024u, (cub::... │       43.0KB │       1.0% │
│     13 │ void flashinfer::sampling::TopPSamplingFromProbKernel<512u, (cub::C... │       42.9KB │       1.0% │
│     14 │ void flashinfer::sampling::ChainSpeculativeSampling<1024u, (cub::CU... │       42.4KB │       0.9% │
│     15 │ void flashinfer::sampling::MinPSamplingFromProbKernel<1024u, (cub::... │       39.4KB │       0.9% │
│     16 │ void flashinfer::sampling::ChainSpeculativeSampling<512u, (cub::CUB... │       38.8KB │       0.9% │
│     17 │ void flashinfer::sampling::TopPRenormProbKernel<1024u, (cub::CUB_30... │       38.4KB │       0.9% │
│     18 │ void flashinfer::sampling::TopKTopPSamplingFromProbKernel<1024u, (c... │       38.1KB │       0.8% │
│     19 │ void flashinfer::sampling::TopPSamplingFromProbKernel<512u, (cub::C... │       38.0KB │       0.8% │
│     20 │ void flashinfer::sampling::TopKSamplingFromProbKernel<1024u, (cub::... │       37.9KB │       0.8% │
│     21 │ void flashinfer::sampling::MinPSamplingFromProbKernel<512u, (cub::C... │       36.9KB │       0.8% │
│     22 │ void flashinfer::sampling::TopKTopPSamplingFromProbKernel<1024u, (c... │       36.4KB │       0.8% │
│     23 │ void flashinfer::sampling::TopKSamplingFromProbKernel<1024u, (cub::... │       36.2KB │       0.8% │
│     24 │ void flashinfer::sampling::MinPSamplingFromProbKernel<1024u, (cub::... │       36.1KB │       0.8% │
│     25 │ void flashinfer::sampling::TopPRenormProbKernel<512u, (cub::CUB_300... │       34.5KB │       0.8% │
│     26 │ void flashinfer::sampling::TopKMaskLogitsKernel<1024u, (cub::CUB_30... │       34.2KB │       0.8% │
│     27 │ void flashinfer::sampling::TopKTopPSamplingFromProbKernel<512u, (cu... │       33.9KB │       0.8% │
│     28 │ void flashinfer::sampling::TopKSamplingFromProbKernel<512u, (cub::C... │       33.8KB │       0.7% │
│     29 │ void flashinfer::sampling::MinPSamplingFromProbKernel<512u, (cub::C... │       31.9KB │       0.7% │
│     30 │ void flashinfer::sampling::ChainSpeculativeSampling<1024u, (cub::CU... │       31.8KB │       0.7% │
│    ... │ (331 more kernels)                                                     │              │            │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼────────────┤
│        │ TOTAL KERNEL CODE                                                      │        4.4MB │   80.1% of │
│        │                                                                        │              │       file │
╰────────┴────────────────────────────────────────────────────────────────────────┴──────────────┴────────────╯

✓ Analysis complete!
```

## Features

- 📊 **Multi-architecture analysis** - See kernel sizes across sm_70, sm_80, sm_90, etc.
- 🔍 **Regex filtering** - Filter kernels by name pattern
- 📦 **Multiple formats** - `.so` libraries and standalone `.cubin` files
- 🎨 **Rich output** - Beautiful tables or JSON for scripting
- ⚡ **Fast** - Analyzes binaries in seconds

## Dependencies

Cubloaty requires the following tools to be installed and available in your `PATH`:

- **CUDA Toolkit** - for `cuobjdump` (part of the CUDA installation)
- **binutils** - for `objdump`, `objcopy`, and `readelf`
- **gcc/g++** - for `c++filt` (symbol demangling)

On Ubuntu/Debian:

```bash
sudo apt-get install binutils gcc
```

CUDA Toolkit can be downloaded from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

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

### Analyze a shared library

```bash
cubloaty libmykernel.so
```

### Analyze a cubin file

```bash
cubloaty kernel.sm_90.cubin
```

### Show top 50 kernels

```bash
cubloaty libmykernel.so --top 50
```

### Filter by architecture

```bash
cubloaty libmykernel.so --arch sm_90
```

### Filter kernels by name (regex)

```bash
# Find all GEMM kernels
cubloaty libmykernel.so --filter "gemm"

# Find attention-related kernels
cubloaty libmykernel.so --filter "attention|flash"
```

### Output as JSON

```bash
cubloaty libmykernel.so --format json > analysis.json
```

### Show full kernel names without truncation

```bash
cubloaty libmykernel.so --full-names
```

### Combine filters

```bash
# Show top 20 GEMM kernels for sm_90 in JSON format
cubloaty lib.so --arch sm_90 --filter "gemm" --top 20 --format json
```

## Advanced Examples

### Compare kernel sizes across architectures

```bash
# Show per-architecture breakdown
cubloaty libmykernel.so --verbose
```

### Find the largest kernels

```bash
# Show just the top 10
cubloaty libmykernel.so --top 10
```

### Export for further analysis

```bash
# Get JSON output and process with jq
cubloaty lib.so --format json | jq '.kernels[] | select(.size > 100000)'
```

## Options

```
  file                    Path to .so or .cubin file to analyze
  --top N, -n N          Show top N kernels (default: 30)
  --arch ARCH, -a ARCH   Filter by architecture (e.g., sm_90, sm_80)
  --filter REGEX, -r     Filter kernel names by regex (case-insensitive)
  --format {table,json}  Output format (default: table)
  --full-names           Show full kernel names without truncation
  --no-color             Disable colored output
  --verbose, -v          Show detailed processing information
  --version              Show version number
```

## How It Works

Cubloaty extracts CUDA fatbinary sections from shared libraries using `objdump` and `objcopy`, then uses `cuobjdump` to extract individual cubins for each architecture. It analyzes each cubin with `readelf` to extract kernel symbols and their sizes, and uses `c++filt` to demangle C++ symbol names.

## Contributing

Issues and pull requests are welcome!
