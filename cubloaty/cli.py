"""
cubloaty - Analyze CUDA binary sizes in .so files
Similar to bloaty but for CUDA kernels
"""

import subprocess
import sys
import tempfile
import os
import argparse
import json
from collections import defaultdict
import re
import logging

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.logging import RichHandler

# Get logger for this module
logger = logging.getLogger("cubloaty")


def setup_logging(verbose=False):
    """Setup logging configuration with Rich handler

    Rich automatically detects terminal capabilities and falls back to
    plain text when output is redirected or terminal doesn't support colors.

    Args:
        verbose: If True, set level to DEBUG, otherwise WARNING (quiet by default)
    """
    level = logging.DEBUG if verbose else logging.WARNING

    # Clear any existing handlers
    logger.handlers.clear()

    # Always use Rich handler - it handles plain terminals gracefully
    handler = RichHandler(
        rich_tracebacks=True,
        show_time=False,
        show_path=False,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.setLevel(level)


def extract_cubins(so_file):
    """Extract cubin sections from .so file"""
    cubins = []

    # Use objcopy to extract .nv_fatbin sections
    try:
        result = subprocess.run(
            ["objdump", "-h", so_file], capture_output=True, text=True, check=True
        )

        # Find all CUDA-related sections
        for line in result.stdout.split("\n"):
            if ".nv_fatbin" in line or "nv_fatbin" in line:
                # Extract section name
                parts = line.split()
                if len(parts) > 1:
                    section_name = parts[1]
                    cubins.append(section_name)
    except subprocess.CalledProcessError as e:
        logger.error(f"Could not read sections from {so_file}: {e}")
        return []

    return cubins


def extract_cubin_data(so_file, section_name, output_file):
    """Extract cubin binary data from section"""
    try:
        subprocess.run(
            ["objcopy", "--dump-section", f"{section_name}={output_file}", so_file],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def extract_cubins_from_fatbin(fatbin_file, output_dir):
    """Extract individual cubins from a fatbin using cuobjdump

    This function attempts to extract CUDA ELF binaries from a fatbin container.
    If cuobjdump is not available, it logs an error and returns empty list.
    """
    try:
        # First, list all ELF files in the fatbin
        result = subprocess.run(
            ["cuobjdump", "-lelf", fatbin_file],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the output to get cubin names
        cubin_names = []
        for line in result.stdout.split("\n"):
            if "ELF file" in line:
                # Extract the filename from "ELF file    1: filename.cubin"
                parts = line.split(":", 1)
                if len(parts) == 2:
                    cubin_name = parts[1].strip()
                    cubin_names.append(cubin_name)

        if not cubin_names:
            logger.debug(f"No ELF files found in fatbin: {fatbin_file}")
            return []

        # Extract all cubins at once using 'all'
        # cuobjdump will extract them to the current directory
        old_cwd = os.getcwd()
        try:
            os.chdir(output_dir)
            subprocess.run(
                ["cuobjdump", "-xelf", "all", fatbin_file],
                capture_output=True,
                check=True,
            )

            # Find all extracted .cubin files
            extracted_files = []
            for cubin_name in cubin_names:
                cubin_path = os.path.join(output_dir, cubin_name)
                if os.path.exists(cubin_path):
                    extracted_files.append((cubin_name, cubin_path))

            logger.debug(f"Extracted {len(extracted_files)} cubin(s) from fatbin")
            return extracted_files
        finally:
            os.chdir(old_cwd)

    except subprocess.CalledProcessError as e:
        logger.debug(f"Failed to extract cubins from fatbin: {e}")
        return []
    except FileNotFoundError:
        logger.error(
            "cuobjdump not found. Please ensure CUDA toolkit is installed and in PATH."
        )
        return []


def demangle_symbol(symbol):
    """Demangle C++ symbol names

    Uses c++filt to demangle C++ symbols. If demangling fails,
    returns the original symbol name.
    """
    try:
        result = subprocess.run(
            ["c++filt", symbol], capture_output=True, text=True, check=True
        )
        demangled = result.stdout.strip()
        return demangled if demangled else symbol
    except Exception as e:
        # Symbol demangling failure is not critical, just return original
        logger.debug(f"Failed to demangle symbol {symbol}: {e}")
        return symbol


def analyze_cubin_sizes(cubin_file):
    """Analyze a single cubin file and return symbol sizes and section breakdown

    This function uses readelf to parse ELF sections and symbols from a cubin file.
    If the file is not a valid ELF (e.g., it's a fatbin container), it returns
    empty results and the caller should try extracting cubins from it.

    Returns:
        tuple: (symbols_dict, text_section_size)
            - symbols_dict: Dictionary mapping symbol names to sizes (includes special sections)
            - text_section_size: Total size of .text.* sections from section headers
    """
    symbols = {}

    # Parse all sections using readelf
    section_sizes = {}
    try:
        result = subprocess.run(
            ["readelf", "-SW", cubin_file], capture_output=True, text=True, check=True
        )

        # Parse section headers to get sizes
        for line in result.stdout.split("\n"):
            # Match lines like:  [ 5] .debug_line  PROGBITS  0000000000000000 009298 039e76 00  0  0  1
            match = re.match(
                r"\s+\[\s*\d+\]\s+(\S+)\s+\S+\s+\S+\s+\S+\s+([0-9a-f]+)", line
            )
            if match:
                section_name = match.group(1)
                size_hex = match.group(2)
                size = int(size_hex, 16)
                if size > 0:
                    section_sizes[section_name] = size
    except subprocess.CalledProcessError:
        # This is expected for fatbin files - caller will try extracting cubins
        logger.debug(f"Could not parse ELF sections from {cubin_file}, may be a fatbin")
        return {}, 0
    except FileNotFoundError:
        logger.error("readelf not found. Please ensure binutils is installed.")
        return {}, 0

    # Categorize sections
    code_size = 0
    debug_size = 0
    data_size = 0
    metadata_size = 0

    for section_name, size in section_sizes.items():
        # Code sections (including MERC compressed code)
        if section_name.startswith(".text.") or section_name.startswith(
            ".nv.capmerc.text."
        ):
            code_size += size
        # Debug sections (including MERC debug sections)
        elif (
            section_name.startswith(".debug_")
            or section_name.startswith(".nv_debug_")
            or section_name.startswith(".nv.debug_")
            or section_name.startswith(".nv.merc.debug_")
            or section_name.startswith(".nv.merc.nv_debug_")
        ):
            debug_size += size
        # Data sections
        elif (
            section_name.startswith(".nv.shared.")
            or section_name.startswith(".nv.constant")
            or section_name.startswith(".nv.global")
        ):
            data_size += size
        # Metadata sections (including MERC metadata)
        elif (
            section_name in [".symtab", ".strtab", ".shstrtab"]
            or section_name.startswith(".nv.info")
            or section_name.startswith(".nv.merc.nv.info")
            or section_name.startswith(".nv.merc.symtab")
            or section_name.startswith(".nv.merc.strtab")
            or section_name.startswith(".nv.merc.shstrtab")
            or section_name.startswith(".rela.")
        ):
            metadata_size += size

    # Get function symbols using readelf
    try:
        result = subprocess.run(
            ["readelf", "-sW", cubin_file], capture_output=True, text=True, check=True
        )

        # Parse readelf output to extract function names and sizes
        for line in result.stdout.split("\n"):
            # Look for FUNC entries
            if "FUNC" in line:
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        # The size is typically the 3rd field (index 2)
                        size = int(parts[2], 0)  # 0 base to auto-detect hex/dec
                        # The symbol name is the last part
                        name = parts[-1]
                        if size > 0:  # Only include functions with non-zero size
                            # Demangle the symbol
                            demangled = demangle_symbol(name)
                            symbols[demangled] = size
                    except (ValueError, IndexError):
                        # Skip malformed symbol entries
                        continue

        # Add section breakdown as special entries
        # Use a special prefix to avoid Rich markup interpretation
        if debug_size > 0:
            symbols["<Debug Info>"] = debug_size
        if data_size > 0:
            symbols["<Data Sections>"] = data_size
        if metadata_size > 0:
            symbols["<Metadata>"] = metadata_size

        return symbols, code_size
    except subprocess.CalledProcessError:
        # This is expected for fatbin files - caller will try extracting cubins
        logger.debug(f"Could not analyze symbols from {cubin_file}")
        return {}, 0
    except FileNotFoundError:
        logger.error("readelf not found. Please ensure binutils is installed.")
        return {}, 0


def format_size(size_bytes):
    """Format size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def extract_sm_arch(cubin_name):
    """Extract SM architecture from cubin filename (e.g., 'sm_90a' from 'kernel.sm_90a.cubin')"""
    match = re.search(r"\.sm_(\d+[a-z]?)\.cubin", cubin_name)
    if match:
        return f"sm_{match.group(1)}"
    return "unknown"


def get_cubin_arch(cubin_file):
    """Get architecture from cubin file using cuobjdump

    Tries to determine SM architecture using cuobjdump, falls back to
    parsing the filename if cuobjdump is not available.
    """
    try:
        result = subprocess.run(
            ["cuobjdump", "-lelf", cubin_file],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse output like "ELF file    1: kernel.sm_90.cubin"
        for line in result.stdout.split("\n"):
            if "ELF file" in line and ".sm_" in line:
                match = re.search(r"\.sm_(\d+[a-z]?)\.cubin", line)
                if match:
                    return f"sm_{match.group(1)}"
        return "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Fallback to filename-based detection
        logger.debug(f"Could not get architecture from cuobjdump, using filename: {e}")
        return extract_sm_arch(cubin_file)


def shorten_kernel_name(name, max_length=80):
    """Shorten kernel name for display"""
    if len(name) <= max_length:
        return name
    # Try to extract the main function name
    # For templates like ClassName<Args>::method, try to keep the most important part
    if "::" in name:
        parts = name.split("::")
        if len(parts[-1]) < max_length:
            return "..." + "::".join(parts[-2:])
    return name[: max_length - 3] + "..."


def output_json(
    all_symbols,
    symbols_by_arch,
    arch_totals,
    special_sections=None,
    actual_kernels=None,
    kernel_counts_by_arch=None,
    total_kernel_count=0,
):
    """Output results in JSON format"""
    total_size = sum(all_symbols.values())
    kernels_total_size = sum(actual_kernels.values()) if actual_kernels else total_size
    special_total_size = sum(special_sections.values()) if special_sections else 0

    result = {
        "total_size": total_size,
        "total_size_formatted": format_size(total_size),
        "kernel_code_size": kernels_total_size,
        "kernel_code_size_formatted": format_size(kernels_total_size),
        "non_code_size": special_total_size,
        "non_code_size_formatted": format_size(special_total_size),
        "total_kernels": total_kernel_count,
        "architectures": {},
        "non_code_sections": [],
        "kernels": [],
    }

    # Architecture summary
    for arch in sorted(arch_totals.keys()):
        size = arch_totals[arch]
        percentage = (
            (size / sum(arch_totals.values()) * 100)
            if sum(arch_totals.values()) > 0
            else 0
        )
        kernel_count = (
            kernel_counts_by_arch.get(arch, 0) if kernel_counts_by_arch else 0
        )
        result["architectures"][arch] = {
            "size": size,
            "size_formatted": format_size(size),
            "percentage": round(percentage, 2),
            "kernel_count": kernel_count,
        }

    # Non-code sections
    if special_sections:
        sorted_special = sorted(
            special_sections.items(), key=lambda x: x[1], reverse=True
        )
        for name, size in sorted_special:
            percentage = (size / total_size * 100) if total_size > 0 else 0
            result["non_code_sections"].append(
                {
                    "name": name.strip("<>"),
                    "size": size,
                    "size_formatted": format_size(size),
                    "percentage_of_total": round(percentage, 2),
                }
            )

    # Actual CUDA kernels
    if actual_kernels:
        sorted_kernels = sorted(
            actual_kernels.items(), key=lambda x: x[1], reverse=True
        )
        for name, size in sorted_kernels:
            percentage = (
                (size / kernels_total_size * 100) if kernels_total_size > 0 else 0
            )
            kernel_info = {
                "name": name,
                "size": size,
                "size_formatted": format_size(size),
                "percentage_of_code": round(percentage, 2),
            }

            # Add per-arch breakdown if available
            kernel_info["by_arch"] = {}
            for arch in symbols_by_arch:
                if name in symbols_by_arch[arch]:
                    kernel_info["by_arch"][arch] = symbols_by_arch[arch][name]

            result["kernels"].append(kernel_info)

    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CUDA binary sizes in .so files - bloaty for CUDA kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cubloaty library.so                    # Analyze CUDA kernels in .so
  cubloaty kernel.cubin                  # Analyze single .cubin file
  cubloaty library.so --top 50           # Show top 50 kernels
  cubloaty library.so --arch sm_90       # Filter by architecture
  cubloaty library.so --filter "gemm"    # Filter kernels by name (regex)
  cubloaty library.so --format json      # Output as JSON
  cubloaty library.so --full-names       # Show full kernel names
        """,
    )

    parser.add_argument("file", help="Path to .so or .cubin file to analyze")
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=30,
        metavar="N",
        help="Show top N kernels (default: 30)",
    )
    parser.add_argument(
        "--arch",
        "-a",
        type=str,
        metavar="ARCH",
        help="Filter by architecture (e.g., sm_90, sm_80)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--filter",
        "-r",
        type=str,
        metavar="REGEX",
        help="Filter kernel names by regular expression (case-insensitive)",
    )
    parser.add_argument(
        "--full-names",
        action="store_true",
        help="Show full kernel names without truncation",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed processing information",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    args = parser.parse_args()

    input_file = args.file

    # Setup logging - Rich handler auto-detects terminal capabilities
    setup_logging(verbose=args.verbose)

    # Validate input file
    if not os.path.exists(input_file):
        logger.error(f"File not found: {input_file}")
        sys.exit(1)

    # Create console for formatted output (tables, etc.)
    # Only tables need to respect --no-color flag
    use_rich = not args.no_color and args.format == "table"
    console = Console() if use_rich else None

    # Check if input is a cubin file
    is_cubin = input_file.endswith(".cubin")

    file_type = "cubin file" if is_cubin else "shared library"
    logger.debug(
        f"[bold cyan]🔍 Analyzing CUDA binaries:[/bold cyan] {os.path.basename(input_file)} ({file_type})"
    )

    # Track symbols by architecture and overall
    symbols_by_arch = defaultdict(lambda: defaultdict(int))
    all_symbols = defaultdict(int)
    arch_totals = defaultdict(int)
    text_section_size_by_arch = defaultdict(int)  # Track actual .text section sizes
    total_text_section_size = 0

    if is_cubin:
        # Direct cubin analysis
        logger.debug("Processing cubin file...")

        # Try to analyze directly
        symbols, text_size = analyze_cubin_sizes(input_file)

        if symbols:
            logger.debug(f"Found {len(symbols)} symbol(s) in cubin")

            # Extract architecture from cubin file
            arch = get_cubin_arch(input_file)

            for name, size in symbols.items():
                all_symbols[name] += size
                symbols_by_arch[arch][name] += size
                arch_totals[arch] += size

            text_section_size_by_arch[arch] += text_size
            total_text_section_size += text_size
        else:
            logger.error("No symbols found in cubin file")
            sys.exit(1)
    else:
        # .so file processing (existing logic)
        # Extract cubin sections
        sections = extract_cubins(input_file)

        if not sections:
            logger.error("No CUDA binary sections found in the file.")
            logger.debug("Trying to extract using cuobjdump...")
            try:
                subprocess.run(["cuobjdump", "-elf", input_file], check=True)
                logger.info("Use cuobjdump -elf <file> to extract cubins manually")
            except Exception:
                pass
            sys.exit(1)

        logger.debug(f"Found {len(sections)} CUDA binary section(s)")

        # Process each section
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, section in enumerate(sections):
                logger.debug(f"Processing section: {section}")

                cubin_file = os.path.join(tmpdir, f"cubin_{i}.bin")

                if not extract_cubin_data(input_file, section, cubin_file):
                    logger.warning(f"Could not extract {section}")
                    continue

                # First try to analyze directly (may be an ELF cubin)
                symbols, text_size = analyze_cubin_sizes(cubin_file)

                # If that failed, it's likely a fatbin container, try extracting cubins from it
                if not symbols:
                    logger.debug("File is a fatbin, extracting individual cubins...")
                    extracted_cubins = extract_cubins_from_fatbin(cubin_file, tmpdir)

                    if extracted_cubins:
                        logger.debug(
                            f"Extracted {len(extracted_cubins)} cubin(s) from fatbin"
                        )

                        # Group by architecture
                        arch_groups = defaultdict(list)
                        for cubin_name, cubin_path in extracted_cubins:
                            arch = extract_sm_arch(cubin_name)
                            arch_groups[arch].append((cubin_name, cubin_path))

                        for arch in sorted(arch_groups.keys()):
                            cubins = arch_groups[arch]
                            arch_kernel_count = 0
                            for cubin_name, cubin_path in cubins:
                                cubin_symbols, cubin_text_size = analyze_cubin_sizes(
                                    cubin_path
                                )
                                if cubin_symbols:
                                    arch_kernel_count += len(cubin_symbols)
                                    for name, size in cubin_symbols.items():
                                        symbols_by_arch[arch][name] += size
                                        all_symbols[name] += size
                                        arch_totals[arch] += size
                                    text_section_size_by_arch[arch] += cubin_text_size
                                    total_text_section_size += cubin_text_size

                            logger.debug(
                                f"  {arch}: {len(cubins)} cubin(s), {arch_kernel_count} kernel(s), {format_size(arch_totals[arch])}"
                            )
                    else:
                        logger.warning(f"No symbols found in {section}")
                    continue

                logger.debug(f"Found {len(symbols)} symbol(s) in section")

                for name, size in symbols.items():
                    all_symbols[name] += size

                total_text_section_size += text_size

    # Filter by architecture if specified
    if args.arch:
        if args.arch not in symbols_by_arch:
            logger.error(
                f"Architecture '{args.arch}' not found. Available: {', '.join(sorted(symbols_by_arch.keys()))}"
            )
            sys.exit(1)
        # Replace all_symbols with filtered symbols
        all_symbols = symbols_by_arch[args.arch]
        # Keep only the requested arch
        symbols_by_arch = {args.arch: symbols_by_arch[args.arch]}
        arch_totals = {args.arch: arch_totals[args.arch]}

    # Separate debug/metadata from actual kernels (unified pass)
    def classify_symbols(symbols_dict):
        """Classify symbols into special sections and kernels"""
        special, kernels = {}, {}
        for name, size in symbols_dict.items():
            (special if name.startswith("<") and name.endswith(">") else kernels)[
                name
            ] = size
        return special, kernels

    special_sections, actual_kernels = classify_symbols(all_symbols)
    special_sections_by_arch = defaultdict(dict)
    actual_kernels_by_arch = defaultdict(dict)
    for arch, symbols in symbols_by_arch.items():
        special_sections_by_arch[arch], actual_kernels_by_arch[arch] = classify_symbols(
            symbols
        )

    # Filter by regex pattern if specified
    if args.filter:
        try:
            pattern = re.compile(args.filter, re.IGNORECASE)
        except re.error as e:
            logger.error(f"Invalid regular expression: {e}")
            sys.exit(1)

        # Count before filtering
        total_before = len(actual_kernels)

        # Filter actual_kernels and by_arch in one pass
        actual_kernels = {
            name: size for name, size in actual_kernels.items() if pattern.search(name)
        }
        for arch in actual_kernels_by_arch:
            actual_kernels_by_arch[arch] = {
                name: size
                for name, size in actual_kernels_by_arch[arch].items()
                if pattern.search(name)
            }

        matched = len(actual_kernels)
        logger.debug(
            f"Filter matched {matched}/{total_before} kernels with pattern '{args.filter}'"
        )

        if not actual_kernels:
            logger.warning(f"No kernels matched the filter pattern '{args.filter}'")
            sys.exit(0)

    # Count kernels by architecture
    kernel_counts_by_arch = {
        arch: len(actual_kernels_by_arch[arch]) for arch in actual_kernels_by_arch
    }
    total_kernel_count = len(actual_kernels)

    # Output based on format
    if args.format == "json":
        output_json(
            all_symbols,
            symbols_by_arch,
            arch_totals,
            special_sections,
            actual_kernels,
            kernel_counts_by_arch,
            total_kernel_count,
        )
        return

    # Print results using rich tables
    if console and use_rich:
        console.print()
        console.print(
            Panel.fit(
                "[bold cyan]📊 CUDA Kernel Size Analysis Report[/bold cyan]",
                border_style="cyan",
            )
        )

        # Architecture summary table
        if arch_totals:
            arch_table = Table(
                title="Architecture Summary",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )
            arch_table.add_column("Architecture", style="cyan", width=15)
            arch_table.add_column("Kernels", justify="right", style="blue", width=10)
            arch_table.add_column(
                "Total Size", justify="right", style="yellow", width=15
            )
            arch_table.add_column(
                "Percentage", justify="right", style="green", width=12
            )

            total_all_arch = sum(arch_totals.values())
            for arch in sorted(arch_totals.keys()):
                size = arch_totals[arch]
                percentage = (size / total_all_arch * 100) if total_all_arch > 0 else 0
                kernel_count = kernel_counts_by_arch.get(arch, 0)
                arch_table.add_row(
                    arch.upper(),
                    str(kernel_count),
                    format_size(size),
                    f"{percentage:.1f}%",
                )

            arch_table.add_section()
            arch_table.add_row(
                "[bold]TOTAL[/bold]",
                f"[bold]{total_kernel_count}[/bold]",
                f"[bold]{format_size(total_all_arch)}[/bold]",
                "[bold]100.0%[/bold]",
            )
            console.print(arch_table)
            console.print()

        # Section breakdown table (code + non-code sections)
        section_table = Table(
            title="Section Breakdown",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        section_table.add_column("Section Type", style="cyan", width=25)
        section_table.add_column(
            "Total Size", justify="right", style="yellow", width=15
        )
        section_table.add_column("% of Total", justify="right", style="green", width=12)

        total_size = sum(all_symbols.values())

        # Add code section - use real .text section size
        if total_text_section_size > 0:
            code_pct = (
                (total_text_section_size / total_size * 100) if total_size > 0 else 0
            )
            section_table.add_row(
                "[bold]Code Sections[/bold]",
                f"[bold]{format_size(total_text_section_size)}[/bold]",
                f"[bold]{code_pct:.1f}%[/bold]",
            )

        # Add special sections (debug, metadata, data)
        if special_sections:
            sorted_special = sorted(
                special_sections.items(), key=lambda x: x[1], reverse=True
            )
            for name, size in sorted_special:
                display_name = name.strip("<>")
                percentage = (size / total_size * 100) if total_size > 0 else 0
                section_table.add_row(
                    display_name, format_size(size), f"{percentage:.1f}%"
                )

        section_table.add_section()
        section_table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{format_size(total_size)}[/bold]",
            "[bold]100.0%[/bold]",
        )
        console.print(section_table)
        console.print()

        # Overall top kernels table
        title = (
            f"Top CUDA Kernels (All Architectures) - {total_kernel_count} Total"
            if not args.arch
            else f"Top CUDA Kernels ({args.arch.upper()}) - {total_kernel_count} Total"
        )
        if args.filter:
            title += f" - Filter: '{args.filter}'"
        kernel_table = Table(
            title=title, box=box.ROUNDED, show_header=True, header_style="bold magenta"
        )
        kernel_table.add_column("Rank", style="dim", width=6, justify="right")
        name_width = 120 if args.full_names else 70
        kernel_table.add_column("Kernel Name", style="cyan", width=name_width)
        kernel_table.add_column("Code Size", justify="right", style="yellow", width=12)
        kernel_table.add_column("% of Code", justify="right", style="green", width=10)

        sorted_kernels = sorted(
            actual_kernels.items(), key=lambda x: x[1], reverse=True
        )
        kernels_total_size = sum(actual_kernels.values())
        total_size = sum(all_symbols.values())

        # Show top N kernels
        display_count = min(args.top, len(sorted_kernels))
        for idx, (name, size) in enumerate(sorted_kernels[:display_count], 1):
            # Percentage relative to kernel code only
            percentage = (
                (size / kernels_total_size * 100) if kernels_total_size > 0 else 0
            )
            short_name = (
                name if args.full_names else shorten_kernel_name(name, name_width)
            )
            kernel_table.add_row(
                str(idx), short_name, format_size(size), f"{percentage:.1f}%"
            )

        if len(sorted_kernels) > display_count:
            kernel_table.add_row(
                "...",
                f"[dim]({len(sorted_kernels) - display_count} more kernels)[/dim]",
                "",
                "",
            )

        kernel_table.add_section()
        kernels_pct = (kernels_total_size / total_size * 100) if total_size > 0 else 0
        kernel_table.add_row(
            "",
            "[bold]TOTAL KERNEL CODE[/bold]",
            f"[bold]{format_size(kernels_total_size)}[/bold]",
            f"[bold]{kernels_pct:.1f}% of file[/bold]",
        )
        console.print(kernel_table)

        # Per-architecture breakdown (only if not filtering and multiple archs)
        if not args.arch and len(actual_kernels_by_arch) > 1:
            for arch in sorted(actual_kernels_by_arch.keys()):
                console.print()
                arch_kernels = actual_kernels_by_arch[arch]
                arch_sorted = sorted(
                    arch_kernels.items(), key=lambda x: x[1], reverse=True
                )
                arch_kernel_total = sum(arch_kernels.values())

                arch_kernel_count = kernel_counts_by_arch.get(arch, len(arch_kernels))
                per_arch_table = Table(
                    title=f"CUDA Kernels for {arch.upper()} - {arch_kernel_count} Total",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold magenta",
                )
                per_arch_table.add_column("Rank", style="dim", width=6, justify="right")
                per_arch_table.add_column("Kernel Name", style="cyan", width=name_width)
                per_arch_table.add_column(
                    "Code Size", justify="right", style="yellow", width=12
                )
                per_arch_table.add_column(
                    "% of Code", justify="right", style="green", width=10
                )

                # Show top 15 per architecture
                arch_display = min(15, len(arch_sorted))
                for idx, (name, size) in enumerate(arch_sorted[:arch_display], 1):
                    percentage = (
                        (size / arch_kernel_total * 100) if arch_kernel_total > 0 else 0
                    )
                    short_name = (
                        name
                        if args.full_names
                        else shorten_kernel_name(name, name_width)
                    )
                    per_arch_table.add_row(
                        str(idx), short_name, format_size(size), f"{percentage:.1f}%"
                    )

                if len(arch_sorted) > arch_display:
                    per_arch_table.add_row(
                        "...",
                        f"[dim]({len(arch_sorted) - arch_display} more kernels)[/dim]",
                        "",
                        "",
                    )

                per_arch_table.add_section()
                per_arch_table.add_row(
                    "",
                    "[bold]TOTAL KERNEL CODE[/bold]",
                    f"[bold]{format_size(arch_kernel_total)}[/bold]",
                    "[bold]100.0%[/bold]",
                )
                console.print(per_arch_table)

        console.print("\n[bold green]✓ Analysis complete![/bold green]\n")
    else:
        # Fallback to basic output
        print("\n" + "=" * 100)
        print("CUDA Kernel Size Report")
        print(f"Total Kernels: {total_kernel_count}")
        print("=" * 100)

        # Print special sections first
        if special_sections:
            print("\nNon-Code Sections (Debug Info, Metadata, etc.):")
            print("-" * 100)
            sorted_special = sorted(
                special_sections.items(), key=lambda x: x[1], reverse=True
            )
            total_size = sum(all_symbols.values())
            for name, size in sorted_special:
                display_name = name.strip("<>")
                percentage = (size / total_size * 100) if total_size > 0 else 0
                print(f"{display_name:<70} {format_size(size):>15} {percentage:>9.1f}%")
            print()

        # Print actual kernels
        sorted_kernels = sorted(
            actual_kernels.items(), key=lambda x: x[1], reverse=True
        )
        kernels_total_size = sum(actual_kernels.values())
        total_size = sum(all_symbols.values())

        name_width = 90 if args.full_names else 70
        print(
            f"\n{'CUDA Kernel Name':<{name_width}} {'Code Size':>15} {'% of Code':>12}"
        )
        print("-" * 100)

        display_count = min(args.top, len(sorted_kernels))
        for name, size in sorted_kernels[:display_count]:
            percentage = (
                (size / kernels_total_size * 100) if kernels_total_size > 0 else 0
            )
            short_name = (
                name if args.full_names else shorten_kernel_name(name, name_width)
            )
            print(
                f"{short_name:<{name_width}} {format_size(size):>15} {percentage:>11.1f}%"
            )

        if len(sorted_kernels) > display_count:
            print(f"... ({len(sorted_kernels) - display_count} more kernels)")

        print("-" * 100)
        kernels_pct = (kernels_total_size / total_size * 100) if total_size > 0 else 0
        print(
            f"{'TOTAL KERNEL CODE':<{name_width}} {format_size(kernels_total_size):>15} {kernels_pct:>10.1f}% of file"
        )
        print()


if __name__ == "__main__":
    main()
