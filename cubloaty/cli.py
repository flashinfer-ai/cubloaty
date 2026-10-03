"""
cubloaty - Analyze CUDA binary sizes in .so files
Similar to bloaty but for CUDA kernels
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict

from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.table import Table

from . import __version__
from .analysis import (
    CATEGORIES,
    analyze_file,
    canonical_name,
    find_duplicates,
    function_tu,
    summarize_functions,
)

logger = logging.getLogger("cubloaty")

KIND_LABELS = {"sass": "SASS", "ptx": "PTX", "ltoir": "LTO-IR"}


def setup_logging(verbose=False):
    """Setup logging with a Rich handler (falls back to plain text when
    output is redirected); quiet unless --verbose"""
    logger.handlers.clear()
    handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=True,
        show_time=False,
        show_path=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)


def format_size(size_bytes):
    """Format size in human-readable format"""
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if size_bytes >= scale:
            return f"{size_bytes / scale:.1f}{unit}"
    return f"{size_bytes}B"


def percent(part, whole):
    return f"{part / whole * 100:.1f}%" if whole else "-"


def arch_sort_key(arch):
    """sm_* before compute_* before lto_*, then by number"""
    prefix, _, rest = arch.partition("_")
    digits = re.match(r"\d+", rest)
    order = {"sm": 0, "compute": 1, "lto": 2}.get(prefix, 3)
    return (order, int(digits.group()) if digits else 0, rest)


_QUALIFIER_RE = re.compile(r"\b(?:[A-Za-z_]\w*::)+(?=[A-Za-z_~])")
_ANON_NS = "(anonymous namespace)"


def _split_params(name):
    """Split "ns::f<T>(P)" into ("ns::f<T>", "(P)") at the top-level parameter list"""
    depth = 0
    i = 0
    while i < len(name):
        c = name[i]
        if name.startswith(_ANON_NS, i):
            i += len(_ANON_NS)
            continue
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif c == "(" and depth == 0 and i > 0:
            return name[:i], name[i:]
        i += 1
    return name, ""


def shorten_name(name, width):
    """Fit a demangled name into `width` columns, dropping the least useful
    parts first: namespaces inside template arguments and parameters, then
    the parameter list of templates (it repeats the template arguments),
    then the function's own namespaces. Rich truncates whatever remains."""
    if name.startswith("void "):
        name = name[5:]
    head, params = _split_params(name)
    split = head.find("<")
    own, targs = (head, "") if split < 0 else (head[:split], head[split:])
    targs, params = _QUALIFIER_RE.sub("", targs), _QUALIFIER_RE.sub("", params)
    tail = targs if targs else params
    for candidate in (
        name,
        own + targs + params,
        own + tail,
        own.rpartition("::")[2] + tail,
    ):
        if len(candidate) <= width:
            return candidate
    return candidate


def display_name(name, mangled_names, args, width):
    if args.mangled:
        return ", ".join(sorted(mangled_names))
    if args.full_names:
        return name
    return shorten_name(name, width)


# --------------------------------------------------------------------------
# Data assembly (shared by table and JSON output)


def build_views(report, args):
    images = report.images
    if args.arch:
        images = [i for i in images if i.arch == args.arch]

    functions = summarize_functions(report, images)
    duplicates = find_duplicates(report, images)
    if args.filter:
        functions = {
            k: v
            for k, v in functions.items()
            if args.filter.search(k) or any(map(args.filter.search, v.mangled))
        }
        duplicates = [
            d
            for d in duplicates
            if args.filter.search(d.name)
            or any(args.filter.search(fn.name) for _, fn in d.copies)
        ]

    key = (lambda f: f.file_size) if args.sort == "file" else (lambda f: f.size)
    ordered = sorted(functions.values(), key=lambda f: (key(f), f.size), reverse=True)

    categories = defaultdict(lambda: [0, 0])  # category -> [size, file_size]
    for img in images:
        for cat, n in img.shared.items():
            categories[cat][0] += n
        for cat, n in img.shared_file.items():
            categories[cat][1] += n
        for fn in img.functions.values():
            for cat, n in fn.size.items():
                categories[cat][0] += n
            for cat, n in fn.file_size.items():
                categories[cat][1] += n
    if not args.arch and report.container_overhead:
        categories["fatbin"][1] += report.container_overhead

    components = defaultdict(
        lambda: {"images": 0, "size": 0, "file_size": 0, "opaque": True}
    )
    for img in report.images:
        c = components[(img.section, img.kind, img.arch)]
        c["images"] += 1
        c["size"] += img.size
        c["file_size"] += img.file_size
        c["opaque"] &= img.opaque
    kernel_counts = defaultdict(set)
    for img in report.images:
        for fn in img.functions.values():
            if fn.is_kernel:
                name = canonical_name(report.names.get(fn.name, fn.name))
                kernel_counts[(img.section, img.kind, img.arch)].add(name)
    for k, c in components.items():
        c["kernels"] = len(kernel_counts[k])

    return {
        "images": images,
        "functions": ordered,
        "duplicates": duplicates,
        "categories": categories,
        "components": components,
    }


def component_label(section, kind, arch):
    label = f"{arch} {KIND_LABELS.get(kind, kind)}"
    if section and section != ".nv_fatbin":
        label += f" ({section})"
    return label


# --------------------------------------------------------------------------
# JSON


def output_json(report, views, args):
    device_total = sum(views["categories"][c][1] for c in views["categories"])
    archs = defaultdict(lambda: {"size": 0, "file_size": 0, "images": 0})
    for img in views["images"]:
        a = archs[img.arch]
        a["kind"] = img.kind
        a["size"] += img.size
        a["file_size"] += img.file_size
        a["images"] += 1
    kernels_by_arch = defaultdict(int)
    for f in views["functions"]:
        if f.is_kernel:
            for arch in f.by_arch:
                kernels_by_arch[arch] += 1

    result = {
        "file": report.path,
        "format": report.file_format,
        "file_size": report.file_size,
        "host_file_size": report.host_file_size,
        "device_file_size": report.device_file_size,
        "total_kernels": sum(1 for f in views["functions"] if f.is_kernel),
        "components": [
            {
                "section": section,
                "kind": kind,
                "arch": arch,
                **c,
            }
            for (section, kind, arch), c in sorted(
                views["components"].items(),
                key=lambda kv: (kv[0][0], arch_sort_key(kv[0][2])),
            )
        ],
        "architectures": {
            arch: {**a, "kernel_count": kernels_by_arch[arch]}
            for arch, a in sorted(archs.items(), key=lambda kv: arch_sort_key(kv[0]))
        },
        "categories": [
            {
                "category": cat,
                "description": CATEGORIES.get(cat, cat),
                "size": size,
                "file_size": fsize,
                "percentage_of_device": round(fsize / device_total * 100, 2)
                if device_total
                else 0,
            }
            for cat, (size, fsize) in sorted(
                views["categories"].items(), key=lambda kv: kv[1][1], reverse=True
            )
        ],
        "kernels": [
            {
                "name": f.name,
                "mangled": sorted(f.mangled),
                "kind": "kernel" if f.is_kernel else "device_function",
                "size": f.size,
                "file_size": f.file_size,
                "by_arch": {
                    arch: {"size": s, "file_size": fs, "copies": n}
                    for arch, (s, fs, n) in sorted(
                        f.by_arch.items(), key=lambda kv: arch_sort_key(kv[0])
                    )
                },
                "by_category": dict(f.by_category.most_common()),
            }
            for f in views["functions"]
        ],
        "duplicates": [
            {
                "name": d.name,
                "arch": d.arch,
                "copies": len(d.copies),
                "size": max(d.sizes),
                "wasted_size": d.wasted_size,
                "wasted_file_size": d.wasted_file_size,
                "locations": [
                    {
                        "mangled": fn.name,
                        "tu": function_tu(img, fn),
                        "location": img.location,
                        "size": sum(fn.size.values()),
                        "file_size": sum(fn.file_size.values()),
                    }
                    for img, fn in d.copies
                ],
            }
            for d in views["duplicates"]
        ],
    }
    print(json.dumps(result, indent=2))


# --------------------------------------------------------------------------
# Tables


def new_table(title, args, caption=None):
    return Table(
        title=title,
        caption=caption,
        box=box.ASCII if args.no_color else box.ROUNDED,
        header_style="bold magenta",
        title_style="bold",
    )


def name_column(table, args):
    if args.full_names:
        table.add_column("Kernel", style="cyan", overflow="fold")
    else:
        table.add_column("Kernel", style="cyan", no_wrap=True, overflow="ellipsis")


def fit_name_column(table, console):
    """Give the kernel-name column whatever width the fixed-width columns
    leave; without this Rich squeezes the numeric columns to nothing"""
    name = next(c for c in table.columns if c.header == "Kernel")
    others = sum(c.min_width or 0 for c in table.columns if c is not name)
    name.max_width = max(10, console.width - others - 3 * len(table.columns) - 1)
    return name.max_width


def num_column(table, header, style, width=None):
    table.add_column(
        header,
        justify="right",
        style=style,
        no_wrap=True,
        min_width=max(len(header), width or 0),
    )


def size_columns(table):
    num_column(table, "Uncompressed", "yellow")
    num_column(table, "File Size", "yellow")


def render_composition(console, report, views, args):
    table = new_table("File Composition", args)
    table.add_column("Component", style="cyan")
    num_column(table, "Kernels", "blue")
    num_column(table, "Images", "blue")
    size_columns(table)
    num_column(table, "% of File", "green")

    total = report.file_size
    for (section, kind, arch), c in sorted(
        views["components"].items(), key=lambda kv: (kv[0][0], arch_sort_key(kv[0][2]))
    ):
        table.add_row(
            component_label(section, kind, arch),
            "" if c["opaque"] else str(c["kernels"]),
            str(c["images"]),
            format_size(c["size"]),
            format_size(c["file_size"]),
            percent(c["file_size"], total),
        )
    if report.container_overhead:
        table.add_row(
            "Fatbin container overhead",
            "",
            "",
            "",
            format_size(report.container_overhead),
            percent(report.container_overhead, total),
        )
    if report.host_file_size:
        table.add_row(
            "Host code & data",
            "",
            "",
            "",
            format_size(report.host_file_size),
            percent(report.host_file_size, total),
        )
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        str(sum(c["images"] for c in views["components"].values())),
        "",
        f"[bold]{format_size(total)}[/bold]",
        "[bold]100.0%[/bold]",
    )
    console.print(table)
    console.print()


def render_categories(console, views, args):
    cats = views["categories"]
    total_file = sum(fs for _, fs in cats.values())
    total_size = sum(s for s, _ in cats.values())
    table = new_table("Device Code Breakdown", args)
    table.add_column("Category", style="cyan")
    size_columns(table)
    num_column(table, "% of Device", "green")
    for cat, (size, fsize) in sorted(cats.items(), key=lambda kv: kv[1], reverse=True):
        if not size and not fsize:
            continue
        table.add_row(
            CATEGORIES.get(cat, cat),
            format_size(size) if size else "",
            format_size(fsize),
            percent(fsize, total_file),
        )
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{format_size(total_size)}[/bold]",
        f"[bold]{format_size(total_file)}[/bold]",
        "[bold]100.0%[/bold]",
    )
    console.print(table)
    console.print()


def render_functions(console, title, rows, metric_total, args, limit, size_of):
    table = new_table(title, args)
    num_column(table, "Rank", "dim", len(str(min(limit, len(rows)))))
    name_column(table, args)
    size_columns(table)
    num_column(table, "% of Device", "green")
    width = fit_name_column(table, console)

    for idx, f in enumerate(rows[:limit], 1):
        size, fsize = size_of(f)
        name = escape(display_name(f.name, f.mangled, args, width))
        if not f.is_kernel:
            name += " [dim]\\[device fn][/dim]"
        metric = fsize if args.sort == "file" else size
        table.add_row(
            str(idx),
            name,
            format_size(size),
            format_size(fsize),
            percent(metric, metric_total),
        )
    if len(rows) > limit:
        table.add_row("...", f"[dim]({len(rows) - limit} more)[/dim]", "", "", "")

    sizes = [size_of(f) for f in rows]
    tsize, tfile = sum(s for s, _ in sizes), sum(fs for _, fs in sizes)
    table.add_section()
    table.add_row(
        "",
        f"[bold]TOTAL ({len(rows)} functions)[/bold]",
        f"[bold]{format_size(tsize)}[/bold]",
        f"[bold]{format_size(tfile)}[/bold]",
        f"[bold]{percent(tfile if args.sort == 'file' else tsize, metric_total)}[/bold]",
    )
    console.print(table)
    console.print()


def render_duplicates(console, views, args):
    dups = views["duplicates"]
    if not dups:
        return
    wasted_file = sum(d.wasted_file_size for d in dups)
    wasted_size = sum(d.wasted_size for d in dups)
    table = new_table(
        f"Duplicate Kernels - {len(dups)} compiled into multiple TUs",
        args,
        caption=(
            "Every TU that instantiates a header-defined kernel embeds its own "
            "copy; define it in a single TU to keep one."
        ),
    )
    limit = args.top
    shown = dups[:limit]
    tu_texts = []
    for d in shown:
        tus = Counter(function_tu(img, fn) or img.location for img, fn in d.copies)
        tu_texts.append(
            ", ".join(f"{tu} x{n}" if n > 1 else tu for tu, n in sorted(tus.items()))
        )

    num_column(table, "Rank", "dim", len(str(len(shown))))
    name_column(table, args)
    table.add_column(
        "Arch",
        style="blue",
        no_wrap=True,
        min_width=max((len(d.arch) for d in shown), default=4),
    )
    num_column(table, "Copies", "blue")
    num_column(table, "Wasted Uncomp.", "yellow")
    num_column(table, "Wasted File", "red")
    # Leave the kernel name at least two thirds of the flexible width
    fixed = sum(c.min_width or 0 for c in table.columns)
    flexible = console.width - fixed - 3 * (len(table.columns) + 1) - 1
    tu_width = min(40, max([3, *map(len, tu_texts)]), max(10, flexible // 3))
    table.add_column(
        "TUs",
        style="dim",
        no_wrap=True,
        overflow="ellipsis",
        min_width=tu_width,
        max_width=tu_width,
    )
    width = fit_name_column(table, console)

    for idx, (d, tu_text) in enumerate(zip(shown, tu_texts), 1):
        mangled = {fn.name for _, fn in d.copies}
        name = escape(display_name(d.name, mangled, args, width))
        if not any(fn.is_kernel for _, fn in d.copies):
            name += " [dim]\\[device fn][/dim]"
        table.add_row(
            str(idx),
            name,
            d.arch,
            str(len(d.copies)),
            format_size(d.wasted_size),
            format_size(d.wasted_file_size),
            escape(tu_text),
        )
    if len(dups) > limit:
        table.add_row(
            "...", f"[dim]({len(dups) - limit} more)[/dim]", "", "", "", "", ""
        )
    table.add_section()
    table.add_row(
        "",
        "[bold]TOTAL WASTED[/bold]",
        "",
        "",
        f"[bold]{format_size(wasted_size)}[/bold]",
        f"[bold]{format_size(wasted_file)}[/bold]",
        "",
    )
    console.print(table)
    console.print()


def output_tables(report, views, args):
    if args.no_color:
        console = Console(no_color=True, highlight=False, emoji=False)
    else:
        console = Console(highlight=False)
    if not console.is_terminal and "COLUMNS" not in os.environ:
        console.width = 160

    console.print()
    console.print(
        f"[bold cyan]{escape(os.path.basename(report.path))}[/bold cyan]: "
        f"{report.file_format}, {format_size(report.file_size)} "
        f"({report.file_size:,} bytes)"
    )
    console.print()

    if report.file_format != "cubin":
        render_composition(console, report, views, args)
    render_categories(console, views, args)

    cats = views["categories"]
    device_total = sum((fs if args.sort == "file" else s) for s, fs in cats.values())
    functions = views["functions"]
    title = f"Top Kernels ({args.arch or 'all architectures'})"
    if args.filter:
        title += f" - filter: '{args.filter.pattern}'"
    render_functions(
        console,
        title,
        functions,
        device_total,
        args,
        args.top,
        lambda f: (f.size, f.file_size),
    )

    archs = sorted({i.arch for i in views["images"]}, key=arch_sort_key)
    if not args.arch and len(archs) > 1:
        for arch in archs:
            rows = [f for f in functions if arch in f.by_arch]
            if args.sort == "file":
                rows.sort(key=lambda f: f.by_arch[arch][1], reverse=True)
            else:
                rows.sort(key=lambda f: f.by_arch[arch][0], reverse=True)
            if rows:
                render_functions(
                    console,
                    f"Top Kernels ({arch})",
                    rows,
                    device_total,
                    args,
                    min(args.top, 15),
                    lambda f, a=arch: (f.by_arch[a][0], f.by_arch[a][1]),
                )

    render_duplicates(console, views, args)


# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CUDA binary sizes - bloaty for CUDA kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cubloaty library.so                    # Analyze CUDA kernels in .so
  cubloaty kernel.cubin                  # Analyze single .cubin file
  cubloaty libfoo.a                      # Analyze a static library
  cubloaty library.so --top 50           # Show top 50 kernels
  cubloaty library.so --arch sm_90       # Filter by architecture
  cubloaty library.so --filter "gemm"    # Filter kernels by name (regex)
  cubloaty library.so --format json      # Output as JSON
  cubloaty library.so --full-names       # Show full kernel names
        """,
    )
    parser.add_argument(
        "file", help="Path to a .so/.o/.a/executable, .cubin or .fatbin file"
    )
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
        help="Filter by architecture (e.g., sm_90, sm_100a, compute_90)",
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
        "--sort",
        "-s",
        choices=["file", "size"],
        default="file",
        help="Rank by bytes in the file (default) or by uncompressed size",
    )
    parser.add_argument(
        "--full-names",
        action="store_true",
        help="Show full kernel names without truncation",
    )
    parser.add_argument(
        "--mangled", action="store_true", help="Show mangled instead of demangled names"
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
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if not os.path.isfile(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)

    if args.filter:
        try:
            args.filter = re.compile(args.filter, re.IGNORECASE)
        except re.error as e:
            logger.error(f"Invalid regular expression: {e}")
            sys.exit(1)

    try:
        report = analyze_file(args.file)
    except (OSError, ValueError) as e:
        logger.error(f"Could not analyze {args.file}: {e}")
        sys.exit(1)

    if not report.images:
        logger.error("No CUDA device code found in the file.")
        sys.exit(1)

    if args.arch:
        available = sorted({i.arch for i in report.images}, key=arch_sort_key)
        if args.arch not in available:
            logger.error(
                f"Architecture '{args.arch}' not found. "
                f"Available: {', '.join(available)}"
            )
            sys.exit(1)

    views = build_views(report, args)
    if args.filter and not views["functions"]:
        logger.warning(f"No kernels matched the filter pattern '{args.filter.pattern}'")

    if args.format == "json":
        output_json(report, views, args)
    else:
        output_tables(report, views, args)


if __name__ == "__main__":
    main()
