"""
Command-line interface for KohakuPaper
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="KohakuPaper - Local Paper Copilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=48890, help="Port to bind to (default: 48890)"
    )
    serve_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    # download command
    download_parser = subparsers.add_parser(
        "download", help="Download paper data from papercopilot/paperlists"
    )
    download_parser.add_argument(
        "conference", help="Conference name (e.g., iclr, nips)"
    )
    download_parser.add_argument(
        "--year", type=int, help="Specific year (downloads all if not specified)"
    )
    download_parser.add_argument(
        "--min-year", type=int, help="Minimum year to download"
    )
    download_parser.add_argument(
        "--max-year", type=int, help="Maximum year to download"
    )
    download_parser.add_argument(
        "--force", action="store_true", help="Force re-download existing files"
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List available data")
    list_parser.add_argument(
        "--remote", action="store_true", help="List available files on GitHub"
    )
    list_parser.add_argument(
        "--conference", help="Conference to list files for (with --remote)"
    )

    # query command
    query_parser = subparsers.add_parser("query", help="Query local paper data")
    query_parser.add_argument("--conference", help="Filter by conference")
    query_parser.add_argument("--year", type=int, help="Filter by year")
    query_parser.add_argument("--search", help="Search in title/abstract/keywords")
    query_parser.add_argument("--limit", type=int, default=10, help="Max results")

    args = parser.parse_args()

    match args.command:
        case "serve":
            run_server(args.host, args.port, args.reload)
        case "download":
            run_download(args)
        case "list":
            run_list(args)
        case "query":
            run_query(args)
        case _:
            parser.print_help()
            sys.exit(1)


def run_server(host: str, port: int, reload: bool):
    """Start the API server"""
    import uvicorn

    # Configure reload to exclude data directories (use relative glob patterns)
    reload_excludes = None
    if reload:
        # Exclude paperlists, data, and cache directories from file watching
        # Use glob patterns relative to the working directory
        reload_excludes = [
            "paperlists/*",
            "data/*",
            ".cache/*",
            ".venv/*",
        ]

    uvicorn.run(
        "kohakupaper.api:app",
        host=host,
        port=port,
        reload=reload,
        reload_excludes=reload_excludes,
    )


def run_download(args):
    """Download paper data"""
    from .downloader import download_paper_list, download_all_for_conference

    if args.year:
        path = download_paper_list(args.conference, args.year, force=args.force)
        if path:
            print(f"Downloaded: {path.name}")
    else:
        paths = download_all_for_conference(
            args.conference,
            min_year=args.min_year,
            max_year=args.max_year,
            force=args.force,
        )
        for path in paths:
            print(f"Downloaded: {path.name}")


def run_list(args):
    """List available data"""
    from .downloader import list_conference_files
    from .database import get_available_data
    from .config import CONFERENCES

    if args.remote:
        if args.conference:
            files = list_conference_files(args.conference)
            print(f"Available files for {args.conference}:")
            for f in files:
                print(f"  {f}")
        else:
            print("Supported conferences:")
            for key, name in CONFERENCES.items():
                print(f"  {key}: {name}")
    else:
        available = get_available_data()
        if available:
            print("Local data (JSON files directly queryable):")
            for conf, years in available.items():
                print(f"  {conf}: {', '.join(map(str, years))}")
        else:
            print(
                "No local data found. Use 'kohakupaper download <conference>' to download."
            )


def run_query(args):
    """Query local paper data"""
    from .database import query_papers, count_papers

    total = count_papers(
        conference=args.conference,
        year=args.year,
        search=args.search,
    )

    papers = query_papers(
        conference=args.conference,
        year=args.year,
        search=args.search,
        limit=args.limit,
    )

    print(f"Found {total} papers (showing {len(papers)}):\n")
    for paper in papers:
        rating = paper.get("rating_avg")
        rating_str = f"{rating:.2f}" if rating else "N/A"
        print(f"[{rating_str}] {paper.get('title', 'Untitled')}")
        print(f"       Status: {paper.get('status', 'N/A')}")
        print(f"       Area: {paper.get('primary_area', 'N/A')}")
        print()


if __name__ == "__main__":
    main()
