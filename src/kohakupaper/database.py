"""
DuckDB database layer - directly query JSON files without importing
"""

import math
import re
from pathlib import Path

import duckdb

from .config import DATA_DIR


def clean_nan(value):
    """Convert NaN/Inf to None for JSON serialization"""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def clean_dict(d: dict) -> dict:
    """Clean all NaN values in a dict"""
    return {k: clean_nan(v) for k, v in d.items()}


def clean_records(records: list[dict]) -> list[dict]:
    """Clean all NaN values in a list of dicts"""
    return [clean_dict(r) for r in records]


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get an in-memory DuckDB connection"""
    conn = duckdb.connect(":memory:")
    # Install and load JSON extension
    conn.execute("INSTALL json; LOAD json;")
    return conn


def list_json_files() -> list[Path]:
    """List all JSON files in data directory"""
    return sorted(DATA_DIR.glob("*.json"))


def parse_filename(filepath: Path) -> tuple[str, int] | None:
    """Parse conference and year from filename like 'iclr2024.json'"""
    match = re.match(r"([a-z]+)(\d{4})", filepath.stem.lower())
    if match:
        return match.group(1), int(match.group(2))
    return None


def get_available_data() -> dict[str, list[int]]:
    """Get dict of conferences to available years from local files"""
    result: dict[str, list[int]] = {}
    for path in list_json_files():
        parsed = parse_filename(path)
        if parsed:
            conf, year = parsed
            if conf not in result:
                result[conf] = []
            result[conf].append(year)
    for conf in result:
        result[conf].sort()
    return result


def build_json_source(
    conference: str | None = None,
    year: int | None = None,
    files: list[str] | None = None,
) -> str:
    """Build the JSON file source for DuckDB query"""
    if files:
        # Specific files provided
        paths = [DATA_DIR / f for f in files if (DATA_DIR / f).exists()]
    else:
        # Filter by conference/year
        paths = []
        for path in list_json_files():
            parsed = parse_filename(path)
            if not parsed:
                continue
            conf, yr = parsed
            if conference and conf != conference.lower():
                continue
            if year and yr != year:
                continue
            paths.append(path)

    if not paths:
        return ""

    # DuckDB can read multiple JSON files with glob or list
    if len(paths) == 1:
        return f"'{paths[0].as_posix()}'"
    else:
        # Use list of files
        file_list = ", ".join(f"'{p.as_posix()}'" for p in paths)
        return f"[{file_list}]"


def query_papers(
    conference: str | None = None,
    year: int | None = None,
    status: str | None = None,
    primary_area: str | None = None,
    search: str | None = None,
    min_rating: float | None = None,
    max_rating: float | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    has_rating_diff: bool | None = None,
    has_confidence_diff: bool | None = None,
    title_filter: str | None = None,
    # New filters for init and diff
    min_rating_init: float | None = None,
    max_rating_init: float | None = None,
    min_rating_diff: float | None = None,
    max_rating_diff: float | None = None,
    min_confidence_init: float | None = None,
    max_confidence_init: float | None = None,
    min_confidence_diff: float | None = None,
    max_confidence_diff: float | None = None,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "rating_avg",
    order_dir: str = "DESC",
    files: list[str] | None = None,
) -> list[dict]:
    """Query papers directly from JSON files"""
    json_source = build_json_source(conference, year, files)
    if not json_source:
        return []

    conn = get_connection()

    conditions = []

    if status:
        conditions.append(f"status = '{status}'")

    if primary_area:
        conditions.append(f"primary_area ILIKE '%{primary_area}%'")

    if search:
        search_escaped = search.replace("'", "''")
        conditions.append(
            f"(title ILIKE '%{search_escaped}%' OR abstract ILIKE '%{search_escaped}%' OR keywords ILIKE '%{search_escaped}%')"
        )

    if min_rating is not None:
        conditions.append(f"rating_avg[1] >= {min_rating}")

    if max_rating is not None:
        conditions.append(f"rating_avg[1] <= {max_rating}")

    if min_confidence is not None:
        conditions.append(f"confidence_avg[1] >= {min_confidence}")

    if max_confidence is not None:
        conditions.append(f"confidence_avg[1] <= {max_confidence}")

    if has_rating_diff is True:
        # Check if _diff.has_diff is true
        conditions.append("(_diff IS NOT NULL AND _diff.has_diff = true)")

    if has_confidence_diff is True:
        # Check if _diff.confidence_diff has non-zero values
        conditions.append(
            "(_diff IS NOT NULL AND _diff.confidence_diff IS NOT NULL AND list_reduce(_diff.confidence_diff, (a, b) -> a + abs(b), 0) > 0)"
        )

    # Filter by initial rating average (computed from _diff.rating_first)
    if min_rating_init is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.rating_first IS NOT NULL AND list_reduce(_diff.rating_first, (a, b) -> a + b, 0) / len(_diff.rating_first) >= {min_rating_init})"
        )

    if max_rating_init is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.rating_first IS NOT NULL AND list_reduce(_diff.rating_first, (a, b) -> a + b, 0) / len(_diff.rating_first) <= {max_rating_init})"
        )

    # Filter by rating diff (current avg - init avg)
    if min_rating_diff is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.rating_first IS NOT NULL AND _diff.rating_current IS NOT NULL AND (list_reduce(_diff.rating_current, (a, b) -> a + b, 0) / len(_diff.rating_current) - list_reduce(_diff.rating_first, (a, b) -> a + b, 0) / len(_diff.rating_first)) >= {min_rating_diff})"
        )

    if max_rating_diff is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.rating_first IS NOT NULL AND _diff.rating_current IS NOT NULL AND (list_reduce(_diff.rating_current, (a, b) -> a + b, 0) / len(_diff.rating_current) - list_reduce(_diff.rating_first, (a, b) -> a + b, 0) / len(_diff.rating_first)) <= {max_rating_diff})"
        )

    # Filter by initial confidence average
    if min_confidence_init is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.confidence_first IS NOT NULL AND list_reduce(_diff.confidence_first, (a, b) -> a + b, 0) / len(_diff.confidence_first) >= {min_confidence_init})"
        )

    if max_confidence_init is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.confidence_first IS NOT NULL AND list_reduce(_diff.confidence_first, (a, b) -> a + b, 0) / len(_diff.confidence_first) <= {max_confidence_init})"
        )

    # Filter by confidence diff (current avg - init avg)
    if min_confidence_diff is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.confidence_first IS NOT NULL AND _diff.confidence_current IS NOT NULL AND (list_reduce(_diff.confidence_current, (a, b) -> a + b, 0) / len(_diff.confidence_current) - list_reduce(_diff.confidence_first, (a, b) -> a + b, 0) / len(_diff.confidence_first)) >= {min_confidence_diff})"
        )

    if max_confidence_diff is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.confidence_first IS NOT NULL AND _diff.confidence_current IS NOT NULL AND (list_reduce(_diff.confidence_current, (a, b) -> a + b, 0) / len(_diff.confidence_current) - list_reduce(_diff.confidence_first, (a, b) -> a + b, 0) / len(_diff.confidence_first)) <= {max_confidence_diff})"
        )

    if title_filter:
        title_escaped = title_filter.replace("'", "''")
        conditions.append(f"title ILIKE '%{title_escaped}%'")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Validate order_by
    valid_order_cols = {
        "rating_avg": "rating_avg[1]",
        "confidence_avg": "confidence_avg[1]",
        "soundness_avg": "soundness_avg[1]",
        "contribution_avg": "contribution_avg[1]",
        "presentation_avg": "presentation_avg[1]",
        "corr_rating_confidence": "corr_rating_confidence",
        "title": "title",
        "status": "status",
        "gs_citation": "gs_citation",
        # Init and diff ordering
        "rating_init": "CASE WHEN _diff IS NOT NULL AND _diff.rating_first IS NOT NULL AND len(_diff.rating_first) > 0 THEN list_reduce(_diff.rating_first, (a, b) -> a + b, 0.0) / len(_diff.rating_first) ELSE NULL END",
        "rating_diff": "CASE WHEN _diff IS NOT NULL AND _diff.rating_first IS NOT NULL AND _diff.rating_current IS NOT NULL AND len(_diff.rating_first) > 0 AND len(_diff.rating_current) > 0 THEN (list_reduce(_diff.rating_current, (a, b) -> a + b, 0.0) / len(_diff.rating_current)) - (list_reduce(_diff.rating_first, (a, b) -> a + b, 0.0) / len(_diff.rating_first)) ELSE NULL END",
        "confidence_init": "CASE WHEN _diff IS NOT NULL AND _diff.confidence_first IS NOT NULL AND len(_diff.confidence_first) > 0 THEN list_reduce(_diff.confidence_first, (a, b) -> a + b, 0.0) / len(_diff.confidence_first) ELSE NULL END",
        "confidence_diff": "CASE WHEN _diff IS NOT NULL AND _diff.confidence_first IS NOT NULL AND _diff.confidence_current IS NOT NULL AND len(_diff.confidence_first) > 0 AND len(_diff.confidence_current) > 0 THEN (list_reduce(_diff.confidence_current, (a, b) -> a + b, 0.0) / len(_diff.confidence_current)) - (list_reduce(_diff.confidence_first, (a, b) -> a + b, 0.0) / len(_diff.confidence_first)) ELSE NULL END",
    }
    order_col = valid_order_cols.get(order_by, "rating_avg[1]")
    order_dir = "DESC" if order_dir.upper() == "DESC" else "ASC"

    query = f"""
        SELECT
            id,
            title,
            track,
            status,
            abstract,
            tldr,
            keywords,
            primary_area,
            author,
            site,
            rating,
            confidence,
            rating_avg[1] as rating_avg,
            rating_avg[2] as rating_std,
            confidence_avg[1] as confidence_avg,
            confidence_avg[2] as confidence_std,
            soundness_avg[1] as soundness_avg,
            contribution_avg[1] as contribution_avg,
            presentation_avg[1] as presentation_avg,
            corr_rating_confidence,
            _diff,
            gs_citation,
            bibtex,
            github,
            project
        FROM read_json({json_source}, auto_detect=true, format='array')
        WHERE {where_clause}
        ORDER BY {order_col} {order_dir} NULLS LAST
        LIMIT {limit} OFFSET {offset}
    """

    try:
        result = conn.execute(query).fetchdf()
        conn.close()
        return result.to_dict(orient="records")
    except Exception as e:
        print(f"Query error: {e}")
        conn.close()
        return []


def count_papers(
    conference: str | None = None,
    year: int | None = None,
    status: str | None = None,
    primary_area: str | None = None,
    search: str | None = None,
    min_rating: float | None = None,
    max_rating: float | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    has_rating_diff: bool | None = None,
    has_confidence_diff: bool | None = None,
    title_filter: str | None = None,
    # New filters for init and diff
    min_rating_init: float | None = None,
    max_rating_init: float | None = None,
    min_rating_diff: float | None = None,
    max_rating_diff: float | None = None,
    min_confidence_init: float | None = None,
    max_confidence_init: float | None = None,
    min_confidence_diff: float | None = None,
    max_confidence_diff: float | None = None,
    files: list[str] | None = None,
) -> int:
    """Count papers matching filters"""
    json_source = build_json_source(conference, year, files)
    if not json_source:
        return 0

    conn = get_connection()

    conditions = []

    if status:
        conditions.append(f"status = '{status}'")

    if primary_area:
        conditions.append(f"primary_area ILIKE '%{primary_area}%'")

    if search:
        search_escaped = search.replace("'", "''")
        conditions.append(
            f"(title ILIKE '%{search_escaped}%' OR abstract ILIKE '%{search_escaped}%' OR keywords ILIKE '%{search_escaped}%')"
        )

    if min_rating is not None:
        conditions.append(f"rating_avg[1] >= {min_rating}")

    if max_rating is not None:
        conditions.append(f"rating_avg[1] <= {max_rating}")

    if min_confidence is not None:
        conditions.append(f"confidence_avg[1] >= {min_confidence}")

    if max_confidence is not None:
        conditions.append(f"confidence_avg[1] <= {max_confidence}")

    if has_rating_diff is True:
        conditions.append("(_diff IS NOT NULL AND _diff.has_diff = true)")

    if has_confidence_diff is True:
        conditions.append(
            "(_diff IS NOT NULL AND _diff.confidence_diff IS NOT NULL AND list_reduce(_diff.confidence_diff, (a, b) -> a + abs(b), 0) > 0)"
        )

    # Filter by initial rating average
    if min_rating_init is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.rating_first IS NOT NULL AND list_reduce(_diff.rating_first, (a, b) -> a + b, 0) / len(_diff.rating_first) >= {min_rating_init})"
        )

    if max_rating_init is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.rating_first IS NOT NULL AND list_reduce(_diff.rating_first, (a, b) -> a + b, 0) / len(_diff.rating_first) <= {max_rating_init})"
        )

    # Filter by rating diff
    if min_rating_diff is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.rating_first IS NOT NULL AND _diff.rating_current IS NOT NULL AND (list_reduce(_diff.rating_current, (a, b) -> a + b, 0) / len(_diff.rating_current) - list_reduce(_diff.rating_first, (a, b) -> a + b, 0) / len(_diff.rating_first)) >= {min_rating_diff})"
        )

    if max_rating_diff is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.rating_first IS NOT NULL AND _diff.rating_current IS NOT NULL AND (list_reduce(_diff.rating_current, (a, b) -> a + b, 0) / len(_diff.rating_current) - list_reduce(_diff.rating_first, (a, b) -> a + b, 0) / len(_diff.rating_first)) <= {max_rating_diff})"
        )

    # Filter by initial confidence average
    if min_confidence_init is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.confidence_first IS NOT NULL AND list_reduce(_diff.confidence_first, (a, b) -> a + b, 0) / len(_diff.confidence_first) >= {min_confidence_init})"
        )

    if max_confidence_init is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.confidence_first IS NOT NULL AND list_reduce(_diff.confidence_first, (a, b) -> a + b, 0) / len(_diff.confidence_first) <= {max_confidence_init})"
        )

    # Filter by confidence diff
    if min_confidence_diff is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.confidence_first IS NOT NULL AND _diff.confidence_current IS NOT NULL AND (list_reduce(_diff.confidence_current, (a, b) -> a + b, 0) / len(_diff.confidence_current) - list_reduce(_diff.confidence_first, (a, b) -> a + b, 0) / len(_diff.confidence_first)) >= {min_confidence_diff})"
        )

    if max_confidence_diff is not None:
        conditions.append(
            f"(_diff IS NOT NULL AND _diff.confidence_first IS NOT NULL AND _diff.confidence_current IS NOT NULL AND (list_reduce(_diff.confidence_current, (a, b) -> a + b, 0) / len(_diff.confidence_current) - list_reduce(_diff.confidence_first, (a, b) -> a + b, 0) / len(_diff.confidence_first)) <= {max_confidence_diff})"
        )

    if title_filter:
        title_escaped = title_filter.replace("'", "''")
        conditions.append(f"title ILIKE '%{title_escaped}%'")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
        SELECT COUNT(*) as cnt
        FROM read_json({json_source}, auto_detect=true, format='array')
        WHERE {where_clause}
    """

    try:
        result = conn.execute(query).fetchone()[0]
        conn.close()
        return result
    except Exception as e:
        print(f"Count error: {e}")
        conn.close()
        return 0


def get_statistics(
    conference: str | None = None,
    year: int | None = None,
    group_by: str = "status",
    files: list[str] | None = None,
) -> list[dict]:
    """Get aggregated statistics from JSON files"""
    json_source = build_json_source(conference, year, files)
    if not json_source:
        return []

    conn = get_connection()

    # Validate group_by
    valid_groups = ["status", "primary_area", "track"]
    if group_by not in valid_groups:
        group_by = "status"

    query = f"""
        SELECT
            {group_by},
            COUNT(*) as paper_count,
            AVG(rating_avg[1]) as avg_rating,
            STDDEV(rating_avg[1]) as std_rating,
            MIN(rating_avg[1]) as min_rating,
            MAX(rating_avg[1]) as max_rating,
            AVG(confidence_avg[1]) as avg_confidence,
            AVG(gs_citation) FILTER (WHERE gs_citation >= 0) as avg_citations
        FROM read_json({json_source}, auto_detect=true, format='array')
        GROUP BY {group_by}
        ORDER BY paper_count DESC
    """

    try:
        result = conn.execute(query).fetchdf()
        conn.close()
        return result.to_dict(orient="records")
    except Exception as e:
        print(f"Statistics error: {e}")
        conn.close()
        return []


def get_rating_distribution(
    conference: str | None = None,
    year: int | None = None,
    step: float = 0.5,
    primary_area: str | None = None,
    files: list[str] | None = None,
) -> dict:
    """Get rating distribution for histogram with per-status breakdown and cumulative data

    Args:
        step: Bin width (e.g., 0.1 for 6.0-6.1, 0.5 for 6.0-6.5, 1.0 for 6-7)
        primary_area: Filter by primary area (partial match)
    """
    json_source = build_json_source(conference, year, files)
    if not json_source:
        return {
            "bins": [],
            "counts": [],
            "by_status": {},
            "cumulative_counts": [],
            "cumulative_by_status": {},
            "statuses": [],
            "total_count": 0,
            "status_totals": {},
            "step": step,
        }

    conn = get_connection()

    # Build WHERE clause
    conditions = ["rating_avg[1] IS NOT NULL"]
    if primary_area:
        area_escaped = primary_area.replace("'", "''")
        conditions.append(f"primary_area ILIKE '%{area_escaped}%'")

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT
            rating_avg[1] as rating,
            status
        FROM read_json({json_source}, auto_detect=true, format='array')
        WHERE {where_clause}
    """

    try:
        result = conn.execute(query).fetchdf()
        conn.close()
    except Exception as e:
        print(f"Distribution error: {e}")
        conn.close()
        return {
            "bins": [],
            "counts": [],
            "by_status": {},
            "cumulative_counts": [],
            "cumulative_by_status": {},
            "statuses": [],
            "total_count": 0,
            "status_totals": {},
            "step": step,
        }

    if result.empty:
        return {
            "bins": [],
            "counts": [],
            "by_status": {},
            "cumulative_counts": [],
            "cumulative_by_status": {},
            "statuses": [],
            "total_count": 0,
            "status_totals": {},
            "step": step,
        }

    ratings = result["rating"].tolist()
    statuses = result["status"].tolist()

    min_val = min(ratings)
    max_val = max(ratings)

    # Round min down and max up to clean boundaries based on step
    import math

    bin_start = math.floor(min_val / step) * step
    bin_end = math.ceil(max_val / step) * step

    # Generate clean bin edges
    num_bins = int(round((bin_end - bin_start) / step))
    if num_bins < 1:
        num_bins = 1

    bin_edges = [round(bin_start + i * step, 4) for i in range(num_bins + 1)]
    counts = [0] * num_bins

    # Track counts per status
    status_counts: dict[str, list[int]] = {}
    unique_statuses = set(statuses)
    for status in unique_statuses:
        if status:
            status_counts[status] = [0] * num_bins

    for rating, status in zip(ratings, statuses):
        # Find which bin this rating belongs to
        bin_idx = int((rating - bin_start) / step)
        bin_idx = max(0, min(bin_idx, num_bins - 1))  # Clamp to valid range
        counts[bin_idx] += 1
        if status and status in status_counts:
            status_counts[status][bin_idx] += 1

    # Compute cumulative counts from LOW to HIGH (left to right)
    # cumulative[i] = count of papers with rating <= bin_edges[i+1]
    cumulative_counts = [0] * num_bins
    cumulative_by_status: dict[str, list[int]] = {}
    for status in status_counts:
        cumulative_by_status[status] = [0] * num_bins

    running_total = 0
    running_by_status: dict[str, int] = {s: 0 for s in status_counts}

    for i in range(num_bins):
        running_total += counts[i]
        cumulative_counts[i] = running_total
        for status in status_counts:
            running_by_status[status] += status_counts[status][i]
            cumulative_by_status[status][i] = running_by_status[status]

    bin_centers = [
        round((bin_edges[i] + bin_edges[i + 1]) / 2, 4) for i in range(num_bins)
    ]

    # Status ordering: oral > spotlight > poster > active > reject > withdraw
    # Higher priority = should appear higher in stacked chart (rendered last)
    status_priority = {
        "oral": 6,
        "spotlight": 5,
        "poster": 4,
        "active": 3,
        "reject": 2,
        "withdraw": 1,
        "withdrawn": 1,
    }

    def get_status_order(s: str) -> tuple[int, int]:
        # Sort by priority (descending), then by count (descending)
        priority = status_priority.get(s.lower(), 0)
        total = sum(status_counts[s])
        return (-priority, -total)

    sorted_statuses = sorted(status_counts.keys(), key=get_status_order)

    # Calculate totals for percentage computation
    total_count = len(ratings)
    status_totals = {s: sum(status_counts[s]) for s in status_counts}

    return {
        "bins": bin_centers,
        "counts": counts,
        "bin_edges": bin_edges,
        "by_status": status_counts,
        "cumulative_counts": cumulative_counts,
        "cumulative_by_status": cumulative_by_status,
        "statuses": sorted_statuses,
        "total_count": total_count,
        "status_totals": status_totals,
        "step": step,
    }


def get_available_filters(
    conference: str | None = None,
    year: int | None = None,
    files: list[str] | None = None,
) -> dict:
    """Get available filter options from JSON files"""
    json_source = build_json_source(conference, year, files)
    if not json_source:
        return {"statuses": [], "primary_areas": [], "tracks": []}

    conn = get_connection()

    try:
        statuses = conn.execute(
            f"""
            SELECT DISTINCT status
            FROM read_json({json_source}, auto_detect=true, format='array')
            WHERE status IS NOT NULL AND status != ''
            ORDER BY status
        """
        ).fetchall()

        areas = conn.execute(
            f"""
            SELECT DISTINCT primary_area
            FROM read_json({json_source}, auto_detect=true, format='array')
            WHERE primary_area IS NOT NULL AND primary_area != ''
            ORDER BY primary_area
        """
        ).fetchall()

        tracks = conn.execute(
            f"""
            SELECT DISTINCT track
            FROM read_json({json_source}, auto_detect=true, format='array')
            WHERE track IS NOT NULL AND track != ''
            ORDER BY track
        """
        ).fetchall()

        conn.close()

        return {
            "statuses": [s[0] for s in statuses],
            "primary_areas": [a[0] for a in areas],
            "tracks": [t[0] for t in tracks],
        }
    except Exception as e:
        print(f"Filters error: {e}")
        conn.close()
        return {"statuses": [], "primary_areas": [], "tracks": []}


def get_yearly_stats(conference: str | None = None) -> list[dict]:
    """Get statistics grouped by year across multiple files"""
    available = get_available_data()

    if conference:
        if conference.lower() not in available:
            return []
        years = available[conference.lower()]
        files = [f"{conference.lower()}{y}.json" for y in years]
    else:
        files = [f.name for f in list_json_files()]

    results = []
    conn = get_connection()

    for filename in files:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            continue

        parsed = parse_filename(filepath)
        if not parsed:
            continue
        conf, yr = parsed

        try:
            query = f"""
                SELECT
                    COUNT(*) as paper_count,
                    AVG(rating_avg[1]) as avg_rating,
                    STDDEV(rating_avg[1]) as std_rating,
                    MIN(rating_avg[1]) as min_rating,
                    MAX(rating_avg[1]) as max_rating,
                    COUNT(*) FILTER (WHERE status IN ('Accept', 'Oral', 'Spotlight', 'Poster')) as accepted_count,
                    COUNT(*) FILTER (WHERE status IN ('Reject', 'Withdrawn', 'Withdraw')) as rejected_count
                FROM read_json('{filepath.as_posix()}', auto_detect=true, format='array')
            """
            row = conn.execute(query).fetchone()
            results.append(
                {
                    "conference": conf,
                    "year": yr,
                    "paper_count": row[0],
                    "avg_rating": row[1],
                    "std_rating": row[2],
                    "min_rating": row[3],
                    "max_rating": row[4],
                    "accepted_count": row[5],
                    "rejected_count": row[6],
                }
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    conn.close()
    return sorted(results, key=lambda x: (x["conference"], x["year"]))
