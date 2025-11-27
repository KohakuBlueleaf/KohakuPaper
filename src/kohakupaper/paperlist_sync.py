"""
Paperlist sync module - downloads and tracks history from papercopilot/paperlists
Computes rating diffs between first non-empty score and current score
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from .config import (
    PAPERLISTS_DIR,
    PAPERLISTS_GIT_URL,
    PAPERLISTS_RAW_URL,
    PAPERLISTS_API_URL,
    DATA_DIR,
    CONFERENCES,
)

# GitHub token for API requests (optional, but recommended to avoid rate limits)
# Set via environment variable: GITHUB_TOKEN
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def get_repo_path() -> Path:
    """Get path to cloned paperlists repo"""
    return PAPERLISTS_DIR


def clone_or_update_repo() -> bool:
    """Clone the paperlists repo or update if already exists (full clone - legacy)"""
    repo_path = get_repo_path()

    if repo_path.exists() and (repo_path / ".git").exists():
        # Update existing repo
        print(f"Updating existing repo at {repo_path}")
        try:
            subprocess.run(
                ["git", "fetch", "--all"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "reset", "--hard", "origin/main"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            print("Repo updated successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to update repo: {e}")
            return False
    else:
        # Clone new repo
        print(f"Cloning repo to {repo_path}")
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "50", PAPERLISTS_GIT_URL, str(repo_path)],
                check=True,
                capture_output=True,
            )
            print("Repo cloned successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repo: {e}")
            return False


def _init_sparse_repo() -> bool:
    """Initialize sparse checkout repo if it doesn't exist"""
    repo_path = get_repo_path()

    if repo_path.exists() and (repo_path / ".git").exists():
        return True  # Already initialized

    print(f"Initializing sparse checkout repo at {repo_path}...")
    repo_path.mkdir(parents=True, exist_ok=True)

    try:
        # Init empty repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        # Add remote
        subprocess.run(
            ["git", "remote", "add", "origin", PAPERLISTS_GIT_URL],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        # Enable sparse checkout (no-cone mode for file-level patterns)
        subprocess.run(
            ["git", "sparse-checkout", "init", "--no-cone"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        # Start with empty patterns
        subprocess.run(
            ["git", "sparse-checkout", "set", "--no-cone"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        print("Sparse repo initialized")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to init sparse repo: {e}")
        import shutil

        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)
        return False


def ensure_file_available(conference: str, year: int) -> bool:
    """
    Ensure a specific conference file is available locally via sparse checkout.
    Only fetches the single file needed - MUCH faster than full clone!
    """
    repo_path = get_repo_path()
    file_pattern = f"{conference}/{conference}{year}.json"
    file_path = repo_path / conference / f"{conference}{year}.json"

    # Initialize sparse repo if needed
    if not _init_sparse_repo():
        return False

    print(f"Fetching {file_pattern} via sparse checkout...")

    try:
        # Add this file to sparse-checkout patterns
        subprocess.run(
            ["git", "sparse-checkout", "add", file_pattern],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        # Fetch with enough history for diffs
        subprocess.run(
            ["git", "fetch", "origin", "main", "--depth=50"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        # Checkout/update the file
        subprocess.run(
            ["git", "checkout", "FETCH_HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        if file_path.exists():
            print(f"File {file_pattern} ready")
            return True
        else:
            print(f"File {file_pattern} not found in repo")
            return False

    except subprocess.CalledProcessError as e:
        print(f"Failed to fetch {file_pattern}: {e}")
        return False


def _fetch_json(
    url: str,
    timeout: int = 30,
    max_retries: int = 3,
    use_auth: bool = True,
) -> dict | list | None:
    """
    Fetch JSON from URL with retry and exponential backoff.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries on failure
        use_auth: Whether to use GitHub token if available
    """
    headers = {"User-Agent": "KohakuPaper/0.4.0"}

    # Add GitHub token for API requests if available
    if use_auth and GITHUB_TOKEN and "api.github.com" in url:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    last_error = None
    for attempt in range(max_retries):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            last_error = e
            if e.code == 403:
                # Rate limit - check reset time
                reset_time = e.headers.get("X-RateLimit-Reset")
                remaining = e.headers.get("X-RateLimit-Remaining", "0")
                if remaining == "0" and reset_time:
                    wait_seconds = int(reset_time) - int(time.time()) + 1
                    if wait_seconds > 0 and wait_seconds < 300:  # Max 5 min wait
                        print(f"Rate limited. Waiting {wait_seconds}s for reset...")
                        time.sleep(wait_seconds)
                        continue
                print(
                    f"Rate limit exceeded. Set GITHUB_TOKEN env var for higher limits."
                )
            elif e.code == 404:
                # Not found - don't retry
                print(f"Not found: {url}")
                return None
            elif e.code >= 500:
                # Server error - retry with backoff
                pass
            else:
                # Other client error - don't retry
                print(f"HTTP error {e.code}: {url}")
                return None
        except (URLError, json.JSONDecodeError, TimeoutError) as e:
            last_error = e

        # Exponential backoff before retry
        if attempt < max_retries - 1:
            wait_time = 2**attempt  # 1, 2, 4 seconds
            print(f"Retry {attempt + 1}/{max_retries} in {wait_time}s...")
            time.sleep(wait_time)

    print(f"Error fetching {url} after {max_retries} attempts: {last_error}")
    return None


def fetch_conference_file_raw(
    conference: str, year: int, timeout: int = 60
) -> list | None:
    """
    Fetch a single conference JSON file directly from GitHub raw URL.
    This avoids cloning the entire repo - just downloads the one file needed.
    """
    url = f"{PAPERLISTS_RAW_URL}/{conference}/{conference}{year}.json"
    print(f"Fetching {conference}{year}.json from GitHub...")
    return _fetch_json(url, timeout=timeout)


def list_available_conferences_from_api() -> list[dict]:
    """
    List all available conferences and years.

    Uses known conferences from config and generates expected years (2018-2026).
    This avoids hitting GitHub API rate limits (was making N+1 API calls before).
    File sizes are estimated since we don't fetch from API anymore.
    """
    conferences = []

    # Generate expected conference/year combinations from our known conferences
    # Most conferences have data from around 2018-2026
    current_year = 2026  # Update this as needed
    start_year = 2018

    for conf_name in CONFERENCES.keys():
        for year in range(start_year, current_year + 1):
            conferences.append(
                {
                    "conference": conf_name,
                    "year": year,
                    "file": f"{conf_name}{year}.json",
                    "size_mb": 0,  # Unknown without API call
                }
            )

    return sorted(conferences, key=lambda x: (x["conference"], x["year"]))


def get_file_history_from_api(
    conference: str, year: int, max_commits: int = 50
) -> list[dict]:
    """Get commit history for a specific file using GitHub API (no git clone needed)"""
    file_path = f"{conference}/{conference}{year}.json"
    url = f"https://api.github.com/repos/papercopilot/paperlists/commits?path={file_path}&per_page={max_commits}"

    print(f"Fetching commit history for {file_path}...")
    data = _fetch_json(url, timeout=30)
    if not data:
        return []

    commits = []
    for item in data:
        sha = item.get("sha", "")
        date = item.get("commit", {}).get("committer", {}).get("date", "")
        if sha:
            commits.append({"hash": sha, "date": date})

    return commits


def get_file_at_commit_from_api(
    conference: str, year: int, commit_hash: str
) -> Optional[list]:
    """Get file content at a specific commit using GitHub raw URL (no git clone needed)"""
    file_path = f"{conference}/{conference}{year}.json"
    url = f"https://raw.githubusercontent.com/papercopilot/paperlists/{commit_hash}/{file_path}"

    return _fetch_json(url, timeout=60)


def get_file_history(conference: str, year: int, max_commits: int = 50) -> list[dict]:
    """Get git history for a specific conference file (uses local repo if available, else API)"""
    repo_path = get_repo_path()

    # If repo exists locally, use git commands (faster)
    if repo_path.exists() and (repo_path / ".git").exists():
        file_path = f"{conference}/{conference}{year}.json"
        try:
            result = subprocess.run(
                ["git", "log", f"-{max_commits}", "--format=%H|%ci", "--", file_path],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            commits = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("|")
                    if len(parts) == 2:
                        commits.append({"hash": parts[0], "date": parts[1]})

            return commits
        except subprocess.CalledProcessError as e:
            print(f"Failed to get history from local repo: {e}")
            # Fall through to API

    # Use GitHub API (no clone needed)
    return get_file_history_from_api(conference, year, max_commits)


def get_file_at_commit(conference: str, year: int, commit_hash: str) -> Optional[list]:
    """Get file content at a specific commit (uses local repo if available, else API)"""
    repo_path = get_repo_path()

    # If repo exists locally, use git commands (faster)
    if repo_path.exists() and (repo_path / ".git").exists():
        file_path = f"{conference}/{conference}{year}.json"
        try:
            result = subprocess.run(
                ["git", "show", f"{commit_hash}:{file_path}"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"Failed to get file from local repo: {e}")
            # Fall through to API

    # Use GitHub raw URL (no clone needed)
    return get_file_at_commit_from_api(conference, year, commit_hash)


def parse_scores(score_str: str, sort: bool = False) -> list[float]:
    """Parse score string like '4;4;6;6' into list of floats.

    Args:
        score_str: Semicolon-separated score string
        sort: If True, sort high to low (for display). If False, preserve order (for diff calculation)
    """
    if not score_str:
        return []
    try:
        scores = [float(s.strip()) for s in score_str.split(";") if s.strip()]
        if sort:
            return sorted(scores, reverse=True)
        return scores
    except ValueError:
        return []


def compute_paper_diffs(
    conference: str, year: int, max_history_commits: int = 30
) -> dict[str, dict]:
    """
    Compute rating/confidence diffs for all papers.
    Uses GitHub API if repo not cloned locally - NO git clone required!

    Returns dict mapping paper_id to diff info

    Diff info includes:
    - rating_first: first non-empty rating
    - rating_current: current rating
    - rating_diff: per-reviewer diff (current - first)
    - confidence_first, confidence_current, confidence_diff: same for confidence
    """

    # Get commit history
    commits = get_file_history(conference, year, max_history_commits)
    if not commits:
        print("No commits found")
        return {}

    print(f"Found {len(commits)} commits for {conference}{year}")

    # Get current data (most recent commit)
    current_data = get_file_at_commit(conference, year, commits[0]["hash"])
    if not current_data:
        return {}

    # Build lookup by paper ID
    papers_current = {p["id"]: p for p in current_data}

    # Track first non-empty scores for each paper
    first_scores: dict[str, dict] = (
        {}
    )  # paper_id -> {rating: str, confidence: str, date: str}

    # Go through history from oldest to newest to find first non-empty scores
    for commit in reversed(commits):
        historical_data = get_file_at_commit(conference, year, commit["hash"])
        if not historical_data:
            continue

        for paper in historical_data:
            paper_id = paper.get("id")
            if not paper_id:
                continue

            rating = paper.get("rating", "")
            confidence = paper.get("confidence", "")

            # Check if this paper has scores and we haven't recorded first scores yet
            if paper_id not in first_scores:
                if rating and rating.strip():
                    first_scores[paper_id] = {
                        "rating": rating,
                        "confidence": confidence,
                        "date": commit["date"],
                    }

    # Compute diffs
    diffs = {}
    for paper_id, paper in papers_current.items():
        current_rating = paper.get("rating", "")
        current_confidence = paper.get("confidence", "")

        # Parse scores - unsorted for diff calculation (preserve reviewer order)
        current_rating_scores_raw = parse_scores(current_rating, sort=False)
        current_confidence_scores_raw = parse_scores(current_confidence, sort=False)

        # Sorted for display
        current_rating_scores = parse_scores(current_rating, sort=True)
        current_confidence_scores = parse_scores(current_confidence, sort=True)

        diff_info = {
            # Only set if we have actual scores (not empty arrays)
            "rating_current": current_rating_scores if current_rating_scores else None,
            "confidence_current": (
                current_confidence_scores if current_confidence_scores else None
            ),
            "rating_first": None,
            "confidence_first": None,
            "rating_diff": None,
            "confidence_diff": None,
            "has_diff": False,
            "first_date": None,
        }

        if paper_id in first_scores:
            first = first_scores[paper_id]
            # Parse first scores - unsorted for diff calculation
            first_rating_scores_raw = parse_scores(first["rating"], sort=False)
            first_confidence_scores_raw = parse_scores(first["confidence"], sort=False)

            # Sorted for display
            first_rating_scores = parse_scores(first["rating"], sort=True)
            first_confidence_scores = parse_scores(first["confidence"], sort=True)

            diff_info["first_date"] = first["date"]

            # Compute per-score diff if lengths match (using raw unsorted scores)
            # ONLY set rating_first if we can compute a valid diff (same number of reviewers)
            if (
                len(current_rating_scores_raw) == len(first_rating_scores_raw)
                and len(first_rating_scores_raw) > 0
            ):
                rating_diff_raw = [
                    c - f
                    for c, f in zip(current_rating_scores_raw, first_rating_scores_raw)
                ]
                # Always set rating_first when lengths match (for Init column)
                diff_info["rating_first"] = first_rating_scores

                if any(d != 0 for d in rating_diff_raw):
                    # Sort the diffs to match the sorted display order
                    # Pair current scores with their diffs, sort by current score descending
                    paired = list(zip(current_rating_scores_raw, rating_diff_raw))
                    paired.sort(key=lambda x: -x[0])
                    rating_diff = [d for _, d in paired]
                    diff_info["rating_diff"] = rating_diff
                    diff_info["has_diff"] = True

            # Same for confidence
            if (
                len(current_confidence_scores_raw) == len(first_confidence_scores_raw)
                and len(first_confidence_scores_raw) > 0
            ):
                confidence_diff_raw = [
                    c - f
                    for c, f in zip(
                        current_confidence_scores_raw, first_confidence_scores_raw
                    )
                ]
                # Always set confidence_first when lengths match
                diff_info["confidence_first"] = first_confidence_scores

                if any(d != 0 for d in confidence_diff_raw):
                    # Sort the diffs to match the sorted display order
                    paired = list(
                        zip(current_confidence_scores_raw, confidence_diff_raw)
                    )
                    paired.sort(key=lambda x: -x[0])
                    confidence_diff = [d for _, d in paired]
                    diff_info["confidence_diff"] = confidence_diff

        diffs[paper_id] = diff_info

    return diffs


def sync_conference_data(conference: str, year: int) -> Path:
    """
    Sync conference data: fetch file via sparse checkout and compute diffs.
    Uses git sparse-checkout to fetch only the needed file - fast and no API rate limits!

    Args:
        conference: Conference name (e.g., 'iclr')
        year: Conference year (e.g., 2026)

    Returns path to the output file
    """
    repo_path = get_repo_path()
    file_path = repo_path / conference / f"{conference}{year}.json"

    # Fetch the file via sparse checkout
    if not ensure_file_available(conference, year):
        raise FileNotFoundError(f"Failed to fetch {conference}{year}.json")

    # Load the file
    with open(file_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # Compute diffs using local git history
    print(f"Computing diffs for {conference}{year}...")
    diffs = compute_paper_diffs(conference, year)

    # Enhance papers with diff info
    for paper in papers:
        paper_id = paper.get("id")
        if paper_id and paper_id in diffs:
            diff_info = diffs[paper_id]
            paper["_diff"] = diff_info

    # Save to data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / f"{conference}{year}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    print(f"Saved data to {output_file}")
    return output_file


def list_available_conferences(use_api: bool = True) -> list[dict]:
    """
    List all available conferences and years.

    Args:
        use_api: If True (default), fetch from GitHub API without cloning.
                 If False, read from local cloned repo (requires clone_or_update_repo first).
    """
    if use_api:
        return list_available_conferences_from_api()

    # Fallback to local repo (only if explicitly requested and repo exists)
    repo_path = get_repo_path()
    if not repo_path.exists():
        # Don't auto-clone, return empty or use API
        return list_available_conferences_from_api()

    conferences = []
    for conf_dir in repo_path.iterdir():
        if conf_dir.is_dir() and not conf_dir.name.startswith("."):
            for json_file in conf_dir.glob("*.json"):
                # Parse conference and year from filename
                name = json_file.stem
                # Extract year (last 4 digits)
                if len(name) >= 4 and name[-4:].isdigit():
                    conf_name = name[:-4]
                    year = int(name[-4:])
                    conferences.append(
                        {
                            "conference": conf_name,
                            "year": year,
                            "file": str(json_file),
                            "size_mb": json_file.stat().st_size / (1024 * 1024),
                        }
                    )

    return sorted(conferences, key=lambda x: (x["conference"], x["year"]))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python paperlist_sync.py <command> [args]")
        print("Commands:")
        print("  list                - List available conferences")
        print(
            "  sync <conf> <year>  - Sync conference with diffs (no git clone needed!)"
        )
        print(
            "  update              - Update local git repo (optional, for faster local access)"
        )
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        confs = list_available_conferences(use_api=True)
        for c in confs:
            print(f"{c['conference']}{c['year']}")

    elif command == "sync":
        # Sync with diffs - no git clone needed!
        if len(sys.argv) < 4:
            print("Usage: python paperlist_sync.py sync <conference> <year>")
            sys.exit(1)
        conf = sys.argv[2]
        year = int(sys.argv[3])
        sync_conference_data(conf, year)

    elif command == "update":
        # Optional - clone/update local repo for faster access
        clone_or_update_repo()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
