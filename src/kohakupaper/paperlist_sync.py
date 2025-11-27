"""
Paperlist sync module - downloads and tracks history from papercopilot/paperlists
Computes rating diffs between first non-empty score and current score
"""

import json
import subprocess
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from .config import (
    PAPERLISTS_DIR,
    PAPERLISTS_GIT_URL,
    PAPERLISTS_API_URL,
    DATA_DIR,
)


def get_repo_path() -> Path:
    """Get path to cloned paperlists repo"""
    return PAPERLISTS_DIR


def clone_or_update_repo() -> bool:
    """Clone the paperlists repo or update if already exists"""
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


def _fetch_json(url: str, timeout: int = 30) -> dict | list | None:
    """Fetch JSON from URL"""
    try:
        req = Request(url, headers={"User-Agent": "KohakuPaper/0.4.0"})
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        print(f"Error fetching {url}: {e}")
        return None


def list_available_conferences_from_api() -> list[dict]:
    """
    List all available conferences and years from GitHub API (no git clone required).
    This is used for browsing available data before deciding what to sync.
    """
    # First, get all directories (conferences) from the repo root
    data = _fetch_json(PAPERLISTS_API_URL)
    if not data:
        return []

    conferences = []

    for item in data:
        if item.get("type") != "dir":
            continue
        conf_name = item.get("name", "")
        if conf_name.startswith("."):
            continue

        # Get JSON files in this conference directory
        conf_url = f"{PAPERLISTS_API_URL}/{conf_name}"
        conf_data = _fetch_json(conf_url)
        if not conf_data:
            continue

        for file_item in conf_data:
            if file_item.get("type") != "file":
                continue
            filename = file_item.get("name", "")
            if not filename.endswith(".json"):
                continue

            # Extract year from filename (e.g., iclr2024.json -> 2024)
            name = filename[:-5]  # Remove .json
            if len(name) >= 4 and name[-4:].isdigit():
                year = int(name[-4:])
                size_bytes = file_item.get("size", 0)
                conferences.append(
                    {
                        "conference": conf_name,
                        "year": year,
                        "file": filename,
                        "size_mb": size_bytes / (1024 * 1024),
                    }
                )

    return sorted(conferences, key=lambda x: (x["conference"], x["year"]))


def get_file_history(conference: str, year: int, max_commits: int = 50) -> list[dict]:
    """Get git history for a specific conference file"""
    repo_path = get_repo_path()
    file_path = f"{conference}/{conference}{year}.json"

    try:
        # Get commit hashes for this file
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
        print(f"Failed to get history: {e}")
        return []


def get_file_at_commit(conference: str, year: int, commit_hash: str) -> Optional[list]:
    """Get file content at a specific commit"""
    repo_path = get_repo_path()
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
        print(f"Failed to get file at commit {commit_hash}: {e}")
        return None


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
    Compute rating/confidence diffs for all papers
    Returns dict mapping paper_id to diff info

    Diff info includes:
    - rating_first: first non-empty rating
    - rating_current: current rating
    - rating_diff: per-reviewer diff (current - first)
    - confidence_first, confidence_current, confidence_diff: same for confidence
    """
    repo_path = get_repo_path()
    if not repo_path.exists():
        print("Repo not found, cloning...")
        clone_or_update_repo()

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
    Sync conference data: download/update repo, compute diffs, save enhanced JSON
    Returns path to the output file
    """
    # Ensure repo is up to date
    clone_or_update_repo()

    repo_path = get_repo_path()
    source_file = repo_path / conference / f"{conference}{year}.json"

    if not source_file.exists():
        raise FileNotFoundError(f"Conference file not found: {source_file}")

    # Load current data
    with open(source_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # Compute diffs
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

    print(f"Saved enhanced data to {output_file}")
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
        print("  list              - List available conferences")
        print("  sync <conf> <year> - Sync conference data (e.g., sync iclr 2026)")
        print("  update            - Update repo only")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        clone_or_update_repo()
        confs = list_available_conferences()
        for c in confs:
            print(f"{c['conference']}{c['year']}: {c['size_mb']:.1f} MB")

    elif command == "sync":
        if len(sys.argv) < 4:
            print("Usage: python paperlist_sync.py sync <conference> <year>")
            sys.exit(1)
        conf = sys.argv[2]
        year = int(sys.argv[3])
        sync_conference_data(conf, year)

    elif command == "update":
        clone_or_update_repo()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
