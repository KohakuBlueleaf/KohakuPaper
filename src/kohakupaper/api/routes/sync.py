"""
Sync endpoints for papercopilot data
"""

from fastapi import APIRouter, Query
from pathlib import Path
from ...paperlist_sync import (
    clone_or_update_repo,
    sync_conference_data,
    list_available_conferences,
    get_repo_path,
    DATA_DIR,
)

router = APIRouter()


def get_local_conferences() -> set[str]:
    """Get set of locally downloaded conference files (e.g., 'iclr2026')"""
    if not DATA_DIR.exists():
        return set()
    return {f.stem for f in DATA_DIR.glob("*.json")}


@router.get("/available")
async def list_conferences():
    """List all available conferences from papercopilot with local status"""
    try:
        available = list_available_conferences()
        local = get_local_conferences()

        # Group by conference name
        conferences_map = {}
        for item in available:
            conf = item["conference"]
            if conf not in conferences_map:
                conferences_map[conf] = []
            key = f"{conf}{item['year']}"
            conferences_map[conf].append(
                {
                    "year": item["year"],
                    "size_mb": round(item["size_mb"], 1),
                    "downloaded": key in local,
                }
            )

        # Sort years descending for each conference
        result = []
        for conf in sorted(conferences_map.keys()):
            years = sorted(conferences_map[conf], key=lambda x: -x["year"])
            result.append(
                {
                    "conference": conf,
                    "years": years,
                }
            )

        return {
            "conferences": result,
            "local_count": len(local),
        }
    except Exception as e:
        return {"error": str(e), "conferences": []}


@router.get("/local")
async def list_local():
    """List locally downloaded conferences"""
    local = get_local_conferences()
    result = []
    for name in sorted(local):
        # Parse conference and year
        for i in range(len(name) - 1, 3, -1):
            if name[i - 4 : i].isdigit():
                conf = name[: i - 4]
                year = int(name[i - 4 : i])
                file_path = DATA_DIR / f"{name}.json"
                result.append(
                    {
                        "conference": conf,
                        "year": year,
                        "file": str(file_path),
                        "size_mb": (
                            round(file_path.stat().st_size / (1024 * 1024), 1)
                            if file_path.exists()
                            else 0
                        ),
                    }
                )
                break
    return result


@router.post("/update-repo")
async def update_repo():
    """Update the paperlists repository"""
    try:
        success = clone_or_update_repo()
        return {"success": success}
    except Exception as e:
        return {"error": str(e), "success": False}


@router.post("/conference")
async def sync_conference(
    conference: str = Query(..., description="Conference name (e.g., iclr, nips)"),
    year: int = Query(..., description="Conference year (e.g., 2026)"),
):
    """
    Sync a specific conference - downloads data and computes diffs.
    NO git clone required - uses GitHub API for everything!
    """
    try:
        output_path = sync_conference_data(conference, year)
        return {
            "success": True,
            "message": f"Synced {conference}{year} with diffs",
            "output_file": str(output_path),
        }
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/status")
async def get_sync_status():
    """Check sync status"""
    repo_path = get_repo_path()
    local = get_local_conferences()

    return {
        "repo_exists": repo_path.exists(),
        "repo_path": str(repo_path),
        "local_count": len(local),
        "data_dir": str(DATA_DIR),
    }
