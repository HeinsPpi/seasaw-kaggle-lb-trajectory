#!/usr/bin/env python3
"""Plot the per-episode skill rating of Seasaw's latest two submissions."""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest


COMPETITION = "pokemon-tcg-ai-battle"
TEAM_NAME = "Seasaw"
DEFAULT_FINAL_SUBMISSION_DEADLINE = pd.Timestamp("2026-08-16T23:59:59Z")
META_KAGGLE = "kaggle/meta-kaggle"
EPISODE_AGENTS_FILE = "EpisodeAgents.csv"
INTERNAL_EPISODES_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
RANGE_CHUNK_BYTES = 64 * 1024 * 1024
RANGE_OVERLAP_BYTES = 16 * 1024
MAX_RANGE_BYTES = 6 * 1024 * 1024 * 1024


def find_team(api: KaggleApi) -> tuple[int, str]:
    page_token = None
    for _ in range(100):
        diagnostic = io.StringIO()
        with contextlib.redirect_stdout(diagnostic):
            rows = api.competition_leaderboard_view(
                COMPETITION, page_size=200, page_token=page_token
            ) or []
        if not rows:
            break
        for row in rows:
            if str(getattr(row, "team_name", "")).casefold() == TEAM_NAME.casefold():
                return int(row.team_id), str(getattr(row, "team_name", TEAM_NAME))
        match = re.search(r"Next Page Token\s*=\s*(\S+)", diagnostic.getvalue())
        if not match:
            break
        page_token = match.group(1)
    raise RuntimeError(f"{TEAM_NAME!r} was not found in the visible leaderboard")


def latest_submissions(api: KaggleApi, team_id: int) -> list[tuple[str, object]]:
    rows = [
        row for row in (api.competition_team_submissions(team_id) or [])
        if getattr(row, "id", None) is not None
    ]
    rows.sort(
        key=lambda row: pd.Timestamp(getattr(row, "date_submitted", 0)).timestamp()
        if getattr(row, "date_submitted", None) else float("-inf")
    )
    if len(rows) < 2:
        raise RuntimeError(f"Only {len(rows)} active submission(s) found for {TEAM_NAME}")
    return [(str(row.id), row) for row in rows[-2:]]


def final_deadline(api: KaggleApi) -> pd.Timestamp:
    configured = os.environ.get("FINAL_SUBMISSION_DEADLINE")
    if configured:
        value = pd.Timestamp(configured)
        return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    return DEFAULT_FINAL_SUBMISSION_DEADLINE


def submission_episode_targets(
    api: KaggleApi, submission_ids: set[str], deadline: pd.Timestamp
) -> dict[str, dict]:
    """Get EpisodeId, timestamp and our agent index from competition API."""
    result: dict[str, dict] = {}
    for submission_id in sorted(submission_ids):
        for episode in api.competition_list_episodes(int(submission_id)) or []:
            timestamp = getattr(episode, "end_time", None) or getattr(episode, "create_time", None)
            timestamp = pd.to_datetime(timestamp, utc=True, errors="coerce")
            if pd.isna(timestamp) or timestamp < deadline:
                continue
            agent_index = next(
                (
                    int(agent.index)
                    for agent in (getattr(episode, "agents", None) or [])
                    if str(getattr(agent, "submission_id", "")) == submission_id
                ),
                None,
            )
            if agent_index is not None:
                result[str(episode.id)] = {
                    "timestamp": timestamp,
                    "submission_id": submission_id,
                    "index": str(agent_index),
                }
    if not result:
        raise RuntimeError("Kaggle returned no post-deadline episodes for the latest submissions")
    return result


def internal_episode_rating_history(submission_ids: set[str]) -> pd.DataFrame:
    """Try Kaggle Web's episode endpoint with the configured API bearer token.

    The endpoint is queried at most once per selected submission.  Some Kaggle
    accounts accept API-token auth here; others require a browser/XSRF session,
    in which case this returns an empty frame and Meta Kaggle is used.
    """
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        return pd.DataFrame()
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "seasaw-lb-trajectory/1.0",
    })
    records = []
    for submission_id in sorted(submission_ids):
        response = session.post(
            INTERNAL_EPISODES_URL,
            json={"submissionId": submission_id},
            timeout=60,
        )
        if response.status_code != 200:
            print(
                f"Kaggle EpisodeService token auth unavailable (HTTP {response.status_code})",
                flush=True,
            )
            return pd.DataFrame()
        for episode in response.json().get("episodes", []) or []:
            timestamp = episode.get("endTime") or episode.get("createTime")
            for agent in episode.get("agents", []) or []:
                if str(agent.get("submissionId")) != submission_id:
                    continue
                score = agent.get("updatedScore")
                if score is None:
                    continue
                records.append({
                    "timestamp": timestamp,
                    "submission_id": submission_id,
                    "episode_id": str(episode.get("id")),
                    "initial_score": agent.get("initialScore"),
                    "score": score,
                })
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["initial_score"] = pd.to_numeric(frame["initial_score"], errors="coerce")
    return frame.dropna(subset=["timestamp", "score"])


def meta_kaggle_raw_url(api: KaggleApi) -> str:
    """Ask Kaggle Dataset API for a short-lived raw EpisodeAgents CSV URL."""
    request = ApiDownloadDatasetRequest()
    request.owner_slug = "kaggle"
    request.dataset_slug = "meta-kaggle"
    request.file_name = EPISODE_AGENTS_FILE
    request.raw = True
    with api.build_kaggle_client() as kaggle:
        response = kaggle.datasets.dataset_api_client.download_dataset(request)
        url = response.url
        response.close()
    if not url:
        raise RuntimeError("Kaggle did not return the Meta Kaggle raw file URL")
    return url


def _range(session: requests.Session, url: str, start: int, end: int) -> tuple[bytes, int]:
    response = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    if response.status_code != 206:
        response.close()
        raise RuntimeError(
            f"Meta Kaggle storage ignored byte range (HTTP {response.status_code}); "
            "refusing to download the 24GB file"
        )
    content_range = response.headers.get("Content-Range", "")
    match = re.search(r"/(\d+)$", content_range)
    if not match:
        raise RuntimeError(f"Missing total size in Content-Range: {content_range!r}")
    return response.content, int(match.group(1))


def _episode_id_at(lines: list[str], index: int) -> int | None:
    try:
        return int(next(csv.reader([lines[index]]))[1])
    except (ValueError, IndexError, StopIteration):
        return None


def fetch_episode_rating_history(
    api: KaggleApi, submission_ids: set[str], deadline: pd.Timestamp
) -> pd.DataFrame:
    """Extract only the wanted rows from Meta Kaggle EpisodeAgents.csv.

    EpisodeAgents is currently about 24GB.  Its rows are emitted in increasing
    database/episode order.  We obtain the wanted EpisodeIds using the small
    competition endpoint, then range-read the CSV backwards and stop as soon as
    the scan has passed the oldest wanted EpisodeId.  No win/loss-derived score
    or current-rating snapshot is used.
    """
    live = internal_episode_rating_history(submission_ids)
    if not live.empty:
        return live.sort_values(["timestamp", "episode_id"]).reset_index(drop=True)

    episode_targets = submission_episode_targets(api, submission_ids, deadline)
    wanted_episode_ids = set(episode_targets)
    oldest_episode_id = min(map(int, wanted_episode_ids))
    url = meta_kaggle_raw_url(api)
    session = requests.Session()

    header_bytes, total_bytes = _range(session, url, 0, 4095)
    header_line = header_bytes.decode("utf-8", errors="replace").splitlines()[0]
    columns = next(csv.reader([header_line.lstrip("\ufeff")]))
    required = {"EpisodeId", "SubmissionId", "InitialScore", "UpdatedScore"}
    if not required.issubset(columns):
        raise RuntimeError(f"Unexpected EpisodeAgents columns: {columns}")

    records: dict[tuple[str, str], dict] = {}
    scanned = 0
    end = total_bytes - 1
    while end > 0 and scanned < MAX_RANGE_BYTES:
        start = max(0, end - RANGE_CHUNK_BYTES + 1)
        # Extend toward the already-read/newer side so a CSV row cut at the
        # logical ``end`` is complete in this request. Dict keys de-duplicate
        # rows seen in the overlap.
        request_end = min(total_bytes - 1, end + RANGE_OVERLAP_BYTES)
        data, confirmed_total = _range(session, url, start, request_end)
        if confirmed_total != total_bytes:
            raise RuntimeError("Meta Kaggle file changed during the range scan; retry the workflow")
        lines = data.decode("utf-8", errors="replace").splitlines()
        first = 1 if start else 0
        last = None if request_end == total_bytes - 1 else -1
        complete_lines = lines[first:last]
        for line in complete_lines:
            # EpisodeAgents' relational key is (EpisodeId, Index).  Join on
            # that public key instead of relying on a textual SubmissionId
            # representation in the 24GB dump.
            prefix = line.split(",", 3)
            if len(prefix) < 3:
                continue
            episode_id, agent_index = prefix[1], prefix[2]
            target = episode_targets.get(episode_id)
            if target is None or agent_index != target["index"]:
                continue
            values = next(csv.reader([line]))
            if len(values) != len(columns):
                continue
            row = dict(zip(columns, values))
            submission_id = target["submission_id"]
            records[(submission_id, episode_id)] = {
                "timestamp": target["timestamp"],
                "submission_id": submission_id,
                "episode_id": episode_id,
                "initial_score": row["InitialScore"],
                "score": row["UpdatedScore"],
            }
        scanned += end - start + 1
        newest_in_chunk = _episode_id_at(complete_lines, -1) if complete_lines else None
        if newest_in_chunk is not None and newest_in_chunk < oldest_episode_id:
            break
        end = start - 1
        print(
            f"Meta Kaggle scan: {scanned / (1024**2):.0f} MiB, "
            f"matched {len(records)} rating rows",
            flush=True,
        )

    if not records:
        raise RuntimeError(
            "No matching UpdatedScore rows were found in Meta Kaggle. "
            "The daily dataset may not have published these episodes yet."
        )
    frame = pd.DataFrame(records.values())
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["initial_score"] = pd.to_numeric(frame["initial_score"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "score"])
    frame = frame.drop_duplicates(["submission_id", "episode_id"], keep="last")
    return frame.sort_values(["timestamp", "episode_id"]).reset_index(drop=True)


def load_after_deadline(
    frame: pd.DataFrame, deadline: pd.Timestamp, submission_ids: set[str]
) -> pd.DataFrame:
    result = frame[
        frame["submission_id"].astype(str).isin(submission_ids)
        & (frame["timestamp"] >= deadline)
    ].copy()
    if result.empty:
        raise RuntimeError("No post-deadline rating rows matched the latest submissions")
    return result.sort_values("timestamp")


def plot(frame: pd.DataFrame, deadline: pd.Timestamp, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for submission_id, rows in frame.groupby("submission_id", sort=False):
        axis.plot(
            rows["timestamp"], rows["score"], marker=".", linewidth=1.5,
            label=f"Submission {submission_id}",
        )
    axis.axvline(
        deadline, color="black", linestyle="--", linewidth=1,
        label="Final Submission Deadline",
    )
    axis.set_title("Seasaw — per-episode Kaggle skill rating")
    axis.set_xlabel("Episode time (UTC)")
    axis.set_ylabel("Updated skill rating")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("lb_trajectory"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    team_id, _ = find_team(api)
    submissions = latest_submissions(api, team_id)
    submission_ids = {submission_id for submission_id, _ in submissions}
    deadline = final_deadline(api)
    frame = load_after_deadline(
        fetch_episode_rating_history(api, submission_ids, deadline), deadline, submission_ids
    )
    frame.to_csv(args.output_dir / "lb_trajectory_after_deadline.csv", index=False)
    plot(frame, deadline, args.output_dir / "lb_trajectory_after_deadline.png")
    print(f"team={TEAM_NAME} team_id={team_id} rows={len(frame)}")


if __name__ == "__main__":
    main()
