#!/usr/bin/env python3
"""Plot the per-episode skill rating of Seasaw's latest two submissions."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from kaggle.api.kaggle_api_extended import KaggleApi


COMPETITION = "pokemon-tcg-ai-battle"
TEAM_NAME = "Seasaw"
DEFAULT_FINAL_SUBMISSION_DEADLINE = pd.Timestamp("2026-08-16T23:59:59Z")
EPISODES_URL = "https://www.kaggle.com/api/v1/competitions/submissions/{submission_id}/episodes"


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


def _first(mapping: dict, *names: str):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def fetch_episode_rating_history(submission_ids: set[str]) -> pd.DataFrame:
    """Read absolute post-episode ratings from Kaggle's official episode API.

    Kaggle CLI's generated ``ApiEpisodeAgent`` currently omits score fields
    while parsing the response.  The API response itself carries the simulation
    fields ``initialScore`` and ``updatedScore``.  Reading the JSON before the
    generated-model conversion preserves the rating trajectory.
    """
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN is not set")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "seasaw-lb-trajectory/1.0",
    })
    records: list[dict] = []
    for submission_id in sorted(submission_ids):
        response = session.get(
            EPISODES_URL.format(submission_id=submission_id), timeout=60
        )
        response.raise_for_status()
        payload = response.json()
        episodes = payload.get("episodes", payload if isinstance(payload, list) else [])
        for episode in episodes:
            timestamp = _first(episode, "endTime", "end_time", "createTime", "create_time")
            episode_id = _first(episode, "id", "episodeId", "episode_id")
            for agent in episode.get("agents", []) or []:
                agent_submission = _first(agent, "submissionId", "submission_id")
                if str(agent_submission) != submission_id:
                    continue
                updated_score = _first(agent, "updatedScore", "updated_score")
                initial_score = _first(agent, "initialScore", "initial_score")
                if updated_score is None:
                    continue
                records.append({
                    "timestamp": timestamp,
                    "submission_id": submission_id,
                    "episode_id": str(episode_id),
                    "initial_score": initial_score,
                    "score": updated_score,
                })

    if not records:
        raise RuntimeError(
            "Kaggle returned episodes, but no updatedScore values. "
            "Do not substitute win/loss rewards: they are not leaderboard ratings."
        )
    frame = pd.DataFrame(records)
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
        fetch_episode_rating_history(submission_ids), deadline, submission_ids
    )
    frame.to_csv(args.output_dir / "lb_trajectory_after_deadline.csv", index=False)
    plot(frame, deadline, args.output_dir / "lb_trajectory_after_deadline.png")
    print(f"team={TEAM_NAME} team_id={team_id} rows={len(frame)}")


if __name__ == "__main__":
    main()
