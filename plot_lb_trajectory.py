#!/usr/bin/env python3
"""Plot Seasaw's two latest submission ratings after the final deadline."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


COMPETITION = "pokemon-tcg-ai-battle"
TEAM_NAME = "Seasaw"
EPISODE_FILE = "EpisodeAgents.parquet"
# Kaggle's public competition page specifies 2026-08-16 23:59 UTC.  The
# simulation API currently omits this field for the closed competition.
DEFAULT_FINAL_SUBMISSION_DEADLINE = pd.Timestamp("2026-08-16T23:59:59Z")


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def column(df: pd.DataFrame, *names: str) -> str | None:
    wanted = {norm(name) for name in names}
    for name in df.columns:
        if norm(name) in wanted:
            return name
    return None


def find_team(api: KaggleApi) -> tuple[int, str]:
    # The client returns rows but prints the continuation token. Capture that
    # diagnostic so we can walk every page without leaking it into CI logs.
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
    raise RuntimeError(
        f"{TEAM_NAME!r} was not found in the visible leaderboard. "
        "Use an account that can see the team leaderboard."
    )


def latest_submissions(api: KaggleApi, team_id: int) -> list[tuple[str, object]]:
    rows = api.competition_team_submissions(team_id) or []
    rows = [row for row in rows if getattr(row, "id", None) is not None]
    def submitted_at(row: object) -> float:
        value = getattr(row, "date_submitted", None)
        if value is None:
            return float("-inf")
        timestamp = pd.Timestamp(value)
        return timestamp.timestamp()

    rows.sort(key=submitted_at)
    if len(rows) < 2:
        raise RuntimeError(f"Only {len(rows)} active submission(s) found for {TEAM_NAME}")
    return [(str(row.id), row) for row in rows[-2:]]


def final_deadline(api: KaggleApi) -> pd.Timestamp:
    # kaggle 2.x names this method ``competitions_list`` and returns a
    # response object; older clients used ``competition_list``/a plain list.
    list_method = getattr(api, "competitions_list", None)
    if list_method is not None:
        # ``pokemon-tcg-ai-battle`` is closed; the default competition tab can
        # omit closed competitions, so explicitly request all categories.
        response = list_method(category="all", search=COMPETITION, page_size=100)
        competitions = getattr(response, "competitions", None) or []
    else:
        competitions = api.competition_list(search=COMPETITION, page_size=100) or []
    for competition in competitions:
        if str(getattr(competition, "ref", "")) == COMPETITION:
            # Simulation competitions may expose this under a camelCase or
            # final-deadline field rather than the regular ``deadline``.
            deadline = next(
                (
                    getattr(competition, name, None)
                    for name in (
                        "deadline",
                        "final_submission_deadline",
                        "finalSubmissionDeadline",
                    )
                    if getattr(competition, name, None)
                ),
                None,
            )
            if deadline:
                timestamp = pd.Timestamp(deadline)
                return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    configured = os.environ.get("FINAL_SUBMISSION_DEADLINE")
    if configured:
        timestamp = pd.Timestamp(configured)
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    # Keep the scheduled dashboard running even when Kaggle omits the field.
    # An Actions variable can override this if the organizers revise dates.
    return DEFAULT_FINAL_SUBMISSION_DEADLINE


def download_episode_agents(api: KaggleApi, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    api.competition_download_file(COMPETITION, EPISODE_FILE, path=str(directory), force=False, quiet=True)
    parquet = directory / EPISODE_FILE
    if parquet.exists():
        return parquet
    archives = sorted(directory.glob("*.zip"))
    if archives:
        with zipfile.ZipFile(archives[-1]) as archive:
            archive.extractall(directory)
        if parquet.exists():
            return parquet
    raise FileNotFoundError(f"{EPISODE_FILE} was not found in {directory}")


def load_after_deadline(path: Path, deadline: pd.Timestamp, submission_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    submission_col = column(frame, "submissionId", "submissionRef", "agentSubmissionId")
    time_col = column(frame, "createTime", "episodeCreateTime", "timestamp", "date", "endTime")
    score_col = column(frame, "updatedScore", "score", "ratingAfter", "eloAfter", "publicScore")
    if not submission_col or not time_col or not score_col:
        raise RuntimeError(
            "Could not infer EpisodeAgents.parquet columns. "
            f"Columns: {', '.join(map(str, frame.columns))}"
        )

    frame = frame[frame[submission_col].astype(str).isin(submission_ids)].copy()
    frame["timestamp"] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    frame["score"] = pd.to_numeric(frame[score_col], errors="coerce")
    frame = frame[(frame["timestamp"] >= deadline) & frame["score"].notna()].copy()
    frame["submission_id"] = frame[submission_col].astype(str)
    episode_col = column(frame, "episodeId", "episodeRef", "id")
    if episode_col:
        frame["episode_id"] = frame[episode_col].astype(str)
        frame = frame.drop_duplicates(["submission_id", "episode_id"], keep="last")
    return frame[["timestamp", "score", "submission_id"]].sort_values("timestamp")


def plot(frame: pd.DataFrame, deadline: pd.Timestamp, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for submission_id, rows in frame.groupby("submission_id", sort=False):
        axis.plot(rows["timestamp"], rows["score"], marker=".", linewidth=1.5, label=f"Submission {submission_id}")
    axis.axvline(deadline, color="black", linestyle="--", linewidth=1, label="Final Submission Deadline")
    axis.set_title("Seasaw — LB trajectory after Final Submission Deadline")
    axis.set_xlabel("Episode time (UTC)")
    axis.set_ylabel("EpisodeAgents score / rating")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("lb_trajectory"))
    parser.add_argument("--parquet", type=Path, help="Use an existing EpisodeAgents.parquet")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    team_id, _ = find_team(api)
    submissions = latest_submissions(api, team_id)
    submission_ids = {submission_id for submission_id, _ in submissions}
    deadline = final_deadline(api)
    parquet = args.parquet or download_episode_agents(api, args.output_dir / "data")
    frame = load_after_deadline(parquet, deadline, submission_ids)
    if frame.empty:
        raise RuntimeError("No post-deadline EpisodeAgents rows matched the two latest submissions")
    frame.to_csv(args.output_dir / "lb_trajectory_after_deadline.csv", index=False)
    plot(frame, deadline, args.output_dir / "lb_trajectory_after_deadline.png")
    print(f"team={TEAM_NAME} team_id={team_id}")
    print(f"submissions={', '.join(sorted(submission_ids))}")
    print(f"deadline={deadline.isoformat()}")
    print(f"rows={len(frame)}")
    print(f"png={args.output_dir / 'lb_trajectory_after_deadline.png'}")


if __name__ == "__main__":
    main()
