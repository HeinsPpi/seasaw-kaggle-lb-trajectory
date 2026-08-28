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


def download_episode_agents(api: KaggleApi, directory: Path, submission_ids: set[str] | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    try:
        api.competition_download_file(COMPETITION, EPISODE_FILE, path=str(directory), force=False, quiet=True)
    except Exception as error:
        if submission_ids is None:
            raise
        # Official daily simulation datasets publish rating snapshots in a
        # small manifest.csv. Prefer these over per-episode +/-1 rewards.
        manifests = []
        for dataset in api.dataset_list(search="pokemon-tcg-ai-battle-episodes", page=1) or []:
            ref = str(getattr(dataset, "ref", ""))
            if not ref or ref.endswith("-index"):
                continue
            match = re.search(r"(2026-\d{2}-\d{2})$", ref)
            if not match:
                continue
            dataset_dir = directory / match.group(1)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            target = dataset_dir / "manifest.csv"
            try:
                api.dataset_download_file(ref, "manifest.csv", path=str(dataset_dir), force=False, quiet=True)
                archives = sorted(dataset_dir.glob("manifest.csv.zip"))
                if archives:
                    with zipfile.ZipFile(archives[-1]) as archive:
                        archive.extract("manifest.csv", dataset_dir)
                if target.exists():
                    manifests.append(target)
            except Exception:
                continue
        rating_rows = []
        for manifest in manifests:
            table = pd.read_csv(manifest)
            norm_cols = {c: norm(c) for c in table.columns}
            time_col = next((c for c, n in norm_cols.items() if n in {"createtime", "timestamp", "date"}), None)
            if not time_col:
                time_col = table.columns[0]
            for _, record in table.iterrows():
                timestamp = record[time_col]
                for col, name in norm_cols.items():
                    if "submission" not in name and "agent" not in name:
                        continue
                    sid = str(record[col])
                    if sid not in submission_ids:
                        continue
                    groups = re.findall(r"\d+", name)
                    score_col = next(
                        (
                            c for c, n in norm_cols.items()
                            if ("rating" in n or "score" in n)
                            and (not groups or any(g in n for g in groups))
                        ),
                        None,
                    )
                    if score_col is not None:
                        rating_rows.append({"submissionId": sid, "episodeId": f"{manifest.stem}-{len(rating_rows)}", "createTime": timestamp, "updatedScore": record[score_col]})
        if rating_rows:
            path = directory / EPISODE_FILE
            pd.DataFrame(rating_rows).to_parquet(path, index=False)
            return path
        rows = []
        for submission_id in submission_ids:
            for episode in api.competition_list_episodes(int(submission_id)) or []:
                for agent in getattr(episode, "agents", None) or []:
                    if str(getattr(agent, "submission_id", "")) != str(submission_id):
                        continue
                    rows.append({
                        "submissionId": str(submission_id),
                        "episodeId": str(getattr(episode, "id", "")),
                        "createTime": getattr(episode, "create_time", None),
                        "updatedScore": getattr(agent, "reward", None),
                    })
        if rows:
            path = directory / EPISODE_FILE
            pd.DataFrame(rows).to_parquet(path, index=False)
            return path
        raise error
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
    frame = frame[["timestamp", "score", "submission_id"]].sort_values("timestamp")
    # Legacy episode fallback contains +/-1 rewards; rating manifests contain
    # absolute Elo-like values (hundreds). Preserve the latter as-is.
    if frame["score"].abs().max() <= 1:
        frame["score"] = frame.groupby("submission_id")["score"].transform(
            lambda values: (values.gt(0).cumsum() / values.expanding().count())
        )
    return frame


def plot(frame: pd.DataFrame, deadline: pd.Timestamp, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for submission_id, rows in frame.groupby("submission_id", sort=False):
        axis.plot(rows["timestamp"], rows["score"], marker=".", linewidth=1.5, label=f"Submission {submission_id}")
    axis.axvline(deadline, color="black", linestyle="--", linewidth=1, label="Final Submission Deadline")
    axis.set_title("Seasaw — rating trajectory after Final Submission Deadline")
    axis.set_xlabel("Episode time (UTC)")
    axis.set_ylabel("Rating")
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
