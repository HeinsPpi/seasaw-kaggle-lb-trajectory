#!/usr/bin/env python3
"""Fetch Kaggle data and render the public GitHub Pages dashboard."""

from __future__ import annotations

import html
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from plot_lb_trajectory import (
    COMPETITION, TEAM_NAME, download_episode_agents, final_deadline,
    find_team, latest_submissions, load_after_deadline, plot,
)


def render(site: Path) -> None:
    site.mkdir(parents=True, exist_ok=True)
    (site / ".nojekyll").write_text("", encoding="utf-8")
    api = KaggleApi()
    api.authenticate()
    team_id, _ = find_team(api)
    submissions = latest_submissions(api, team_id)
    submission_ids = {submission_id for submission_id, _ in submissions}
    deadline = final_deadline(api)
    with tempfile.TemporaryDirectory() as temporary:
        parquet = download_episode_agents(api, Path(temporary), submission_ids)
        frame = load_after_deadline(parquet, deadline, submission_ids)
    if frame.empty:
        raise RuntimeError("No post-deadline EpisodeAgents rows matched the latest submissions")

    image_name = "lb_trajectory_after_deadline.png"
    csv_name = "lb_trajectory_after_deadline.csv"
    frame.to_csv(site / csv_name, index=False)
    plot(frame, deadline, site / image_name)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rows = "".join(
        f"<tr><td>{html.escape(submission_id)}</td><td>{len(group)}</td>"
        f"<td>{group['score'].iloc[-1]:.3f}</td><td>{html.escape(group['timestamp'].iloc[-1].isoformat())}</td></tr>"
        for submission_id, group in frame.groupby("submission_id", sort=False)
    )
    links = " / ".join(
        f'<a href="https://www.kaggle.com/competitions/{COMPETITION}/episodes/{html.escape(sid)}">{html.escape(sid)}</a>'
        for sid in sorted(submission_ids)
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="600"><title>Seasaw episode win-rate trajectory</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}}img{{max-width:100%;height:auto;border:1px solid #ddd}}table{{border-collapse:collapse}}th,td{{padding:.45rem .8rem;border-bottom:1px solid #ddd;text-align:left}}small{{color:#666}}</style>
</head><body><h1>Seasaw — cumulative episode win rate</h1>
<p>Competition: <a href="https://www.kaggle.com/competitions/{COMPETITION}">{COMPETITION}</a><br>
Final Submission Deadline: <code>{html.escape(deadline.isoformat())}</code><br>Latest active submissions: {links}</p>
<img src="{image_name}" alt="Cumulative episode win-rate trajectory after Final Submission Deadline">
<h2>Latest values</h2><table><thead><tr><th>Submission</th><th>Episodes</th><th>Cumulative win rate</th><th>Latest episode time</th></tr></thead><tbody>{rows}</tbody></table>
<p><a href="{csv_name}">Download CSV</a></p><small>Updated {generated}. This page refreshes every 10 minutes; the GitHub Actions job also runs every 10 minutes.</small>
</body></html>"""
    (site / "index.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    render(Path("site"))
