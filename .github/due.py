"""Decide whether a scheduled picks run is due.

GitHub starts scheduled workflows best-effort: runs can start hours late or
never. So the workflow also wakes every hour, and this check says whether
the most recent planned run time (SLOTS) has been covered yet. A slot counts
as covered when the picks branch was published after it, or when another
run of this workflow started after it and is still going. Anything other
than a scheduled event (a manual run) always runs.

SLOTS is a space-separated list of "DOW@HH:MM" in UTC, DOW as in cron
(0 = Sunday). A slot more than WINDOW_HOURS old no longer counts as due.
On any error it says "run": a spare run is cheap, a missed one isn't.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys

WINDOW_HOURS = 10


def parse_slots(text: str) -> list[tuple[int, int, int]]:
    out = []
    for item in text.split():
        dow, hm = item.split("@")
        h, m = hm.split(":")
        out.append((int(dow), int(h), int(m)))
    return out


def last_slot(slots, now: dt.datetime) -> dt.datetime | None:
    """The latest slot time at or before `now`, within the window."""
    best = None
    for back in range(0, 2):
        day = (now - dt.timedelta(days=back)).date()
        cron_dow = (day.weekday() + 1) % 7
        for dow, h, m in slots:
            if dow != cron_dow:
                continue
            t = dt.datetime(day.year, day.month, day.day, h, m,
                            tzinfo=dt.timezone.utc)
            if t <= now and now - t <= dt.timedelta(hours=WINDOW_HOURS):
                best = t if best is None or t > best else best
    return best


def is_due(slot, published, others) -> bool:
    """`published`: last picks-branch publish time (or None); `others`:
    start times of other runs of this workflow still queued or running."""
    if slot is None:
        return False
    if published is not None and published >= slot:
        return False
    return not any(t >= slot for t in others)


def _sh(*cmd) -> str:
    return subprocess.run(cmd, check=True, capture_output=True,
                          text=True).stdout.strip()


def main() -> bool:
    if os.environ.get("GITHUB_EVENT_NAME") != "schedule":
        print("manual run: go")
        return True
    now = dt.datetime.now(dt.timezone.utc)
    slot = last_slot(parse_slots(os.environ["SLOTS"]), now)
    if slot is None:
        print("no planned run in the last", WINDOW_HOURS, "hours: skip")
        return False
    published = None
    try:
        _sh("git", "fetch", "-q", "--depth", "1", "origin", "picks")
        ts = int(_sh("git", "log", "-1", "--format=%ct", "FETCH_HEAD"))
        published = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
    except Exception as exc:  # no picks branch yet
        print("no picks branch:", exc)
    me = os.environ.get("GITHUB_RUN_ID", "")
    others = []
    for status in ("in_progress", "queued"):
        rows = json.loads(_sh("gh", "run", "list", "--workflow", "picks.yml",
                              "--status", status, "--limit", "20", "--json",
                              "databaseId,createdAt"))
        for r in rows:
            if str(r["databaseId"]) != me:
                others.append(dt.datetime.fromisoformat(
                    r["createdAt"].replace("Z", "+00:00")))
    due = is_due(slot, published, others)
    print(f"slot {slot:%a %H:%M}Z, last publish {published}, "
          f"{len(others)} other run(s) going: {'go' if due else 'skip'}")
    return due


if __name__ == "__main__":
    try:
        go = main()
    except Exception as exc:
        print("check failed, running anyway:", exc)
        go = True
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"run={'true' if go else 'false'}\n")
    sys.exit(0)
