import datetime as dt
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "due", pathlib.Path(__file__).resolve().parents[1] / ".github" / "due.py")
due = importlib.util.module_from_spec(spec)
spec.loader.exec_module(due)

U = dt.timezone.utc
SLOTS = due.parse_slots("0@22:00 4@19:00 6@15:00")


def t(*a):
    return dt.datetime(*a, tzinfo=U)


def test_finds_latest_slot_in_window():
    # Thu 2026-11-05 23:40 UTC: the Thursday 19:00 run is the latest
    assert due.last_slot(SLOTS, t(2026, 11, 5, 23, 40)) == t(2026, 11, 5, 19)
    # before 19:00 Thursday nothing is due within the window
    assert due.last_slot(SLOTS, t(2026, 11, 5, 18, 0)) is None
    # Sunday 22:00 still counts just after midnight Monday
    assert due.last_slot(SLOTS, t(2026, 11, 9, 1, 0)) == t(2026, 11, 8, 22)


def test_due_only_when_slot_not_covered():
    slot = t(2026, 11, 5, 19)
    assert due.is_due(slot, t(2026, 11, 4, 0, 27), [])          # missed: run
    assert not due.is_due(slot, t(2026, 11, 5, 19, 5), [])      # published
    assert not due.is_due(slot, None, [t(2026, 11, 5, 19, 2)])  # running
    assert not due.is_due(None, None, [])
