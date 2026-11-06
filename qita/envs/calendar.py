"""Calendar server, English tool names as in common calendar MCP servers."""

from __future__ import annotations

import datetime as dt

from .base import Env, ToolError

SEED = [
    {"id": "ev1", "title": "Dentista", "start": "2026-11-04T15:00", "end": "2026-11-04T16:00", "location": "via XX Settembre 40"},
    {"id": "ev2", "title": "Palestra", "start": "2026-11-05T18:30", "end": "2026-11-05T19:30", "location": None},
    {"id": "ev3", "title": "Call con Marco: server di prova", "start": "2026-11-06T09:30", "end": "2026-11-06T10:30", "location": None},
    {"id": "ev4", "title": "Mercatino Il Faro", "start": "2026-11-07T10:00", "end": "2026-11-07T12:00", "location": "piazza Matteotti"},
    {"id": "ev5", "title": "Riunione di progetto sito", "start": "2026-11-10T15:00", "end": "2026-11-10T16:30", "location": "sede Il Faro"},
    {"id": "ev6", "title": "Compleanno Sara", "start": "2026-11-12T20:00", "end": "2026-11-12T23:00", "location": "Trattoria da Maria"},
]


def parse_dt(value: str, field: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError as e:
        raise ToolError(f"Invalid {field}: '{value}'. Use ISO 8601, e.g. 2026-11-05T14:00") from e


def fmt(e: dict) -> str:
    loc = f" @ {e['location']}" if e.get("location") else ""
    return f"{e['id']}: {e['title']} | {e['start']} -> {e['end']}{loc}"


class CalendarEnv(Env):
    name = "calendar"
    title = "calendar"

    def build(self) -> None:
        seed = SEED if self.ws.world.events is None else self.ws.world.events
        self.events = [dict(e) for e in seed]
        self.next_id = len(seed) + 1
        s = self.server

        def get(eid: str) -> dict:
            for e in self.events:
                if e["id"] == eid:
                    return e
            raise ToolError(f"Event not found: {eid}")

        @s.tool()
        def list_events(start_date: str, end_date: str) -> str:
            """List calendar events between two dates (inclusive), ISO format YYYY-MM-DD."""
            a = parse_dt(start_date, "start_date").date()
            b = parse_dt(end_date, "end_date").date()
            if b < a:
                raise ToolError("end_date is before start_date")
            hits = [fmt(e) for e in sorted(self.events, key=lambda e: e["start"])
                    if a <= dt.datetime.fromisoformat(e["start"]).date() <= b]
            return "\n".join(hits) or "No events in this range"

        @s.tool()
        def create_event(title: str, start: str, end: str, location: str | None = None) -> str:
            """Create a calendar event. start and end are ISO datetimes, e.g. 2026-11-05T14:00."""
            a, b = parse_dt(start, "start"), parse_dt(end, "end")
            if b <= a:
                raise ToolError("end must be after start")
            e = {"id": f"ev{self.next_id}", "title": title, "start": a.isoformat(timespec="minutes"),
                 "end": b.isoformat(timespec="minutes"), "location": location}
            self.next_id += 1
            self.events.append(e)
            return f"Created {fmt(e)}"

        @s.tool()
        def update_event(event_id: str, title: str | None = None, start: str | None = None,
                         end: str | None = None, location: str | None = None) -> str:
            """Update fields of an existing event. Only the given fields change."""
            e = get(event_id)
            if title is not None:
                e["title"] = title
            if start is not None:
                e["start"] = parse_dt(start, "start").isoformat(timespec="minutes")
            if end is not None:
                e["end"] = parse_dt(end, "end").isoformat(timespec="minutes")
            if location is not None:
                e["location"] = location
            return f"Updated {fmt(e)}"

        @s.tool()
        def delete_event(event_id: str) -> str:
            """Delete a calendar event permanently."""
            e = get(event_id)
            self.events.remove(e)
            return f"Deleted {event_id}"

    def check(self, kind: str, spec: dict) -> bool | None:
        if kind == "event_exists":
            t = spec.get("title_contains", "").casefold()
            return any(
                t in e["title"].casefold()
                and e["start"].startswith(spec.get("start", ""))
                and e["end"].startswith(spec.get("end", ""))
                for e in self.events
            )
        if kind == "event_absent":
            return all(e["id"] != spec["id"] for e in self.events)
        if kind == "event_count":
            return len(self.events) == spec["n"]
        return None
