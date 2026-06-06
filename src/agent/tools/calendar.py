from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import groupby
from zoneinfo import ZoneInfo

import dateparser
import httpx
from livekit.agents.llm import function_tool

from agent.monitoring import MetricsCollector
from utils.config import settings

logger = logging.getLogger(__name__)

# Period-of-day keyword → inclusive local-hour range, for hints like "Thursday afternoon".
_PERIODS: dict[str, tuple[int, int]] = {
    "morning": (0, 11),
    "afternoon": (12, 16),
    "evening": (17, 23),
    "night": (17, 23),
}

# Spoken email dictation: "john at gmail dot com" → "john@gmail.com". Phone STT rarely emits
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Spoken tokens → symbols. Sarvam STT (Indian English) commonly renders "@" as "at the rate"
# and "." as "dot", and spells the username letter by letter. We rebuild the address from
_SPOKEN_EMAIL_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bat\s+the\s+rate(?:\s+of)?\b"), " @ "),
    (re.compile(r"\bat\s+(?:sign|symbol)\b"), " @ "),
    (re.compile(r"\bat\b"), " @ "),
    (re.compile(r"\b(?:dot|dawt|period|point|full\s+stop)\b"), " . "),
    (re.compile(r"\b(?:underscore|under\s+score)\b"), " _ "),
    (re.compile(r"\b(?:hyphen|dash)\b"), " - "),
    (re.compile(r"\bplus\b"), " + "),
)
# Provider domains STT frequently splits into two words.
_DOMAIN_FIXES: tuple[tuple[str, str], ...] = (
    ("g mail", "gmail"), ("hot mail", "hotmail"), ("out look", "outlook"),
    ("i cloud", "icloud"), ("proton mail", "protonmail"), ("y mail", "ymail"),
)

# Hint cleaning for date parsing
_PERIOD_WORDS_RE = re.compile(r"\b(?:morning|afternoon|evening|night)\b")
_FILLER_RE = re.compile(
    r"\b(?:any\s?time|some\s?time|any\s?day|sometimes|please|maybe|just|around)\b"
)
# (Deliberately NOT stripping "at" — it's load-bearing in slot strings like "...at 2 PM".)
_LEADING_QUALIFIER_RE = re.compile(r"^(?:next|this|coming|on|the)\s+")


class CalcomError(Exception):
    """Raised for any non-success Cal.com API response. Carries a clean message for logs;
    tools translate it into a spoken apology — the raw text never reaches the caller."""


# HTTP client (Cal.com API v2)

class CalcomClient:
    """Thin async httpx wrapper over the three Cal.com v2 endpoints. One instance per call."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            if not settings.CALCOM_API_KEY:
                raise CalcomError("CALCOM_API_KEY is not configured")
            self._client = httpx.AsyncClient(
                base_url=settings.CALCOM_API_BASE,
                headers={"Authorization": f"Bearer {settings.CALCOM_API_KEY}"},
                # Connect budget kept tight so a network stall can't blow the voice latency
                # target — a timeout surfaces as a spoken "try again" rather than dead air.
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, *, version: str, params: dict) -> dict:
        resp = await self._http().get(path, params=params, headers={"cal-api-version": version})
        return _unwrap(resp)

    async def _post(self, path: str, *, version: str, json: dict) -> dict:
        resp = await self._http().post(path, json=json, headers={"cal-api-version": version})
        return _unwrap(resp)

    async def list_event_types(self) -> list[dict]:
        """GET /v2/event-types?username=<user> → list of the user's event types."""
        if not settings.CALCOM_USERNAME:
            raise CalcomError("CALCOM_USERNAME is not configured")
        data = await self._get(
            "/event-types",
            version=settings.CALCOM_EVENT_TYPES_API_VERSION,
            params={"username": settings.CALCOM_USERNAME},
        )
        return data.get("data", []) if isinstance(data, dict) else []

    async def resolve_event_type_id(self) -> int:
        """Map the configured CALCOM_EVENT_SLUG to its numeric event-type id"""
        slug = settings.CALCOM_EVENT_SLUG
        for et in await self.list_event_types():
            if et.get("slug") == slug:
                return int(et["id"])
        raise CalcomError(f"no event type with slug {slug!r} for user {settings.CALCOM_USERNAME!r}")

    async def get_slots(self, event_type_id: int, start_date: str, end_date: str) -> dict:
        """GET /v2/slots → raw payload. start_date/end_date are YYYY-MM-DD in the event tz"""
        return await self._get(
            "/slots",
            version=settings.CALCOM_SLOTS_API_VERSION,
            params={
                "eventTypeId": event_type_id,
                "start": start_date,
                "end": end_date,
                "timeZone": settings.CALCOM_TIMEZONE,
            },
        )

    async def create_booking(
        self, event_type_id: int, start_utc_iso: str, name: str, email: str
    ) -> dict:
        """POST /v2/bookings → raw payload."""
        return await self._post(
            "/bookings",
            version=settings.CALCOM_BOOKINGS_API_VERSION,
            json={
                "start": start_utc_iso,
                "eventTypeId": event_type_id,
                "attendee": {
                    "name": name,
                    "email": email,
                    "timeZone": settings.CALCOM_TIMEZONE,
                    "language": settings.CALCOM_BOOKING_LANGUAGE,
                },
            },
        )


def _unwrap(resp: httpx.Response) -> dict:
    """Validate an HTTP response and return its JSON body, or raise CalcomError."""
    try:
        body = resp.json()
    except ValueError:
        body = None

    if resp.status_code // 100 != 2:
        msg = _error_message(body) or f"HTTP {resp.status_code}"
        raise CalcomError(f"{resp.request.method} {resp.request.url.path} failed: {msg}")
    if isinstance(body, dict) and body.get("status") == "error":
        raise CalcomError(_error_message(body) or "Cal.com returned an error")
    if not isinstance(body, dict):
        raise CalcomError("unexpected (non-JSON) Cal.com response")
    return body


def _error_message(body: object) -> str | None:
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict):
        return err.get("message") or err.get("error")
    if isinstance(err, str):
        return err
    return body.get("message")

# Tool set

@dataclass
class _Window:
    """A resolved search window derived from the caller's date hint."""
    start_date: str          # YYYY-MM-DD, event tz
    end_date: str            # YYYY-MM-DD, exclusive upper bound for the /slots query
    single_day: str | None   # YYYY-MM-DD to restrict results to, or None for a multi-day span
    period: tuple[int, int] | None  # inclusive local-hour filter, or None


class CalendarTools:
    """Holds the Cal.com client + per-call state and exposes the three booking tools.

    State that must survive across turns within one call:
      - `_event_type_id`: resolved once (prewarmed during the greeting), then cached.
      - `_last_offered`:  the exact slot datetimes most recently read to the caller, so
                          book_slot can resolve a spoken "Thursday at 2 PM" to an ISO start.
    """

    def __init__(self, collector: MetricsCollector | None = None) -> None:
        self._client = CalcomClient()
        self._collector = collector
        self._tz = ZoneInfo(settings.CALCOM_TIMEZONE)
        self._event_type_id: int | None = None
        self._last_offered: list[datetime] = []

    # lifecycle 

    def function_tools(self) -> list:
        """The bound FunctionTools to hand to the Agent via `tools=[...]`."""
        return [self.get_available_slots, self.collect_contact_info, self.book_slot]

    async def prewarm(self) -> None:
        """Resolve + cache the event-type id and warm the TLS connection up front. Called
        during the greeting so the first booking turn pays neither the slug→id lookup nor the
        handshake — keeping get_available_slots under its 1.5s budget. Best-effort."""
        try:
            await self._get_event_type_id()
        except Exception as exc:
            logger.debug("calendar prewarm skipped: %s", exc)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get_event_type_id(self) -> int:
        if self._event_type_id is None:
            self._event_type_id = await self._client.resolve_event_type_id()
        return self._event_type_id

    # tools

    @function_tool
    async def get_available_slots(self, date_hint: str) -> str:
        """Look up real openings on Ayush's calendar and read back a few to the caller.

        Call this first whenever the caller wants to schedule, book, or set up a
        meeting/interview. Pass the caller's own words for the day; don't invent a date.

        Args:
            date_hint: The day the caller mentioned, in natural language — e.g. "tomorrow",
                "next Tuesday", "this week", or "Thursday afternoon".
        """
        started = time.perf_counter()
        try:
            window = self._parse_window(date_hint)
            if window is None:
                return (
                    "I didn't catch which day you meant. Could you give me a day, "
                    "like tomorrow or next Tuesday?"
                )

            event_type_id = await self._get_event_type_id()
            raw = await self._client.get_slots(event_type_id, window.start_date, window.end_date)
            slots = self._extract_slots(raw, window)

            if not slots:
                self._last_offered = []
                return "I don't have availability that day — would another day work?"

            chosen = _spread_slots(slots, settings.CALCOM_MAX_SLOTS_OFFERED)
            self._last_offered = chosen
            return _format_offer(chosen)
        except CalcomError as exc:
            logger.warning("get_available_slots failed: %s", exc)
            return "I'm having trouble reaching the calendar right now — could you try again in a moment?"
        except Exception as exc:  # noqa: BLE001 — no stack trace ever reaches the caller
            logger.exception("get_available_slots unexpected error: %s", exc)
            return "Something went wrong checking the calendar — could you try again in a moment?"
        finally:
            self._record_latency("get_available_slots", time.perf_counter() - started)

    @function_tool
    async def collect_contact_info(self) -> str:
        """Ask the caller for the name and email needed to send the calendar invite.

        Use this once the caller has agreed to a specific slot but you don't yet have their
        name and email. It only returns the question to ask — it does not book anything.
        """
        return (
            "Could I get your name, and your email address? "
            "If you don't mind, spell out the part before the at-sign so I get it exactly right."
        )

    @function_tool
    async def book_slot(self, datetime_str: str, caller_name: str, caller_email: str) -> str:
        """Confirm a booking on Ayush's calendar after the caller has picked a slot.

        Only call this once you have (1) a specific slot the caller agreed to, taken from a
        previous get_available_slots offer, and (2) their name and email. Confirm the slot
        verbally before calling this.

        Args:
            datetime_str: The agreed slot, as offered earlier — e.g. "Thursday June 12 at 2 PM".
            caller_name: The caller's full name for the invite.
            caller_email: The caller's email address for the invite.
        """
        started = time.perf_counter()
        try:
            email = _normalize_email(caller_email)
            if not email:
                return (
                    "I didn't catch a valid email. Could you spell it out for me, "
                    "like name at gmail dot com?"
                )
            if not (caller_name or "").strip():
                return "Could I get your name as well, for the calendar invite?"

            slot = await self._resolve_requested_slot(datetime_str)
            if slot is None:
                return (
                    "I couldn't match that to an open slot. Could you tell me the day again "
                    "and I'll check what's free?"
                )

            event_type_id = await self._get_event_type_id()
            start_utc = slot.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            await self._client.create_booking(event_type_id, start_utc, caller_name.strip(), email)

            return (
                f"Done — you're booked for {_speak_datetime(slot)} {slot.strftime('%Z')}. "
                "You'll get a confirmation email shortly."
            )
        except CalcomError as exc:
            logger.warning("book_slot failed: %s", exc)
            return (
                "I couldn't lock in that booking just now — the slot may have just been taken. "
                "Want me to check the calendar again?"
            )
        except Exception as exc:  # never surface a stack trace to the caller
            logger.exception("book_slot unexpected error: %s", exc)
            return "Something went wrong while booking — want me to try checking the calendar again?"
        finally:
            self._record_latency("book_slot", time.perf_counter() - started)

    # internals

    def _record_latency(self, name: str, elapsed: float) -> None:
        if self._collector is not None:
            self._collector.record_tool_call(name=name, latency_s=elapsed)
        logger.info("tool %s | %.0fms", name, elapsed * 1000)

    def _parse_date(self, hint: str, now: datetime) -> datetime | None:
        """Resolve a hint to a datetime, working around dateparser 1.x gaps.

        dateparser handles "tomorrow"/"Thursday"/"Friday 3pm" but returns None for two common
        phrasings the brief calls out: "<weekday> afternoon" (a period word it can't combine)
        and "next <weekday>". We strip period + filler words first (the period is handled
        separately as an hour filter), then, if it still won't parse, retry without a leading
        qualifier like "next"/"this" — which recovers "next Tuesday" → the upcoming Tuesday.
        """
        cleaned = _clean_hint(hint)
        parsed = _dateparse(cleaned, now)
        if parsed is None:
            stripped = _LEADING_QUALIFIER_RE.sub("", cleaned).strip()
            if stripped and stripped != cleaned:
                parsed = _dateparse(stripped, now)
        return parsed

    def _parse_window(self, date_hint: str) -> _Window | None:
        """Turn a natural-language hint into a concrete YYYY-MM-DD search window.

        Single-day hints ("tomorrow", "next Tuesday") search just that day. Hints that span a
        range ("this week") widen to CALCOM_SLOT_SEARCH_DAYS. A period word ("afternoon")
        attaches an hour filter applied to the returned slots. Returns None if dateparser can't
        make sense of the hint (the tool then asks the caller to be more specific).
        """
        now = datetime.now(self._tz)
        hint_l = date_hint.lower()
        # Detect period/range intent from the ORIGINAL hint before cleaning strips those words.
        period = next((rng for word, rng in _PERIODS.items() if word in hint_l), None)
        spans_week = any(w in hint_l for w in ("week", "few days", "couple", "any day", "anytime"))

        base = self._parse_date(date_hint, now)
        if base is None:
            return None

        target_day = base.astimezone(self._tz).date()
        start_day = max(now.date(), target_day)
        if spans_week:
            end_day = start_day + timedelta(days=settings.CALCOM_SLOT_SEARCH_DAYS)
            single = None
        else:
            # Query one extra day so an exclusive upper bound can't drop the target date,
            # then restrict results back to that single day.
            end_day = start_day + timedelta(days=1)
            single = start_day.isoformat()

        return _Window(
            start_date=start_day.isoformat(),
            end_date=end_day.isoformat(),
            single_day=single,
            period=period,
        )

    def _extract_slots(self, raw: dict, window: _Window) -> list[datetime]:
        """Flatten the {date: [{start}]} payload into sorted, tz-aware future datetimes,
        applying the single-day and period-of-day filters from the window."""
        data = raw.get("data", {})
        if not isinstance(data, dict):
            return []
        now = datetime.now(self._tz)

        out: list[datetime] = []
        for day, entries in data.items():
            if window.single_day and day != window.single_day:
                continue
            for entry in entries or []:
                iso = entry.get("start") if isinstance(entry, dict) else None
                if not iso:
                    continue
                try:
                    dt = datetime.fromisoformat(iso).astimezone(self._tz)
                except ValueError:
                    continue
                if dt <= now:
                    continue
                if window.period and not (window.period[0] <= dt.hour <= window.period[1]):
                    continue
                out.append(dt)

        # If a period filter wiped out everything but the day had openings, fall back to the
        # day's slots rather than stranding the caller with "no availability".
        if not out and window.period:
            relaxed = _Window(window.start_date, window.end_date, window.single_day, None)
            return self._extract_slots(raw, relaxed)

        return sorted(out)

    async def _resolve_requested_slot(self, datetime_str: str) -> datetime | None:
        """Match the caller's spoken time to an offered slot, re-fetching if needed to confirm it's still open."""
        target = self._parse_date(datetime_str, datetime.now(self._tz))
        if target is None:
            return None
        target = target.astimezone(self._tz)

        match = _match_slot(target, self._last_offered)
        if match is not None:
            return match

        # Not in the offered set — re-fetch the target day's live slots and match exactly.
        event_type_id = await self._get_event_type_id()
        day = target.date()
        raw = await self._client.get_slots(
            event_type_id, day.isoformat(), (day + timedelta(days=1)).isoformat()
        )
        live = self._extract_slots(raw, _Window(day.isoformat(), "", day.isoformat(), None))
        return _match_slot(target, live)

# Pure helpers (no I/O — easy to reason about / unit test)


def _clean_hint(hint: str) -> str:
    """Lowercase the hint and drop period + filler words so dateparser sees a bare date
    expression. Falls back to 'today' if nothing meaningful is left (e.g. hint was only
    'anytime'), which the week/period flags then widen appropriately."""
    s = _PERIOD_WORDS_RE.sub(" ", hint.lower())
    s = _FILLER_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or "today"


def _dateparse(text: str, now: datetime) -> datetime | None:
    """dateparser with the calendar's tz and future-leaning settings, relative to `now`."""
    return dateparser.parse(
        text,
        settings={
            "TIMEZONE": settings.CALCOM_TIMEZONE,
            "TO_TIMEZONE": settings.CALCOM_TIMEZONE,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": now.replace(tzinfo=None),
        },
    )


def _match_slot(target: datetime, candidates: list[datetime]) -> datetime | None:
    """Return the candidate whose wall-clock time equals `target` (to the minute).

    The caller's restated time is matched against known-real slots, so we never book an
    arbitrary time the LLM hallucinated — only a slot the calendar actually offered.
    """
    for c in candidates:
        if (c.year, c.month, c.day, c.hour, c.minute) == (
            target.year, target.month, target.day, target.hour, target.minute
        ):
            return c
    return None


def _spread_slots(slots: list[datetime], limit: int) -> list[datetime]:
    """Pick up to `limit` slots spaced out for natural speech — first of each day, then any
    slot at least an hour after the last kept one. Turns a wall of 9:00/9:30/10:00 openings
    into "9 AM, 10 AM, 11 AM" so the caller isn't read a dense list."""
    chosen: list[datetime] = []
    for dt in slots:
        if not chosen:
            chosen.append(dt)
        elif dt.date() != chosen[-1].date() or dt - chosen[-1] >= timedelta(minutes=60):
            chosen.append(dt)
        if len(chosen) >= limit:
            break
    return chosen


def _speak_time(dt: datetime) -> str:
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{hour}:{dt.minute:02d} {ampm}" if dt.minute else f"{hour} {ampm}"


def _speak_datetime(dt: datetime) -> str:
    return f"{dt:%A %B} {dt.day} at {_speak_time(dt)}"


def _format_offer(slots: list[datetime]) -> str:
    """Render chosen slots as one spoken sentence, grouped by day:
    'I have Thursday June 12 at 2 PM, 3 PM, and Friday June 13 at 10 AM. Which works for you?'

    Times within a day are comma-separated; the day-parts join with an Oxford 'and' so the
    final separator never collides with an in-day comma list ("...3 PM, and Friday...")."""
    parts: list[str] = []
    for _, group in groupby(slots, key=lambda d: d.date()):
        times = list(group)
        label = f"{times[0]:%A %B} {times[0].day}"
        parts.append(f"{label} at {', '.join(_speak_time(t) for t in times)}")
    return f"I have {_oxford(parts)}. Which works for you?"


def _oxford(items: list[str]) -> str:
    """Join with a trailing Oxford 'and': ['a'] → 'a'; ['a','b'] → 'a, and b';
    ['a','b','c'] → 'a, b, and c'."""
    if len(items) <= 1:
        return items[0] if items else ""
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _normalize_email(raw: str) -> str | None:
    """Reconstruct an email dictated over the phone and validate it.

    Phone STT rarely emits literal @/. symbols and tends to spell the username letter by
    letter — and in Indian English "@" is often spoken as "at the rate". We rejoin split
    provider names, map spoken tokens back to symbols, drop spaces (which also collapses
    spelled-out letters: "j o h n" → "john"), tidy stray double dots, and validate. Returns
    the cleaned address, or None if it still doesn't look like an email — in which case the
    tool asks the caller to spell it out and the agent reads it back to confirm before booking.
    """
    if not raw:
        return None
    s = raw.lower().strip()
    for spoken, joined in _DOMAIN_FIXES:
        s = s.replace(spoken, joined)
    for pattern, repl in _SPOKEN_EMAIL_SUBS:
        s = pattern.sub(repl, s)
    s = re.sub(r"\s+", "", s)                  # collapse spaces, incl. spelled-out letters
    s = re.sub(r"\.{2,}", ".", s).strip(".")   # tidy double dots dictation can introduce
    return s if _EMAIL_RE.match(s) else None
