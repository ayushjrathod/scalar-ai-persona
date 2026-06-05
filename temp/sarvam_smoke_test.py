#!/usr/bin/env python3
"""
Sarvam AI TTS + STT Latency Diagnostic 

TTS : model: bulbul:v3
STT : model: saarika:v2.5
"""

import argparse
import asyncio
import base64
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI, AudioOutput, SarvamAI

load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY")
# 15-word English sentence
TTS_TEXT = (
    "Your voice assistant is ready to help you schedule meetings, "
    "make calls, and answer questions."
)
AUDIO_FILE = Path("tts_output.mp3")
RUNS = 5
TTFB_PASS_MS = 800
RECV_TIMEOUT_S = 10.0


async def test_tts_ws() -> tuple[float, float, bytes]:
    """Returns (ttfb_ms, total_ms, audio_bytes).

    Pattern taken verbatim from Sarvam docs:
      connect(model=...) → configure(...) → convert(...) → flush() → recv() loop
    Clock starts after configure (fixed overhead) — immediately before convert+flush.
    """
    client = AsyncSarvamAI(api_subscription_key=API_KEY)

    async with client.text_to_speech_streaming.connect(
        model="bulbul:v3",
        send_completion_event=True,
    ) as ws:
        await ws.configure(
            target_language_code="en-IN",
            speaker="ritu",
            output_audio_codec="mp3",
            pace=1.0,
        )

        # Clock starts: send text + flush
        t0 = time.perf_counter()
        await ws.convert(TTS_TEXT)
        await ws.flush()

        first_chunk_t: float | None = None
        chunks: list[bytes] = []

        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_S)
            except asyncio.TimeoutError:
                break

            if isinstance(message, AudioOutput):
                if first_chunk_t is None:
                    first_chunk_t = time.perf_counter()
                chunks.append(base64.b64decode(message.data.audio))
            else:
                # final event, error, or unknown — stop receiving
                break

    t_end = time.perf_counter()
    ttfb_ms = (first_chunk_t - t0) * 1000 if first_chunk_t else (t_end - t0) * 1000
    total_ms = (t_end - t0) * 1000
    return ttfb_ms, total_ms, b"".join(chunks)


def test_stt(audio_path: Path) -> tuple[float, str]:
    """Returns (total_ms, transcript) using the batch job SDK.

    Pattern from official Sarvam example:
      create_job → upload_files → start → wait_until_complete → download_outputs
    """
    client = SarvamAI(api_subscription_key=API_KEY)

    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.perf_counter()

        job = client.speech_to_text_job.create_job(
            model="saaras:v3",
            mode="transcribe",
            language_code="en-IN",
        )
        job.upload_files(file_paths=[str(audio_path)])
        job.start()
        job.wait_until_complete()

        total_ms = (time.perf_counter() - t0) * 1000

        results = job.get_file_results()
        if results["failed"]:
            err = results["failed"][0].get("error_message", "unknown error")
            raise RuntimeError(f"Batch STT failed: {err}")

        job.download_outputs(output_dir=tmpdir)

        transcript = ""
        for out in Path(tmpdir).rglob("*"):
            if out.is_file():
                raw = out.read_text(encoding="utf-8")
                try:
                    data = json.loads(raw)
                    transcript = data.get("transcript") or data.get("text") or ""
                except json.JSONDecodeError:
                    transcript = raw.strip()
                break

    return total_ms, transcript


def percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    idx = pct / 100 * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def print_row(label: str, vals: list[float]) -> None:
    if not vals:
        return
    print(
        f"  {label:<30} {min(vals):>7.0f}  "
        f"{statistics.median(vals):>7.0f}  "
        f"{percentile(vals, 95):>7.0f}"
    )


async def run_tts_suite() -> tuple[list[float], list[float], bytes]:
    tts_ttfb: list[float] = []
    tts_total: list[float] = []
    last_audio = b""

    print(f"TTS runs ({RUNS}x)  —  AsyncSarvamAI  bulbul:v3  speaker=ritu")
    for i in range(RUNS):
        try:
            ttfb, total, audio = await test_tts_ws()
            tts_ttfb.append(ttfb)
            tts_total.append(total)
            last_audio = audio
            print(f"  [{i + 1}] TTFB {ttfb:6.0f} ms   total {total:6.0f} ms")
        except Exception as exc:
            print(f"  [{i + 1}] FAILED: {exc}")

    return tts_ttfb, tts_total, last_audio


def run_stt_suite() -> tuple[list[float], str]:
    stt_total: list[float] = []
    last_transcript = ""

    print(f"STT runs ({RUNS}x)  —  Batch job SDK  saaras:v3")
    for i in range(RUNS):
        try:
            total, transcript = test_stt(AUDIO_FILE)
            stt_total.append(total)
            last_transcript = transcript
            print(f"  [{i + 1}] total {total:6.0f} ms   \"{transcript}\"")
        except Exception as exc:
            print(f"  [{i + 1}] FAILED: {exc}")

    return stt_total, last_transcript


async def main() -> None:
    parser = argparse.ArgumentParser(description="Sarvam AI TTS + STT latency diagnostic")
    parser.add_argument("--STT", action="store_true", help="Run STT only (skips TTS, uses existing tts_output.wav)")
    args = parser.parse_args()

    if not API_KEY:
        sys.exit("ERROR: SARVAM_API_KEY not found — add it to .env")

    word_count = len(TTS_TEXT.split())
    print("Sarvam AI Latency Diagnostic")
    print(f"  TTS  : AsyncSarvamAI SDK  connect(model='bulbul:v3')  speaker=ritu  en-IN")
    print(f"  STT  : Batch job SDK  speech_to_text_job  (saaras:v3  en-IN)")
    print(f"  Auth : api_subscription_key (SDK) / api-subscription-key header (STT REST)")
    print(f"  Text ({word_count} words): \"{TTS_TEXT}\"")
    print(f"  TTFB : convert()+flush() sent → first AudioOutput chunk received\n")

    tts_ttfb: list[float] = []
    tts_total: list[float] = []

    if not args.STT:
        tts_ttfb, tts_total, last_audio = await run_tts_suite()

        if not tts_ttfb:
            sys.exit("All TTS runs failed — check API key and network.")

        AUDIO_FILE.write_bytes(last_audio)
        print(f"\nAudio saved → {AUDIO_FILE}  ({len(last_audio):,} bytes)\n")
    else:
        if not AUDIO_FILE.exists():
            sys.exit(f"ERROR: {AUDIO_FILE} not found — run without --STT first to generate it.")
        print(f"Skipping TTS — using existing {AUDIO_FILE}  ({AUDIO_FILE.stat().st_size:,} bytes)\n")

    stt_total, last_transcript = run_stt_suite()

    # ── Summary table ─────────────────────────────────────────────────────────
    SEP = "=" * 64
    print(f"\n{SEP}")
    print(f"  {'METRIC':<30} {'MIN':>7}  {'MEDIAN':>7}  {'P95':>7}")
    print(f"  {'-' * 58}")
    if tts_ttfb:
        print_row("TTS first-byte (ms)", tts_ttfb)
        print_row("TTS total       (ms)", tts_total)
    print_row("STT total       (ms)", stt_total)
    print(SEP)

    # ── Verdict ───────────────────────────────────────────────────────────────
    if tts_ttfb:
        med_ttfb = statistics.median(tts_ttfb)
        verdict = (
            f"PASS (<{TTFB_PASS_MS}ms)"
            if med_ttfb < TTFB_PASS_MS
            else f"FAIL (>{TTFB_PASS_MS}ms)"
        )
        print(f"\nTTS first-byte median: {med_ttfb:.0f}ms — {verdict}")
    if last_transcript:
        print(f"STT last transcript:   \"{last_transcript}\"")


if __name__ == "__main__":
    asyncio.run(main())
