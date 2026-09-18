"""Synthetic call recordings with exact speaker ground truth (macOS only).

Real calls with a known speaker timeline are the evaluation set worth having,
and they are private. This builds a stand-in: conversations spoken by distinct
macOS `say` voices, each on its own simulated line, mixed and pushed through
Opus at a call bitrate, with the reference turns written alongside.

What it imitates, because it is what makes Auto over-count on Zoom audio:

- the same person varying: every utterance gets its own speaking rate and gain,
  and an "unstable" speaker sometimes switches line profile mid-call, the way a
  headset drops out or a codec adapts;
- short back-channels ("Right.", "Mm-hm.") between long turns;
- different lines per speaker (band limits, room echo, a phone-like one), and
  the whole mix through Opus at 20 kbps.

What it cannot imitate is a human voice: TTS voices are more self-consistent
than people are, so treat results as a lower bound on how hard real calls are.

    uv run python scripts/synth_calls.py tmp/eval/synthetic

Writes <name>.wav files and manifest.json:
[{"path": ..., "speakers": N, "turns": [{"start", "end", "speaker"}]}].
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

RATE = 16000

LINES = [
    "Thanks for joining, let's go through the migration plan for the billing service.",
    "We finished moving the invoices table last week, and nothing broke.",
    "What about the payment webhooks? Those worried me the most.",
    "They're next. I want to run both systems in parallel for a few days first.",
    "How long before we can switch the traffic over?",
    "If the parallel run is clean, probably next Thursday.",
    "I'll let the support team know, in case customers see anything odd.",
    "The dashboard shows a small spike in latency after the last deploy.",
    "That's the cache warming up. It settles after about ten minutes.",
    "Can we add an alert for it anyway? I'd rather know than guess.",
    "Sure, I'll set a threshold at twice the normal response time.",
    "Let's move on to the hiring update for the platform team.",
    "We have two final interviews this week and one offer out.",
    "The candidate from Tuesday asked about remote work, so we should decide that.",
    "I think we can offer three days at home, like everyone else.",
    "Budget is the other topic. We're about eight percent over for the quarter.",
    "Most of that is the new database cluster, which we planned for.",
    "Then it's fine, as long as finance knows it was expected.",
    "I'll write it up in the monthly report and flag it clearly.",
    "One more thing: the release notes for version two point four need a review.",
    "I can take that, if someone sends me the draft by Wednesday.",
    "Great. Any other questions before we wrap up?",
    "Nothing from me. Thanks, everyone, see you tomorrow.",
    "Can you share your screen? I can't see the chart from here.",
    "Give me a second, my connection keeps dropping today.",
    "You're back now. We can hear you fine.",
    "I disagree slightly: the rollback plan isn't tested yet.",
    "Fair point. Let's schedule a rollback drill before the switch.",
    "Who owns the on-call rotation next week?",
    "That's me, and I've already swapped the Friday shift.",
]
BACKCHANNELS = ["Right.", "Mm-hm.", "Yeah, exactly.", "Okay.", "Sure.", "Got it."]

# One "line" per speaker: how their audio reaches the call.
PROFILES = [
    "highpass=f=100,lowpass=f=7000",
    "highpass=f=150,lowpass=f=5500,volume=1.3",
    "highpass=f=90,lowpass=f=7500,aecho=0.8:0.6:35:0.3",
    "highpass=f=250,lowpass=f=3800,volume=0.8",
    "highpass=f=120,lowpass=f=6000,aecho=0.8:0.5:20:0.2,volume=1.1",
]

# (name, voices, unstable speakers). Similar pairs on purpose: two US female
# voices, two UK male ones.
CALLS = [
    ("2spk-distinct", ["Daniel", "Samantha"], set()),
    ("2spk-similar-female", ["Samantha", "Kathy"], set()),
    ("2spk-similar-male", ["Daniel", "Rocko (English (UK))"], set()),
    ("2spk-unstable", ["Karen", "Fred"], {0}),
    ("2spk-both-unstable", ["Moira", "Albert"], {0, 1}),
    ("2spk-backchannel", ["Tessa", "Reed (English (US))"], {1}),
    ("3spk-distinct", ["Daniel", "Samantha", "Rishi"], set()),
    ("3spk-unstable", ["Karen", "Fred", "Tara"], {2}),
    ("3spk-similar", ["Samantha", "Kathy", "Daniel"], set()),
    ("4spk-distinct", ["Daniel", "Samantha", "Rishi", "Moira"], set()),
    ("4spk-unstable", ["Karen", "Fred", "Tara", "Albert"], {0, 3}),
    ("4spk-similar", ["Samantha", "Kathy", "Shelley (English (US))", "Daniel"], set()),
]


def run(*argv: str) -> None:
    subprocess.run(argv, check=True, capture_output=True)  # noqa: S603


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.0


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def utterance(tmp: Path, voice: str, text: str, rate: int, chain: str) -> np.ndarray:
    raw, shaped = tmp / "u.aiff", tmp / "u.wav"
    run("say", "-v", voice, "-r", str(rate), "-o", str(raw), text)
    run(
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(raw),
        "-af", chain, "-ar", str(RATE), "-ac", "1", str(shaped),
    )  # fmt: skip
    return read_wav(shaped)


def build(name: str, voices: list[str], unstable: set[int], out: Path, seed: int):
    rng = random.Random(seed)  # noqa: S311 - reproducible test data, not secrets
    lines = LINES[:]
    rng.shuffle(lines)
    pieces: list[np.ndarray] = []
    turns: list[dict] = []
    clock = 0.5
    pieces.append(np.zeros(int(clock * RATE)))
    speaker = 0
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for i in range(34):
            # Mostly hand the floor on; sometimes the same person continues.
            if i and rng.random() < 0.8:
                speaker = rng.choice([s for s in range(len(voices)) if s != speaker])
            back = i and rng.random() < 0.25
            text = rng.choice(BACKCHANNELS) if back else lines[i % len(lines)]
            profile = PROFILES[speaker % len(PROFILES)]
            if speaker in unstable and rng.random() < 0.35:
                profile = PROFILES[(speaker + 2) % len(PROFILES)]
            gain = 10 ** (rng.uniform(-4, 3) / 20)
            chain = f"{profile},volume={gain:.3f}"
            audio = utterance(tmp, voices[speaker], text, rng.randint(150, 225), chain)
            turns.append(
                {
                    "start": round(clock, 3),
                    "end": round(clock + len(audio) / RATE, 3),
                    "speaker": speaker + 1,
                }
            )
            gap = rng.uniform(0.15, 0.9)
            pieces += [audio, np.zeros(int(gap * RATE))]
            clock += len(audio) / RATE + gap
        mix = np.concatenate(pieces)
        mix += np.random.default_rng(seed).normal(0, 0.003, len(mix))  # line hiss
        clean = tmp / "mix.wav"
        write_wav(clean, mix)
        # The whole call through Opus at a call bitrate, and back.
        run(
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(clean),
            "-c:a", "libopus", "-b:a", "20k", str(tmp / "mix.opus"),
        )  # fmt: skip
        run(
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp / "mix.opus"),
            "-ar", str(RATE), "-ac", "1", str(out / f"{name}.wav"),
        )  # fmt: skip
    return {"path": f"{name}.wav", "speakers": len(voices), "turns": turns}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if sys.platform != "darwin":
        print("needs macOS: the voices come from `say`", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed, (name, voices, unstable) in enumerate(CALLS, 1):
        manifest.append(build(name, voices, unstable, args.out, seed))
        print(f"{name}: {manifest[-1]['turns'][-1]['end']:.0f}s", flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
