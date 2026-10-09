#!/usr/bin/env python3
"""slop_audit.py — flag AI-writing tells in a markdown/text file.

Slop phrase list (517 phrases): sam-paech/antislop-sampler, slop_phrase_prob_adjustments.json
  https://github.com/sam-paech/antislop-sampler  (MIT)
The upstream list was built from over-represented phrases in LLM-generated fiction.
This script adds a small set of tech-register phrases and four structural checks,
because README-style writing fails differently than stories do.

Usage:  python3 slop_audit.py README.md
Exit 0 = clean, exit 1 = flags found.
"""
import json
import math
import re
import sys
from pathlib import Path

LIST_PATH = Path(__file__).parent / "slop_phrase_prob_adjustments.json"

# Tech-register additions (not in the upstream fiction-oriented list).
TECH_SLOP = [
    "delve into", "dive deep", "dive into", "in the realm of", "in the landscape of",
    "it's worth noting", "it is worth noting", "leverage", "leveraging", "seamless",
    "seamlessly", "robust", "cutting-edge", "state-of-the-art", "game-changer",
    "game changer", "paradigm", "unlock", "unleash", "supercharge", "turbocharge",
    "elevate", "empower", "empowering", "revolutionize", "transformative",
    "holistic", "synergy", "myriad", "plethora", "beacon", "pivotal", "crucial role",
    "plays a crucial role", "at the end of the day", "the key takeaway",
    "in conclusion", "furthermore", "moreover", "additionally,", "notably,",
    "significantly,", "underscores", "underscoring", "spearhead", "pave the way",
    "navigate the", "navigating the", "embark", "journey", "landscape of",
]

# Upstream repo also recommends regex bans; these three catch the worst cadences.
REGEX_BANS = [
    (r"\bnot (?:just|only|merely) [^.\n]{1,60}\bbut\b", "not-X-but-Y construction"),
    (r"\b(?:isn't|is not) (?:just|only|merely) [^.\n]{1,60}\bit'?s\b", "isn't-X-it's-Y construction"),
    (r"\bwhether you'?re\b", "whether-you're address"),
]

EM_DASH = re.compile(r"—")


def load_phrases():
    if LIST_PATH.exists():
        data = json.load(open(LIST_PATH))
        return [p for p, _ in data]
    return []


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def audit(path):
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    lower = text.lower()
    words = re.findall(r"\S+", text)
    flags = []

    # 1. phrase hits, one per unique phrase, with first line number
    hits = []
    for phrase in load_phrases() + TECH_SLOP:
        pat = re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b", re.I)
        m = pat.search(text)
        if m:
            line_no = text[: m.start()].count("\n") + 1
            hits.append((phrase, line_no))
    for phrase, line_no in hits:
        flags.append(f"L{line_no}: slop phrase: \"{phrase}\"")

    # 2. regex bans
    for pattern, label in REGEX_BANS:
        for m in re.finditer(pattern, text, re.I):
            line_no = text[: m.start()].count("\n") + 1
            flags.append(f"L{line_no}: {label}: \"{m.group(0)[:70]}\"")

    # 3. em-dash density (flag above 2 per 1000 words)
    n_em = len(EM_DASH.findall(text))
    per_k = 1000.0 * n_em / max(1, len(words))
    if per_k > 2.0:
        flags.append(f"em-dash density {per_k:.1f}/1000 words ({n_em} total) — rewrite with commas, colons or periods")

    # 4. rule-of-three: >=4 triads per 1000 words
    triads = re.findall(r"\b[\w-]+,\s+[\w-]+,\s+and\s+[\w-]+\b", lower)
    per_k = 1000.0 * len(triads) / max(1, len(words))
    if per_k > 2.0:
        flags.append(f"rule-of-three cadence x{len(triads)} ({per_k:.1f}/1000 words) — vary the list lengths")

    # 5. sentence-length uniformity: stdev < 3.5 words across >=12 sentences
    sents = split_sentences(text)
    if len(sents) >= 12:
        lens = [len(s.split()) for s in sents]
        mean = sum(lens) / len(lens)
        stdev = math.sqrt(sum((x - mean) ** 2 for x in lens) / len(lens))
        if stdev < 3.5:
            flags.append(f"sentence lengths too uniform (stdev {stdev:.1f} words, mean {mean:.1f}) — mix short and long sentences")

    # 6. bullet-only body: bullets > 60% of non-empty lines
    nonempty = [l for l in lines if l.strip()]
    if nonempty:
        share = sum(1 for l in nonempty if re.match(r"\s*[-*]\s", l)) / len(nonempty)
        if share > 0.6:
            flags.append(f"{share:.0%} of lines are bullets — write some prose paragraphs")

    return flags, len(words), n_em


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    flags, n_words, n_em = audit(sys.argv[1])
    print(f"audited {sys.argv[1]}: {n_words} words, {n_em} em-dashes")
    if flags:
        print(f"\n{len(flags)} flag(s):")
        for f in flags:
            print("  " + f)
        sys.exit(1)
    print("clean — no slop flags")
