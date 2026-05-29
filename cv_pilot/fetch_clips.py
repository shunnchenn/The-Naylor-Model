#!/usr/bin/env python3
"""
fetch_clips.py  —  Find & download Statcast pitch clips from Baseball Savant
============================================================================

Discovers pitch video clips on Savant *itself* and downloads them with full
relational metadata, so the CV pilot's outputs join straight back to the
Naylor Model (game_pk + at_bat_number + pitch_number + pitcher_id + catcher_id).

The bridge nobody documents: Savant's game feed
    https://baseballsavant.mlb.com/gf?game_pk=<PK>
returns every pitch with its **play_id** (Film Room GUID) plus pitcher,
catcher, batter, p_throws, ab_number, pitch_number, inning, des, start_speed.
Given a play_id, the Film Room page
    https://baseballsavant.mlb.com/sporty-videos?playId=<GUID>
embeds the downloadable mp4 (https://sporty-clips.mlb.com/....mp4).

So: game_pk → gf feed → (play_id, pitcher, catcher, …) → sporty-videos → mp4.

Usage
-----
# 1) Whole games (all pitches), download the clips:
python3 cv_pilot/fetch_clips.py --game-pks 746161 746158 746150 --download

# 2) Only specific pitches (e.g. the model's SB attempts):
#    attempts.csv must have columns: game_pk,at_bat_number,pitch_number
python3 cv_pilot/fetch_clips.py --attempts attempts.csv --download

# 3) Reverse-resolve known play_ids to full metadata (scan a team's games):
python3 cv_pilot/fetch_clips.py --resolve PLAYID1,PLAYID2 \
        --season 2024 --home-team 119 --opp 135,120,113 --download

# Manifest only (no video download):
python3 cv_pilot/fetch_clips.py --game-pks 746161

Outputs
-------
cv_pilot/clips_manifest.csv   one row per resolved pitch (relational keys + mp4_url + clip_id)
cv_pilot/clips_meta.csv       the subset the detector reads (clip_id, pitcher/catcher, join keys)
cv_pilot/clips/<clip_id>.mp4  downloaded video (with --download; gitignored)
"""

from __future__ import annotations
import argparse
import html
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("requests not installed.  pip install requests")

ROOT          = Path(__file__).resolve().parent
CLIPS_DIR     = ROOT / "clips"
MANIFEST_CSV  = ROOT / "clips_manifest.csv"
META_CSV      = ROOT / "clips_meta.csv"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
GF_URL     = "https://baseballsavant.mlb.com/gf?game_pk={pk}"
SPORTY_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={pid}"
SCHED_URL  = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}"
              "&teamId={team}&opponentId={opp}&gameType=R")
MP4_RE = re.compile(r"https?://sporty-clips\.mlb\.com/[^\"' ]+\.mp4", re.I)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})
SLEEP = 0.5      # be polite between requests

# Per-pitch fields we keep (all are relational / descriptive)
GF_FIELDS = ["play_id", "pitcher", "pitcher_name", "catcher", "catcher_name",
             "batter", "batter_name", "p_throws", "ab_number", "pitch_number",
             "inning", "des", "start_speed", "pitch_type", "call_name"]


def _get_json(url, tries=3):
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == tries - 1:
                print(f"  ! GET failed {url}: {e}")
                return None
            time.sleep(1.0 + i)
    return None


def gf_records(game_pk: int) -> list[dict]:
    """All pitches in a game, with relational metadata, from the Savant gf feed."""
    d = _get_json(GF_URL.format(pk=game_pk))
    if not d:
        return []
    out = []
    for side in ("team_home", "team_away"):
        for p in d.get(side, []):
            if not p.get("play_id"):
                continue
            rec = {k: p.get(k) for k in GF_FIELDS}
            rec["game_pk"] = game_pk
            rec["at_bat_number"] = p.get("ab_number")   # gf ab_number == statcast at_bat_number
            out.append(rec)
    time.sleep(SLEEP)
    return out


def sporty_mp4_url(play_id: str) -> str | None:
    """Scrape the Film Room page for the downloadable mp4 URL."""
    try:
        r = SESSION.get(SPORTY_URL.format(pid=play_id), timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  ! sporty page failed {play_id}: {e}")
        return None
    m = MP4_RE.search(r.text)
    time.sleep(SLEEP)
    # the scraped URL can carry HTML entities (e.g. trailing == shows as
    # &#x3D;&#x3D;); unescape so the link actually resolves
    return html.unescape(m.group(0)) if m else None


def _ascii_last(name: str) -> str:
    """'Emilio Pagán' -> 'pagan' (ascii, lowercase last token)."""
    if not isinstance(name, str) or not name.strip():
        return "pitcher"
    last = name.strip().split()[-1]
    norm = unicodedata.normalize("NFKD", last).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", norm.lower()) or "pitcher"


def clip_id_for(rec: dict) -> str:
    pid8 = str(rec["play_id"])[:8]
    last = _ascii_last(rec.get("pitcher_name"))
    arm = (rec.get("p_throws") or "R").upper()[0]
    return f"{pid8}_{last}_{arm}.mp4"


def download(url: str, dest: Path) -> bool:
    try:
        with SESSION.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"  ! download failed {url}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Selection strategies
# ─────────────────────────────────────────────────────────────────────────────
def from_game_pks(game_pks, sb_only=False) -> list[dict]:
    recs = []
    for pk in game_pks:
        print(f"[gf] game_pk={pk}")
        g = gf_records(int(pk))
        if sb_only:
            g = [r for r in g if re.search(r"steal|caught stealing|stole",
                                           str(r.get("des", "")), re.I)]
        recs.extend(g)
    return recs


def from_attempts(csv_path: Path) -> list[dict]:
    """Target exact pitches: attempts.csv with game_pk,at_bat_number,pitch_number."""
    df = pd.read_csv(csv_path)
    need = {"game_pk", "at_bat_number", "pitch_number"}
    if not need.issubset(df.columns):
        sys.exit(f"{csv_path} must have columns {need}")
    want = {(int(r.game_pk), int(r.at_bat_number), int(r.pitch_number))
            for r in df.itertuples()}
    recs = []
    for pk in sorted({k[0] for k in want}):
        print(f"[gf] game_pk={pk}")
        for r in gf_records(pk):
            key = (int(pk), int(r.get("at_bat_number") or -1),
                   int(r.get("pitch_number") or -1))
            if key in want:
                recs.append(r)
    return recs


def from_targets(csv_path: Path) -> list[dict]:
    """Fetch specific plays by play_id (e.g. exact steal pitches from GUMBO).

    targets.csv needs columns: game_pk, play_id  [, is_naylor, clip_prefix]
    Matches each play_id against its game's gf feed to recover full relational
    metadata (at_bat/pitch/catcher/names) and tags the row for clip naming.
    """
    df = pd.read_csv(csv_path)
    need = {"game_pk", "play_id"}
    if not need.issubset(df.columns):
        sys.exit(f"{csv_path} must have columns {need}")
    want = {}
    for r in df.itertuples():
        want[str(r.play_id)] = {
            "is_naylor": int(getattr(r, "is_naylor", 0) or 0),
            "clip_prefix": str(getattr(r, "clip_prefix", "") or ""),
        }
    recs = []
    for pk in sorted(df["game_pk"].astype(int).unique()):
        print(f"[gf] game_pk={pk}")
        for r in gf_records(int(pk)):
            tag = want.get(str(r.get("play_id")))
            if tag is not None:
                r["_is_naylor"] = tag["is_naylor"]
                r["_clip_prefix"] = tag["clip_prefix"]
                recs.append(r)
    found = {str(r["play_id"]) for r in recs}
    missing = set(want) - found
    if missing:
        print(f"  ! play_ids not found in gf feeds: {len(missing)}")
    return recs


def resolve_play_ids(play_ids, season, home_team, opps) -> list[dict]:
    """Reverse-resolve play_ids by scanning a home team's games vs given opponents."""
    targets = set(play_ids)
    pks = []
    for opp in opps:
        d = _get_json(SCHED_URL.format(season=season, team=home_team, opp=opp))
        if not d:
            continue
        for dt in d.get("dates", []):
            for g in dt["games"]:
                if g["teams"]["home"]["team"]["id"] == int(home_team):
                    pks.append(g["gamePk"])
    recs, found = [], set()
    for pk in pks:
        for r in gf_records(pk):
            if r["play_id"] in targets:
                recs.append(r)
                found.add(r["play_id"])
        if found == targets:
            break
    missing = targets - found
    if missing:
        print(f"  ! unresolved play_ids: {missing}")
    return recs


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Fetch Savant pitch clips with relational metadata")
    ap.add_argument("--game-pks", nargs="*", type=int, help="game_pk(s) to pull")
    ap.add_argument("--attempts", help="CSV with game_pk,at_bat_number,pitch_number")
    ap.add_argument("--targets", help="CSV with game_pk,play_id[,is_naylor,clip_prefix]")
    ap.add_argument("--out-dir", help="base output dir (clips/, clips_meta.csv, clips_manifest.csv)")
    ap.add_argument("--resolve", help="comma-separated play_ids to reverse-resolve")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--home-team", type=int, default=119, help="MLBAM team id (default 119 LAD)")
    ap.add_argument("--opp", default="", help="comma-separated opponent team ids for --resolve")
    ap.add_argument("--sb-only", action="store_true",
                    help="with --game-pks, keep only pitches whose des mentions a steal")
    ap.add_argument("--download", action="store_true", help="download the mp4s (else manifest only)")
    args = ap.parse_args()

    global CLIPS_DIR, MANIFEST_CSV, META_CSV
    if args.out_dir:
        base = Path(args.out_dir)
        base.mkdir(parents=True, exist_ok=True)
        CLIPS_DIR    = base / "clips"
        MANIFEST_CSV = base / "clips_manifest.csv"
        META_CSV     = base / "clips_meta.csv"

    if args.targets:
        recs = from_targets(Path(args.targets))
    elif args.attempts:
        recs = from_attempts(Path(args.attempts))
    elif args.resolve:
        opps = [int(x) for x in args.opp.split(",") if x.strip()]
        if not opps:
            sys.exit("--resolve needs --opp <team_ids>")
        recs = resolve_play_ids(args.resolve.split(","), args.season, args.home_team, opps)
    elif args.game_pks:
        recs = from_game_pks(args.game_pks, sb_only=args.sb_only)
    else:
        sys.exit("Provide one of: --game-pks / --attempts / --targets / --resolve")

    if not recs:
        sys.exit("No records resolved.")

    # build manifest + resolve mp4 urls (+ optional download)
    for r in recs:
        cid = clip_id_for(r)
        # name the clip Naylor actually faced so it is self-identifying
        if r.get("_is_naylor"):
            cid = "NAYLOR_" + cid
        elif r.get("_clip_prefix"):
            cid = f"{r['_clip_prefix']}_" + cid
        r["clip_id"] = cid
        r["is_naylor"] = int(r.get("_is_naylor", 0) or 0)
        r["mp4_url"] = sporty_mp4_url(r["play_id"])
        if args.download and r["mp4_url"]:
            dest = CLIPS_DIR / r["clip_id"]
            ok = download(r["mp4_url"], dest)
            r["downloaded"] = ok
            print(f"  {'✓' if ok else '✗'} {r['clip_id']}  ({r.get('pitcher_name')} -> {r.get('catcher_name')})")
        else:
            r["downloaded"] = False

    man = pd.DataFrame(recs)
    man.to_csv(MANIFEST_CSV, index=False)
    print(f"\n[write] {MANIFEST_CSV}  ({len(man)} pitches)")

    # detector-facing metadata (relational join keys + names)
    meta_cols = ["clip_id", "pitcher_name", "pitcher_id", "p_throws",
                 "catcher_name", "catcher_id", "batter_name", "play_id",
                 "game_pk", "at_bat_number", "pitch_number", "is_naylor"]
    meta = man.rename(columns={"pitcher": "pitcher_id", "catcher": "catcher_id"})
    meta = meta.reindex(columns=meta_cols)
    meta.to_csv(META_CSV, index=False)
    print(f"[write] {META_CSV}  (feeds extract_delivery.py)")
    if args.download:
        n = int(man["downloaded"].sum())
        print(f"[done] downloaded {n}/{len(man)} clips to {CLIPS_DIR}/")
    else:
        print("[note] manifest only — re-run with --download to fetch mp4s")


if __name__ == "__main__":
    main()
