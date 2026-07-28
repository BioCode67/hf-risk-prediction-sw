#!/usr/bin/env bash
# Download the open datasets this project runs on.
#
# The datasets are not in git: MIMIC-IV's DUA forbids redistribution and the
# competition data may not leave the 안심존 at all, so data/ is git-ignored
# wholesale rather than per-file — that way a stray `git add` cannot leak a
# restricted dataset. The two PhysioNet Challenge sets below are openly
# licensed (ODC-BY) and need no credentials, so fetching them is scriptable.
#
# Usage:
#   scripts/fetch_data.sh challenge2012              # ICU mortality, 4,000 stays, ~27 MB
#   scripts/fetch_data.sh challenge2019              # sepsis, 20,336 patients, ~162 MB
#   scripts/fetch_data.sh challenge2019 --limit=3000 # only the first 3,000 patients
#   scripts/fetch_data.sh all
#
# Options:
#   --limit=N   fetch only the first N patient files (enough to follow the
#               notebooks on a laptop; omit for the full set)
#   --jobs=N    parallel connections (default 12)
#
# Target directory defaults to ./data, override with DATA_DIR:
#   DATA_DIR=/workspace/data scripts/fetch_data.sh challenge2012
#
# Re-running is cheap and safe: only missing files are requested, so an
# interrupted download resumes where it stopped.

set -euo pipefail

DATA_DIR="${DATA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
BASE="https://physionet.org/files"
LIMIT=0
JOBS=12

log() { printf '\n==> %s\n' "$*"; }

require_wget() {
    command -v wget >/dev/null 2>&1 || {
        echo "wget not found. Install it (conda install -c conda-forge wget," >&2
        echo "  apt install wget, or brew install wget) or download manually from" >&2
        echo "  https://physionet.org/content/" >&2
        exit 1
    }
}

# Fetch a PhysioNet directory of many small per-patient files into $2.
#
# These projects publish no archive, so it is one GET per patient — 4,000 of them
# for 2012 and 20,336 for 2019. The cost is connection setup, not bytes (~9 KB a
# file), which is why `wget -r` crawls at ~11 files/min and would take hours.
# Two fixes: `wget -i` reuses a single connection for a whole list (~215/min),
# and JOBS such lists run at once (~1,900/min — minutes, not hours).
#
#   $1 directory URL   $2 destination   $3 grep -oE pattern matching the filenames
fetch_files() {
    local url="$1" dest="$2" pattern="$3"
    local tmp total want missing
    mkdir -p "$dest"
    tmp="$(mktemp -d)"

    log "Listing $url"
    wget -qO- -T 60 -t 3 "$url" | grep -oE "$pattern" | sort -u > "$tmp/all.txt" || true
    total=$(wc -l < "$tmp/all.txt")
    if [ "$total" -eq 0 ]; then
        rm -rf "$tmp"
        echo "Could not list $url — is the network up?" >&2
        return 1
    fi

    if [ "$LIMIT" -gt 0 ]; then
        head -n "$LIMIT" "$tmp/all.txt" > "$tmp/want.txt"
    else
        cp "$tmp/all.txt" "$tmp/want.txt"
    fi

    # Ask only for what is not on disk yet, so re-runs resume instead of restart.
    (cd "$dest" && ls -1 2>/dev/null) | sort > "$tmp/have.txt" || true
    comm -23 "$tmp/want.txt" "$tmp/have.txt" > "$tmp/missing.txt"
    want=$(wc -l < "$tmp/want.txt")
    missing=$(wc -l < "$tmp/missing.txt")
    log "$total published · $want wanted · $missing to download (${JOBS}-way)"

    if [ "$missing" -eq 0 ]; then
        rm -rf "$tmp"
        log "already complete"
        return 0
    fi

    split -n "l/$JOBS" -d "$tmp/missing.txt" "$tmp/chunk_"
    for chunk in "$tmp"/chunk_*; do
        [ -s "$chunk" ] || continue
        wget -q -nc -T 30 -t 3 -P "$dest" -B "$url" -i "$chunk" &
    done
    wait

    # A truncated response leaves a zero-byte file; drop those so the next run
    # treats them as missing rather than done.
    find "$dest" -type f -size 0 -delete
    rm -rf "$tmp"
}

fetch_challenge2012() {
    log "Challenge 2012 — ICU mortality (4,000 stays, ~27 MB)"
    fetch_files "$BASE/challenge-2012/1.0.0/set-a/" \
        "$DATA_DIR/challenge2012/set-a" '[0-9]{6}\.txt'

    log "Challenge 2012 — outcome labels"
    # Without this file every patient loads as a control and nothing is learnable.
    wget -q --show-progress -N -P "$DATA_DIR/challenge2012" \
        "$BASE/challenge-2012/1.0.0/Outcomes-a.txt"

    local n
    n=$(find "$DATA_DIR/challenge2012/set-a" -name '*.txt' | wc -l)
    log "Done: $n records in $DATA_DIR/challenge2012/set-a"
    cat <<EOF

Run it:
  python src/mortality_explore.py $DATA_DIR/challenge2012/set-a \\
    --outcomes=$DATA_DIR/challenge2012/Outcomes-a.txt --horizon=6
EOF
}

fetch_challenge2019() {
    log "Challenge 2019 — sepsis (20,336 patients, ~162 MB)"
    fetch_files "$BASE/challenge-2019/1.0.0/training/training_setA/" \
        "$DATA_DIR/challenge2019/training_setA" 'p[0-9]+\.psv'

    local n
    n=$(find "$DATA_DIR/challenge2019/training_setA" -name '*.psv' | wc -l)
    log "Done: $n records in $DATA_DIR/challenge2019/training_setA"
    cat <<EOF

Run it:
  python src/sepsis_explore.py $DATA_DIR/challenge2019/training_setA --horizon=6

Or follow it cell by cell:
  jupyter lab notebooks/05_challenge2019_sepsis.ipynb
EOF
}

TARGET=""
for arg in "$@"; do
    case "$arg" in
        --limit=*) LIMIT="${arg#*=}" ;;
        --jobs=*)  JOBS="${arg#*=}" ;;
        -*)        echo "unknown option: $arg" >&2; exit 1 ;;
        *)         TARGET="$arg" ;;
    esac
done

case "$TARGET" in
    challenge2012) require_wget; fetch_challenge2012 ;;
    challenge2019) require_wget; fetch_challenge2019 ;;
    all)           require_wget; fetch_challenge2012; fetch_challenge2019 ;;
    *)
        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
        exit 1
        ;;
esac
