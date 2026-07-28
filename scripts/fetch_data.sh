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
#   scripts/fetch_data.sh challenge2012     # ICU mortality, 4,000 stays, ~27 MB
#   scripts/fetch_data.sh challenge2019     # sepsis, 20,000 patients, ~140 MB
#   scripts/fetch_data.sh all
#
# Target directory defaults to ./data, override with DATA_DIR:
#   DATA_DIR=/workspace/data scripts/fetch_data.sh challenge2012

set -euo pipefail

DATA_DIR="${DATA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
BASE="https://physionet.org/files"

log() { printf '\n==> %s\n' "$*"; }

require_wget() {
    command -v wget >/dev/null 2>&1 || {
        echo "wget not found. Install it (conda install -c conda-forge wget) or" >&2
        echo "download manually from https://physionet.org/content/" >&2
        exit 1
    }
}

# Fetch a PhysioNet directory into $2, flattening $3 leading path components so
# the records land directly in $2. -c resumes and -N skips unchanged files, so
# re-running after an interrupted download is cheap and safe.
fetch_dir() {
    local url="$1" dest="$2" cut="$3"
    mkdir -p "$dest"
    # -R drops the directory index pages wget fetches to discover links.
    wget -q --show-progress -r -N -c -np -nH --cut-dirs="$cut" \
        -R "index.html*" -P "$dest" "$url"
}

fetch_challenge2012() {
    log "Challenge 2012 — ICU mortality (4,000 stays, ~27 MB)"
    # files/challenge-2012/1.0.0/set-a -> 4 components to strip
    fetch_dir "$BASE/challenge-2012/1.0.0/set-a/" "$DATA_DIR/challenge2012/set-a" 4

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
    log "Challenge 2019 — sepsis (20,000 patients, ~140 MB, takes a while)"
    # files/challenge-2019/1.0.0/training/training_setA -> 5 components
    fetch_dir "$BASE/challenge-2019/1.0.0/training/training_setA/" \
        "$DATA_DIR/challenge2019/training_setA" 5

    local n
    n=$(find "$DATA_DIR/challenge2019/training_setA" -name '*.psv' | wc -l)
    log "Done: $n records in $DATA_DIR/challenge2019/training_setA"
    cat <<EOF

Run it:
  python src/sepsis_explore.py $DATA_DIR/challenge2019/training_setA --horizon=6
EOF
}

case "${1:-}" in
    challenge2012) require_wget; fetch_challenge2012 ;;
    challenge2019) require_wget; fetch_challenge2019 ;;
    all)           require_wget; fetch_challenge2012; fetch_challenge2019 ;;
    *)
        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
        exit 1
        ;;
esac
