#!/usr/bin/env bash
# ABBA: two prebuilt benchmark binaries, alternated *within* a round as A B B A.
#
# A plain A-B alternation cannot see an ordering effect, and this machine has one worth
# about eight per cent: whichever binary runs first in a round is the slow one. ABBA
# cancels any drift that is monotone across the round -- thermal, frequency, page cache --
# because each build gets one early slot and one late slot, so the two means carry the same
# share of it. It also measures the drift, as the gap between each build's own two samples,
# which is the number that says whether the round is usable at all.
#
#   usage: abba.sh <a.exe> <b.exe> <bench filter> [rounds] [measurement seconds]

set -u

A="$1"
B="$2"
FILTER="$3"
ROUNDS="${4:-4}"
SECONDS_EACH="${5:-3}"

one() {
    "$1" --bench "$FILTER" --measurement-time "$SECONDS_EACH" --warm-up-time 1 2>&1 |
        grep -oE "time: +\[[0-9.]+ [munp]?s [0-9.]+ [munp]?s" |
        head -1 |
        awk '{print $4, $5}'
}

# Normalise to microseconds so the two columns are comparable whatever criterion picks.
us() {
    local value unit
    value=$(echo "$1" | awk '{print $1}')
    unit=$(echo "$1" | awk '{print $2}')
    case "$unit" in
    ns) echo "$value" | awk '{printf "%.3f", $1 / 1000.0}' ;;
    us) echo "$value" ;;
    ms) echo "$value" | awk '{printf "%.3f", $1 * 1000.0}' ;;
    s) echo "$value" | awk '{printf "%.3f", $1 * 1000000.0}' ;;
    *) echo "$value" ;;
    esac
}

echo "ABBA  $FILTER   $ROUNDS rounds of A B B A, ${SECONDS_EACH}s each"
echo "      A = $(basename "$A")"
echo "      B = $(basename "$B")"
echo

a_all=""
b_all=""
for round in $(seq 1 "$ROUNDS"); do
    a1=$(us "$(one "$A")")
    b1=$(us "$(one "$B")")
    b2=$(us "$(one "$B")")
    a2=$(us "$(one "$A")")
    am=$(echo "$a1 $a2" | awk '{printf "%.3f", ($1 + $2) / 2}')
    bm=$(echo "$b1 $b2" | awk '{printf "%.3f", ($1 + $2) / 2}')
    drift=$(echo "$a1 $a2 $b1 $b2" | awk '{printf "%+.1f%%", 100.0 * (($2 - $1) + ($3 - $4)) / ($1 + $4)}')
    printf "round %s   A %8s %8s -> %8s    B %8s %8s -> %8s    drift %s\n" \
        "$round" "$a1" "$a2" "$am" "$b1" "$b2" "$bm" "$drift"
    a_all="$a_all $am"
    b_all="$b_all $bm"
done

echo
echo "$a_all" | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; printf "A mean %.3f us over %d rounds\n", s/NF, NF}'
echo "$b_all" | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; printf "B mean %.3f us over %d rounds\n", s/NF, NF}'
echo "$a_all|$b_all" | awk -F'|' '{
    na=split($1,a," "); nb=split($2,b," ");
    sa=0; for(i=1;i<=na;i++) sa+=a[i];
    sb=0; for(i=1;i<=nb;i++) sb+=b[i];
    printf "B is %+.1f%% against A\n", 100.0 * ((sb/nb) - (sa/na)) / (sa/na);
}'
