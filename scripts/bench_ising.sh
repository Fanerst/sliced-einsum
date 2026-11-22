#!/bin/bash
# The cubic Ising benchmark. One TreeSA tree for the lattice, one slicing per target on that tree,
# then every saved plan run by tenferro (offset and rebased views) and by PyTorch on the same tree
# and slicing, in turns. Build the examples first (`cargo build --release --examples`); set PY to
# a Python with PyTorch (default python3).
#
# Usage: scripts/bench_ising.sh LX LY LZ "TARGETS" REPEATS OUTDIR [SLICER [TREEFILE]]
#   SLICER: fast (default) or default, omeco's slicer preset.
#   TREEFILE: an earlier `*-none.json` export to reuse its tree instead of running TreeSA again.
set -u
LX=$1 LY=$2 LZ=$3 TARGETS=$4 REPEATS=$5 OUT=$6 SLICER=${7:-fast} TREEFILE=${8:-}
B=./target/release/examples/ising_sliced
PY=${PY:-python3}
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 RAYON_NUM_THREADS=16
mkdir -p "$OUT"
tag=${LX}x${LY}x${LZ}
echo "== $(date +%T) $tag: tree and slicings ($SLICER slicer)"
if [ -n "$TREEFILE" ]; then
    cp "$TREEFILE" "$OUT/$tag-none.json"
else
    "$B" "$LX" "$LY" "$LZ" 0.2 16 treesa-default none offset 0 "$OUT/$tag-none.json" \
        > "$OUT/$tag-none-plan.log"
fi
for t in $TARGETS; do
    "$B" "$LX" "$LY" "$LZ" 0.2 16 "$OUT/$tag-none.json" "$t" offset 0 "$OUT/$tag-t$t.json" "$SLICER" \
        > "$OUT/$tag-t$t-plan.log"
done
plans="none"
[ -n "$TREEFILE" ] && plans=""
for t in $TARGETS; do plans="$plans t$t"; done
for p in $plans; do
    echo "== $(date +%T) $tag $p"
    for mode in offset rebased; do
        "$B" "$LX" "$LY" "$LZ" 0.2 16 "$OUT/$tag-$p.json" saved "$mode" "$REPEATS" >> "$OUT/$tag-runs.jsonl"
    done
    "$PY" scripts/bench_torch.py "$OUT/$tag-$p.json" 16 "$REPEATS" "$p" >> "$OUT/$tag-torch.jsonl"
done
echo "== $(date +%T) done"
