#!/bin/bash
# The prototype's cubic Ising plans, run in turns by the prototype (the repository root's src/,
# built on tenferro-rs main 1b67fb8f) and by tenferro-einsum's SlicedEinsumPlan (#2738), on the
# same build: 16 threads on the cores in CPUS (taskset), f64. Each process plans once and runs
# REP times. Appends JSON lines to logs/ising.jsonl.
# Usage, after building harness/: CPUS=$(./pick_cpus.sh) ROUNDS=2 REP=2 ./run_ising.sh
set -u
cd "$(dirname "$0")"
B=harness/target/release
CPUS=${CPUS:?set CPUS}
ROUNDS=${ROUNDS:-2}
REP=${REP:-2}
mkdir -p logs
export OMP_NUM_THREADS=16 RAYON_NUM_THREADS=16 MKL_NUM_THREADS=16
# d-NAME is ../bench/default/NAME.json and f-NAME is ../bench/fast/NAME.json (omeco's default and
# fast slicer presets).
plan_file() {
    case $1 in
        d-*) echo "../bench/default/${1#d-}.json" ;;
        f-*) echo "../bench/fast/${1#f-}.json" ;;
    esac
}
plans="6:6:6:d-6x6x6-none 6:6:6:d-6x6x6-t24 6:6:6:d-6x6x6-t26 5:5:8:d-5x5x8-none 5:5:8:d-5x5x8-t17 5:5:8:f-5x5x8-t15"
for round in $(seq 1 "$ROUNDS"); do
    for spec in $plans; do
        IFS=: read -r lx ly lz plan <<< "$spec"
        file=$(plan_file "$plan")
        echo "== $(date +%T) round $round $plan"
        taskset -c "$CPUS" $B/examples/ising_sliced "$lx" "$ly" "$lz" 0.2 16 "$file" saved offset "$REP" \
            | sed "s/^{/{\"impl\":\"proto\",\"round\":$round,\"plan\":\"$plan\",/" >> logs/ising.jsonl
        taskset -c "$CPUS" $B/ising_port "$lx" "$ly" "$lz" 0.2 16 "$file" saved "$REP" \
            | sed "s/^{/{\"round\":$round,\"plan\":\"$plan\",/" >> logs/ising.jsonl
    done
done
echo "== $(date +%T) done"
