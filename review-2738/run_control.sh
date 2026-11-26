#!/bin/bash
# Control for the tenferro version: the prototype as published (the repository root built with
# its own Cargo.toml, on tenferro-rs c4db040) against the prototype and the port on main 1b67fb8f,
# in turns, on the same cores and threads. Appends to logs/ising.jsonl as round 3.
# Usage, after building harness/ and the root (cargo build --release --example ising_sliced):
#   CPUS=... REP=2 ./run_control.sh
set -u
cd "$(dirname "$0")"
B=harness/target/release
OLD=../target/release/examples/ising_sliced
CPUS=${CPUS:?set CPUS}
REP=${REP:-2}
mkdir -p logs
export OMP_NUM_THREADS=16 RAYON_NUM_THREADS=16 MKL_NUM_THREADS=16
plan_file() {
    case $1 in
        d-*) echo "../bench/default/${1#d-}.json" ;;
        f-*) echo "../bench/fast/${1#f-}.json" ;;
    esac
}
plans="5:5:8:d-5x5x8-none 5:5:8:d-5x5x8-t17 6:6:6:d-6x6x6-t24 5:5:8:f-5x5x8-t15"
for spec in $plans; do
    IFS=: read -r lx ly lz plan <<< "$spec"
    file=$(plan_file "$plan")
    echo "== $(date +%T) control $plan"
    taskset -c "$CPUS" $OLD "$lx" "$ly" "$lz" 0.2 16 "$file" saved offset "$REP" \
        | sed "s/^{/{\"impl\":\"proto-c4db040\",\"round\":3,\"plan\":\"$plan\",/" >> logs/ising.jsonl
    taskset -c "$CPUS" $B/examples/ising_sliced "$lx" "$ly" "$lz" 0.2 16 "$file" saved offset "$REP" \
        | sed "s/^{/{\"impl\":\"proto\",\"round\":3,\"plan\":\"$plan\",/" >> logs/ising.jsonl
    taskset -c "$CPUS" $B/ising_port "$lx" "$ly" "$lz" 0.2 16 "$file" saved "$REP" \
        | sed "s/^{/{\"round\":3,\"plan\":\"$plan\",/" >> logs/ising.jsonl
done
echo "== $(date +%T) done"
