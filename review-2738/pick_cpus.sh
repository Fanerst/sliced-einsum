#!/bin/bash
# Print the 16 least-busy physical cores of NUMA node 1 of a 2 x 24-core host (CPUs 24-47, with
# hyper-thread siblings 72-95), from a 5 s sample of /proc/stat; busy = core + sibling.
# Adjust the ranges for another host.
tmp=$(mktemp)
a=$(grep '^cpu[0-9]' /proc/stat); sleep 5; b=$(grep '^cpu[0-9]' /proc/stat)
paste <(echo "$a") <(echo "$b") | awk '{n=NF/2; t1=0; t2=0; for(i=2;i<=n;i++){t1+=$i; t2+=$(i+n)}; idle1=$5+$6; idle2=$(5+n)+$(6+n); printf "%s %d\n", substr($1,4), (1-(idle2-idle1)/(t2-t1))*100}' > "$tmp"
for c in $(seq 24 47); do
    echo "$c $(( $(awk -v c=$c '$1==c{print $2}' "$tmp") + $(awk -v c=$((c+48)) '$1==c{print $2}' "$tmp") ))"
done | sort -k2 -n | head -16 | awk '{print $1}' | sort -n | paste -sd, -
rm -f "$tmp"
