#!/bin/bash
# E3: all detection runs (3 seeds x 5 methods); logs in results/detect/*.log
cd "$(dirname "$0")/.."
mkdir -p results/detect
for seed in 0 1 2; do
  for m in heatmap hdome hybrid hdome_K5 hdome_K20; do
    [ -f results/detect/${m}_s${seed}.json ] && continue
    python experiments/detect.py $m $seed > results/detect/${m}_s${seed}.log 2>&1
  done
done
