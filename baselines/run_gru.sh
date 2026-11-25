#!/bin/sh
# Rung 3 GRU baselines, in the registered order: per size, the lr0 grid on
# seed 0, then seeds 1 and 2 at the lr0 with the lower seed-0 validation BPC.
# Finished (hidden, lr0, seed) lines in results/rung3-gru.tsv are skipped.
set -e
PY=.venv/bin/python
T=data/text8/text8
TSV=results/rung3-gru.tsv
touch $TSV
run() { grep -q "^$1	$2	$3	" $TSV || $PY baselines/gru.py $T $1 $2 $3 2; }
for H in 64 128 292; do
  run $H 0.001 0 & run $H 0.003 0 & wait
  BEST=$(awk -F'\t' -v h=$H '$1==h && $3==0 {print $5, $2}' $TSV | sort -n | head -1 | cut -d' ' -f2)
  run $H $BEST 1 & run $H $BEST 2 & wait
done
