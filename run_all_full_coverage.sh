#!/usr/bin/env bash
# run_all.sh
#
# Comprehensive sweep runner for GA, vanilla PBIL, and Island PBIL.


set -euo pipefail

# Activate virtual environment if present
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
banner() {
  echo
  echo "============================================================"
  echo "[$(stamp)] $1"
  echo "============================================================"
}

run_cmd () {
  echo ">>> $*"
  "$@"
}

has_flag () {
  # Usage: has_flag <script.py> <flag>
  python3 "$1" -h 2>&1 | grep -q -- "$2"
}

# Resolve workers for multiprocessing, if supported
NPROC="$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)"

GA="GA.py"
VPBIL="vanilla_pbil.py"
ISLAND="pbil_island.py"

FUNCS=(F1 F2 F3)

# ----------------------------- GA -----------------------------
banner "GA runs"

for f in "${FUNCS[@]}"; do
  run_cmd python3 "$GA" --func "$f" --runs 30
done

# Optional sigma_mating sweeps
if has_flag "$GA" "--sigma_mating"; then
  run_cmd python3 "$GA" --func F1 --runs 30 --sigma_mating 0.1
  run_cmd python3 "$GA" --func F2 --runs 30 --sigma_mating 0.1
  run_cmd python3 "$GA" --func F3 --runs 30 --sigma_mating 0.5
fi

# Optional seed sweeps
if has_flag "$GA" "--seed"; then
  for s in 0 1 2; do
    for f in "${FUNCS[@]}"; do
      run_cmd python3 "$GA" --func "$f" --runs 30 --seed "$s"
    done
  done
fi

# -------------------------- Vanilla PBIL --------------------------
banner "Vanilla PBIL runs"

for f in "${FUNCS[@]}"; do
  run_cmd python3 "$VPBIL" --func "$f" --runs 30
done

# Optional PBIL sweeps: eval budget and sampling pressure
if has_flag "$VPBIL" "--total_evals" && has_flag "$VPBIL" "--num_samples"; then
  EVALS=(20000 40000 60000)
  SAMPLES=(25 50 100)
  for f in "${FUNCS[@]}"; do
    for e in "${EVALS[@]}"; do
      for ns in "${SAMPLES[@]}"; do
        if [ "$e" -eq 20000 ] && [ "$ns" -eq 50 ]; then
          continue
        fi
        run_cmd python3 "$VPBIL" --func "$f" --runs 30 --total_evals "$e" --num_samples "$ns"
      done
    done
  done
fi

# -------------------------- Island PBIL --------------------------
banner "Island PBIL runs"

maybe_switch () {
  local script="$1" flag="$2"
  if has_flag "$script" "$flag"; then
    echo "$flag"
  fi
}

MP_FLAG="$(maybe_switch "$ISLAND" "--mp")"
MP_WORKERS_FLAG=""
if has_flag "$ISLAND" "--mp_workers"; then
  MP_WORKERS_FLAG="--mp_workers"
fi

NO_NUDGE_FLAG="$(maybe_switch "$ISLAND" "--no_nudge")"
NUDGE_FLAG="$(maybe_switch "$ISLAND" "--nudge")"
NUDGE_STRENGTH_FLAG=""
if has_flag "$ISLAND" "--nudge_strength"; then
  NUDGE_STRENGTH_FLAG="--nudge_strength"
fi

REPEL_FLAG=""
if has_flag "$ISLAND" "--repel"; then
  REPEL_FLAG="--repel"
elif has_flag "$ISLAND" "--repulsion"; then
  REPEL_FLAG="--repulsion"
fi

INTERVALS=(2 5 10)
SIGMA_1D=(0.1 0.2 0.4 0.8)
SIGMA_2D=(0.5 1.0 2.0)
NUDGE_STRENGTHS=(0.25 0.60 1.00)

banner "Island PBIL: mode=none (baseline)"
run_cmd python3 "$ISLAND" --func F1 --mode none --num_islands 5 --runs 30
run_cmd python3 "$ISLAND" --func F2 --mode none --num_islands 5 --runs 30
run_cmd python3 "$ISLAND" --func F3 --mode none --num_islands 4 --runs 30

if [ -n "$MP_FLAG" ]; then
  run_cmd python3 "$ISLAND" --func F1 --mode none --num_islands 5 --runs 30 $MP_FLAG ${MP_WORKERS_FLAG:+$MP_WORKERS_FLAG "$NPROC"}
  run_cmd python3 "$ISLAND" --func F2 --mode none --num_islands 5 --runs 30 $MP_FLAG ${MP_WORKERS_FLAG:+$MP_WORKERS_FLAG "$NPROC"}
  run_cmd python3 "$ISLAND" --func F3 --mode none --num_islands 4 --runs 30 $MP_FLAG ${MP_WORKERS_FLAG:+$MP_WORKERS_FLAG "$NPROC"}
fi

if [ -n "$REPEL_FLAG" ]; then
  run_cmd python3 "$ISLAND" --func F1 --mode none --num_islands 5 --runs 30 $REPEL_FLAG
  run_cmd python3 "$ISLAND" --func F2 --mode none --num_islands 5 --runs 30 $REPEL_FLAG
  run_cmd python3 "$ISLAND" --func F3 --mode none --num_islands 4 --runs 30 $REPEL_FLAG
fi

banner "Island PBIL: mode=base (speciation)"
run_cmd python3 "$ISLAND" --func F1 --mode base --num_islands 5 --runs 30
run_cmd python3 "$ISLAND" --func F2 --mode base --num_islands 5 --runs 30
run_cmd python3 "$ISLAND" --func F3 --mode base --num_islands 4 --runs 30

if has_flag "$ISLAND" "--niching_interval" && has_flag "$ISLAND" "--sigma_niche"; then
  for interval in "${INTERVALS[@]}"; do
    for sigma in "${SIGMA_1D[@]}"; do
      run_cmd python3 "$ISLAND" --func F1 --mode base --num_islands 5 --runs 30 --niching_interval "$interval" --sigma_niche "$sigma"
      run_cmd python3 "$ISLAND" --func F2 --mode base --num_islands 5 --runs 30 --niching_interval "$interval" --sigma_niche "$sigma"
    done
    for sigma in "${SIGMA_2D[@]}"; do
      run_cmd python3 "$ISLAND" --func F3 --mode base --num_islands 4 --runs 30 --niching_interval "$interval" --sigma_niche "$sigma"
    done
  done
fi

if [ -n "$NO_NUDGE_FLAG" ]; then
  banner "Island PBIL: base nudging comparison"
  run_cmd python3 "$ISLAND" --func F1 --mode base --num_islands 5 --runs 30
  run_cmd python3 "$ISLAND" --func F1 --mode base --num_islands 5 --runs 30 $NO_NUDGE_FLAG
  run_cmd python3 "$ISLAND" --func F2 --mode base --num_islands 5 --runs 30
  run_cmd python3 "$ISLAND" --func F2 --mode base --num_islands 5 --runs 30 $NO_NUDGE_FLAG
  run_cmd python3 "$ISLAND" --func F3 --mode base --num_islands 4 --runs 30
  run_cmd python3 "$ISLAND" --func F3 --mode base --num_islands 4 --runs 30 $NO_NUDGE_FLAG
elif [ -n "$NUDGE_FLAG" ]; then
  banner "Island PBIL: base nudging comparison"
  run_cmd python3 "$ISLAND" --func F1 --mode base --num_islands 5 --runs 30 $NUDGE_FLAG
  run_cmd python3 "$ISLAND" --func F2 --mode base --num_islands 5 --runs 30 $NUDGE_FLAG
  run_cmd python3 "$ISLAND" --func F3 --mode base --num_islands 4 --runs 30 $NUDGE_FLAG
fi

if [ -n "$NUDGE_STRENGTH_FLAG" ]; then
  banner "Island PBIL: nudge strength sweep (base)"
  for ns in "${NUDGE_STRENGTHS[@]}"; do
    run_cmd python3 "$ISLAND" --func F1 --mode base --num_islands 5 --runs 30 $NUDGE_STRENGTH_FLAG "$ns"
    run_cmd python3 "$ISLAND" --func F2 --mode base --num_islands 5 --runs 30 $NUDGE_STRENGTH_FLAG "$ns"
    run_cmd python3 "$ISLAND" --func F3 --mode base --num_islands 4 --runs 30 $NUDGE_STRENGTH_FLAG "$ns"
  done
fi

if [ -n "$REPEL_FLAG" ] && [ -n "$MP_FLAG" ]; then
  run_cmd python3 "$ISLAND" --func F2 --mode base --num_islands 5 --runs 30 $REPEL_FLAG $MP_FLAG ${MP_WORKERS_FLAG:+$MP_WORKERS_FLAG "$NPROC"}
fi

banner "Island PBIL: mode=proportional"
run_cmd python3 "$ISLAND" --func F1 --mode proportional --num_islands 5 --runs 30
run_cmd python3 "$ISLAND" --func F2 --mode proportional --num_islands 5 --runs 30
run_cmd python3 "$ISLAND" --func F3 --mode proportional --num_islands 4 --runs 30

run_cmd python3 "$ISLAND" --func F1 --mode proportional --num_islands 100 --runs 30
run_cmd python3 "$ISLAND" --func F2 --mode proportional --num_islands 100 --runs 30
run_cmd python3 "$ISLAND" --func F3 --mode proportional --num_islands 100 --runs 30

if has_flag "$ISLAND" "--niching_interval" && has_flag "$ISLAND" "--sigma_niche"; then
  for interval in 5 10; do
    run_cmd python3 "$ISLAND" --func F1 --mode proportional --num_islands 100 --runs 30 --niching_interval "$interval" --sigma_niche 0.8
    run_cmd python3 "$ISLAND" --func F2 --mode proportional --num_islands 100 --runs 30 --niching_interval "$interval" --sigma_niche 0.8
    run_cmd python3 "$ISLAND" --func F3 --mode proportional --num_islands 100 --runs 30 --niching_interval "$interval" --sigma_niche 1.0
  done
fi

if [ -n "$NO_NUDGE_FLAG" ]; then
  run_cmd python3 "$ISLAND" --func F2 --mode proportional --num_islands 100 --runs 30 $NO_NUDGE_FLAG
elif [ -n "$NUDGE_FLAG" ]; then
  run_cmd python3 "$ISLAND" --func F2 --mode proportional --num_islands 100 --runs 30 $NUDGE_FLAG
fi

if [ -n "$REPEL_FLAG" ] && [ -n "$MP_FLAG" ]; then
  run_cmd python3 "$ISLAND" --func F2 --mode proportional --num_islands 100 --runs 30 $REPEL_FLAG $MP_FLAG ${MP_WORKERS_FLAG:+$MP_WORKERS_FLAG "$NPROC"}
fi

if python3 "$ISLAND" -h 2>&1 | grep -q "peak_repulsion"; then
  banner "Island PBIL: mode=peak_repulsion"
  if has_flag "$ISLAND" "--total_evals"; then
    run_cmd python3 "$ISLAND" --func F1 --mode peak_repulsion --num_islands 5 --runs 30 --niching_interval 5 --sigma_niche 0.1 --total_evals 60000
    run_cmd python3 "$ISLAND" --func F1 --mode peak_repulsion --num_islands 5 --runs 30 --niching_interval 10 --sigma_niche 0.1 --total_evals 60000
    run_cmd python3 "$ISLAND" --func F2 --mode peak_repulsion --num_islands 5 --runs 30 --niching_interval 5 --sigma_niche 0.1 --total_evals 60000
    run_cmd python3 "$ISLAND" --func F2 --mode peak_repulsion --num_islands 5 --runs 30 --niching_interval 10 --sigma_niche 0.1 --total_evals 60000
    run_cmd python3 "$ISLAND" --func F3 --mode peak_repulsion --num_islands 4 --runs 30 --niching_interval 5 --sigma_niche 0.5 --total_evals 60000
    run_cmd python3 "$ISLAND" --func F3 --mode peak_repulsion --num_islands 4 --runs 30 --niching_interval 10 --sigma_niche 0.5 --total_evals 60000
  else
    run_cmd python3 "$ISLAND" --func F1 --mode peak_repulsion --num_islands 5 --runs 30 --niching_interval 5 --sigma_niche 0.1
    run_cmd python3 "$ISLAND" --func F2 --mode peak_repulsion --num_islands 5 --runs 30 --niching_interval 5 --sigma_niche 0.1
    run_cmd python3 "$ISLAND" --func F3 --mode peak_repulsion --num_islands 4 --runs 30 --niching_interval 5 --sigma_niche 0.5
  fi
fi

# ---------------------- Island PBIL: peak_walk ----------------------
if python3 "$ISLAND" -h 2>&1 | grep -q "peak_walk"; then
  banner "Island PBIL: mode=peak_walk"

  # Use bigger eval budget for walking/coverage behavior
  if has_flag "$ISLAND" "--total_evals"; then
    run_cmd python3 "$ISLAND" --func F1 --mode peak_walk --num_islands 5 --runs 30 --niching_interval 1 --sigma_niche 0.10 --total_evals 60000
    run_cmd python3 "$ISLAND" --func F2 --mode peak_walk --num_islands 5 --runs 30 --niching_interval 1 --sigma_niche 0.10 --total_evals 60000
    run_cmd python3 "$ISLAND" --func F3 --mode peak_walk --num_islands 4 --runs 30 --niching_interval 1 --sigma_niche 0.50 --total_evals 60000
  else
    run_cmd python3 "$ISLAND" --func F1 --mode peak_walk --num_islands 5 --runs 30 --niching_interval 1 --sigma_niche 0.10
    run_cmd python3 "$ISLAND" --func F2 --mode peak_walk --num_islands 5 --runs 30 --niching_interval 1 --sigma_niche 0.10
    run_cmd python3 "$ISLAND" --func F3 --mode peak_walk --num_islands 4 --runs 30 --niching_interval 1 --sigma_niche 0.50
  fi

  # Optional: compare with/without nudging (if you support --no_nudge or --nudge)
  if [ -n "$NO_NUDGE_FLAG" ]; then
    banner "Island PBIL: peak_walk nudging comparison"
    if has_flag "$ISLAND" "--total_evals"; then
      run_cmd python3 "$ISLAND" --func F2 --mode peak_walk --num_islands 5 --runs 30 --niching_interval 1 --sigma_niche 0.10 --total_evals 60000 $NO_NUDGE_FLAG
    else
      run_cmd python3 "$ISLAND" --func F2 --mode peak_walk --num_islands 5 --runs 30 --niching_interval 1 --sigma_niche 0.10 $NO_NUDGE_FLAG
    fi
  elif [ -n "$NUDGE_FLAG" ]; then
    banner "Island PBIL: peak_walk nudging comparison"
    if has_flag "$ISLAND" "--total_evals"; then
      run_cmd python3 "$ISLAND" --func F2 --mode peak_walk --num_islands 5 --runs 30 --niching_interval 1 --sigma_niche 0.10 --total_evals 60000 $NUDGE_FLAG
    else
      run_cmd python3 "$ISLAND" --func F2 --mode peak_walk --num_islands 5 --runs 30 --niching_interval 1 --sigma_niche 0.10 $NUDGE_FLAG
    fi
  fi

  # Optional: sweep nudge strength if supported
  if [ -n "$NUDGE_STRENGTH_FLAG" ]; then
    banner "Island PBIL: nudge strength sweep (peak_walk)"
    for ns in "${NUDGE_STRENGTHS[@]}"; do
      if has_flag "$ISLAND" "--total_evals"; then
        run_cmd python3 "$ISLAND" --func F2 --mode peak_walk --num_islands 5 --runs 30 --niching_interval 1 --sigma_niche 0.10 --total_evals 60000 $NUDGE_STRENGTH_FLAG "$ns"
      else
        run_cmd python3 "$ISLAND" --func F2 --mode peak_walk --num_islands 5 --runs 30 --niching_interval 1 --sigma_niche 0.10 $NUDGE_STRENGTH_FLAG "$ns"
      fi
    done
  fi
fi


banner "All runs complete"
echo "Check run_GA, run_PBIL, and run_IslandPBIL directories for results."
