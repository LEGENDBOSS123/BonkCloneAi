#!/usr/bin/env bash
# ppo9 3-phase from-scratch pipeline.
#
#   PHASE1  kill the idle bot   -> stops at WR_GATE% winrate vs idle
#   PHASE2  fit a fresh critic  -> stops once the critic loss has CONVERGED
#   PHASE3  full league         -> open-ended
#
# Both gates are PURELY outcome-based: there is NO minimum-step floor, so a
# phase advances the moment it has earned it. The only guard is statistical — a
# gate needs GATE_SAMPLES log lines before it may fire, so it cannot trigger on
# two noisy readings.
#
# Unlike the ppo7 pipeline this never edits source: phases come from `--phase N`
# and tweaks from `--set a.b=value`.
#
# Env knobs: RUN_DIR W E DEV WR_GATE GATE_SAMPLES CONVERGE_TOL SAVE_EVERY POLL
set -uo pipefail
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python

RUN_DIR="${RUN_DIR:-../runs/gang-grounds-2-ppo10}"
W="${W:-8}"
E="${E:-320}"
DEV="${DEV:-mps}"
WR_GATE="${WR_GATE:-85}"                # phase-1 winrate gate, percent
GATE_SAMPLES="${GATE_SAMPLES:-20}"      # samples a gate needs before it may fire
CONVERGE_TOL="${CONVERGE_TOL:-0.02}"    # phase-2: relative drop across the window
SAVE_EVERY="${SAVE_EVERY:-40000000}"
POLL="${POLL:-30}"                      # seconds between gate checks

PLOG="$RUN_DIR/pipeline.log"
mkdir -p "$RUN_DIR"; : > "$PLOG"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$PLOG"; }
sweep(){ ps -axo pid,ppid,command | grep multiprocessing.spawn | grep -v grep \
         | awk '$2==1{print $1}' | xargs -r kill -9 2>/dev/null; }

say "===== ppo9 pipeline ====="
say "run dir       $RUN_DIR"
say "parallelism   $W workers x $E envs = $((W*E)) on $DEV"
say "phase-1 gate  mean winrate >= ${WR_GATE}%          (NO step floor)"
say "phase-2 gate  critic loss flat within ${CONVERGE_TOL} rel  (NO step floor)"
say "gate guard    >= ${GATE_SAMPLES} samples required before either may fire"
say "checkpoints   every $(printf "%'d" $SAVE_EVERY) steps, plus -final on each phase end"

assert_cfg(){
  PYTHONPATH=. $PY -c "
import sys
from ppo9.presets import resolve
c = resolve(3)
sys.exit(0 if ($1) else 1)
" 2>/dev/null || { say "CONFIG ASSERT FAILED: $1"; exit 1; }
  say "  ok   $1"
}
say "--- verifying shared config ---"
assert_cfg "c.env.engine.map_name == 'gang-grounds-2-0'"
assert_cfg "c.env.spawn.use_pool is True"
assert_cfg "c.ppo.mirror == 'sample'"
assert_cfg "c.ppo.gamma == 1.0 and c.critic.categorical is True"
assert_cfg "tuple(c.critic.atoms) == (c.env.rewards.win, c.env.rewards.draw, c.env.rewards.loss)"
assert_cfg "c.state_dim == 129 and c.agent_state_dim == 130"
assert_cfg "len(c.layout.mirror_negate) == 30 and len(c.layout.mirror_swap_a) == 3"
PYTHONPATH=. $PY -m ppo9.train --phase 3 --dump-config > "$RUN_DIR/config-phase3.json" 2>/dev/null
say "resolved phase-3 config -> $RUN_DIR/config-phase3.json"

_cur(){ grep -E "^step " "$1" 2>/dev/null | tail -1; }
_cur_step(){ _cur "$1" | grep -oE "^step [0-9,]+" | tr -dc '0-9'; }
_nlines(){ grep -cE "^step " "$1" 2>/dev/null || echo 0; }
_fmt(){ [ -n "${1:-}" ] && printf "%'d" "$1" || echo "?"; }

_verify_start(){   # <name> <logdir> <pid> <expect-regex>
  local name="$1" logdir="$2" pid="$3" expect="$4" ok=0
  for _ in $(seq 1 80); do
    grep -q "phase flags" "$logdir/train.log" 2>/dev/null && { ok=1; break; }
    kill -0 "$pid" 2>/dev/null || { say "$name DIED at startup:"; tail -25 "$logdir/train.log" | tee -a "$PLOG"; exit 1; }
    sleep 3
  done
  [ $ok -eq 1 ] || { say "$name: no phase-flags line after 240s"; kill "$pid" 2>/dev/null; sweep; exit 1; }
  say "$name flags:  $(grep 'phase flags' "$logdir/train.log" | head -1 | sed 's/^ *//')"
  grep 'phase flags' "$logdir/train.log" | head -1 | grep -qE "$expect" \
    || { say "$name FLAGS MISMATCH — wanted ~ [$expect]"; kill -INT "$pid" 2>/dev/null; sweep; exit 1; }
  say "$name setup:  $(grep -m1 '^ppo9:' "$logdir/train.log")"
}

run_phase(){   # <name> <logdir> <expect> <mode> -- <train args...>
  local name="$1" logdir="$2" expect="$3" mode="$4"; shift 5
  rm -rf "$logdir"; mkdir -p "$logdir"
  local t0=$SECONDS
  say "--- $name START (stop: $mode) -> $logdir"
  $PY -u -m ppo9.train "$@" --out "$logdir" > "$logdir/train.log" 2>&1 &
  local pid=$!
  _verify_start "$name" "$logdir" "$pid" "$expect"

  local ticks=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL"; ticks=$((ticks+1))
    local n step; n=$(_nlines "$logdir/train.log"); step=$(_cur_step "$logdir/train.log")
    [ -z "$step" ] && continue
    case "$mode" in
      wr_gate)
        local wr; wr=$(grep -oE "\| wr [0-9]+%" "$logdir/train.log" | grep -oE "[0-9]+" \
                       | tail -"$GATE_SAMPLES" | awk '{s+=$1;c++} END{if(c) printf "%.1f", s/c}')
        if [ $((ticks % 4)) -eq 0 ]; then
          say "  $name  step $(_fmt "$step")  mean wr ${wr:-?}% / ${WR_GATE}%  (${n} samples)  $(_cur "$logdir/train.log" | grep -oE 'H=[0-9.]+ .*hold=[0-9.]+' | cut -c1-60)"
        fi
        if [ "$n" -ge "$GATE_SAMPLES" ] && [ -n "$wr" ]; then
          awk -v a="$wr" -v b="$WR_GATE" 'BEGIN{exit !(a>=b)}' && {
            say "$name GATE HIT: mean wr ${wr}% >= ${WR_GATE}% over the last $GATE_SAMPLES samples, at $(_fmt "$step") steps"
            kill -INT "$pid"; break; }
        fi ;;
      converge)
        # Compare the older half of the window to the newer half; flat = converged.
        local res; res=$(grep -oE " c=[0-9.]+" "$logdir/train.log" | grep -oE "[0-9.]+" \
          | tail -$((GATE_SAMPLES*2)) | awk -v tol="$CONVERGE_TOL" -v need=$((GATE_SAMPLES*2)) '
              {v[n++]=$1}
              END{ if(n<need){print "wait 0 0 0"; exit}
                   h=int(n/2); for(i=0;i<h;i++) a+=v[i]; for(i=h;i<n;i++) b+=v[i];
                   a/=h; b/=(n-h); r=(a>0)?(a-b)/a:0;
                   printf "%s %.5f %.5f %.4f", (r<tol ? "flat":"falling"), a, b, r }')
        set -- $res
        [ $((ticks % 4)) -eq 0 ] && say "  $name  step $(_fmt "$step")  critic $1  ${2} -> ${3} (rel ${4} vs tol ${CONVERGE_TOL}, ${n} samples)"
        [ "$1" = "flat" ] && {
          say "$name GATE HIT: critic converged (${2} -> ${3}, rel ${4} < ${CONVERGE_TOL}) at $(_fmt "$step") steps"
          kill -INT "$pid"; break; } ;;
      forever)
        [ $((ticks % 10)) -eq 0 ] && say "  $name  $(_cur "$logdir/train.log" | cut -c1-165)" ;;
    esac
  done
  wait "$pid" 2>/dev/null
  sweep
  say "$name ENDED at $(_fmt "$(_cur_step "$logdir/train.log")") steps after $(( (SECONDS-t0)/60 )) min"
  say "  last: $(_cur "$logdir/train.log" | cut -c1-175)"
}

latest_final(){ ls -t "$1"/*final*.json 2>/dev/null | head -1; }

run_phase PHASE1 "$RUN_DIR/p1" \
  "scalar_critic=True dense=True.*all_idle=True freeze_actor=False league=False" \
  wr_gate -- \
  --phase 1 --workers $W --envs-per-worker $E --device $DEV \
  --save-every $SAVE_EVERY --set entropy.coef_start=0.1
P1=$(latest_final "$RUN_DIR/p1")
[ -n "$P1" ] || { say "PHASE1: no final checkpoint — aborting"; exit 1; }
say "phase-1 actor -> $P1"

run_phase PHASE2 "$RUN_DIR/p2" \
  "scalar_critic=False dense=False.*all_idle=False freeze_actor=True league=False" \
  converge -- \
  --phase 2 --workers $W --envs-per-worker $E --device $DEV \
  --save-every $SAVE_EVERY --warm-actor "$P1"
P2=$(latest_final "$RUN_DIR/p2")
[ -n "$P2" ] || { say "PHASE2: no final checkpoint — aborting"; exit 1; }
say "phase-2 actor+critic -> $P2"

# Entropy start drops so unfreezing does not wipe the warm prior. ppo7 achieved
# this by running `sed -i` over its own config.py between phases.
run_phase PHASE3 "$RUN_DIR/p3" \
  "scalar_critic=False dense=False.*all_idle=False freeze_actor=False league=True" \
  forever -- \
  --phase 3 --workers $W --envs-per-worker $E --device $DEV \
  --save-every $SAVE_EVERY --load "$P2" --set entropy.coef_start=0.005

say "===== PIPELINE COMPLETE ====="
say "judge progress with: $PY -m ppo9.eval_h2h --a <new>.json --b <old>.json --games 2000"
