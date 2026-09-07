#!/usr/bin/env bash
# Build the Rust bonk engine that bonk3 binds to.
#
# The crate depends on `box2dweb-rs`, a PRIVATE git submodule of the bonk-io
# org. You need access to that org for this to work; if you authenticate to
# GitHub over HTTPS rather than SSH, run once:
#     git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$HERE/../../bonk-recreation/bonk-enviroment"
CRATE="$ENV_DIR/rust/bonk-env"

if [ ! -f "$ENV_DIR/rust/box2dweb-rs/Cargo.toml" ]; then
    echo "==> fetching the box2dweb-rs submodule"
    git -C "$ENV_DIR" submodule update --init --recursive rust/box2dweb-rs
fi

echo "==> cargo build --release"
cargo build --release --manifest-path "$CRATE/Cargo.toml"

LIB="$CRATE/target/release/libbonk_env.dylib"
[ -f "$LIB" ] || LIB="$CRATE/target/release/libbonk_env.so"
echo "==> built $LIB"

echo "==> smoke test"
PYTHONPATH="$HERE/.." python3 -c "
from bonk3.engine import BonkEngine
from bonk3.maps import available, load_map
maps = available()
if not maps:
    print('no decoded maps yet — run:')
    print('  node python/bonk3/tools/decode_maps.mjs --name \"gang grounds 2.0\"')
else:
    e = BonkEngine(load_map(maps[0]))
    s = e.reset(seed=1.0)
    print(f'  {maps[0]}: {len(s.discs)} discs, p0 at ({s.discs[0].x:.2f}, {s.discs[0].y:.2f})')
    for _ in range(30):
        s = e.step([{'right': True}, None])
    print(f'  after 30 frames: p0 x={s.discs[0].x:.3f} vx={s.discs[0].vx:.3f}')
    print('  OK')
"
