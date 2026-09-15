#!/bin/bash
# Copy the PufferLib 5.0 Splendor env into a PufferLib 5.0 checkout.
#   ./pufferlib5/install.sh /path/to/PufferLib
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 PL5_DIR   (a PufferLib 5.0 checkout, the one with build.sh)"
    exit 1
fi
PL5=$1
if [ ! -f "$PL5/build.sh" ]; then
    echo "Error: $PL5 does not look like a PufferLib 5.0 checkout (no build.sh)"
    exit 1
fi
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$PL5/ocean/splendor" "$PL5/config"
cp "$HERE/splendor.h" "$PL5/ocean/splendor/splendor.h"
cp "$HERE/../splendor/game.h" "$PL5/ocean/splendor/game.h"
cp "$HERE/splendor.ini" "$PL5/config/splendor.ini"
echo "Installed into $PL5:"
echo "  ocean/splendor/splendor.h"
echo "  ocean/splendor/game.h"
echo "  config/splendor.ini"
cat <<'MSG'

Build and run (from the PufferLib 5.0 checkout):

  # Train (needs CUDA + nvcc)
  ./build.sh splendor && ./puffer train splendor

  # Four seats per game: the seat count is compiled in
  NVCC_EXTRA="-DSPLENDOR_PLAYERS=4" ./build.sh splendor puffer4

  # CPU eval / play binary (clang + raylib + libomp, no CUDA)
  ./build.sh splendor --cpu && ./splendor --headless --eval_episodes=100
  ./splendor latest        # watch the newest checkpoints/splendor/*.bin play

On macOS run build.sh with a bash 5 (brew) as `bash ./build.sh ...`:
its ${ENV^^} needs bash 4+, and /bin/bash is 3.2.
MSG
