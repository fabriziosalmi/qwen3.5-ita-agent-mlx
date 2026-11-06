#!/bin/sh
# Run a command on two Macs as one MLX ring (rank 0 and rank 1), started from this machine.
#
#   QITA_RANK0=<ssh host> QITA_RANK1=<ssh host> scripts/ring.sh '<cmd>'
#       attached: output here, ends when the command ends
#   QITA_RANK0=... QITA_RANK1=... scripts/ring.sh --bg NAME '<cmd>'
#       detached on both machines (nohup + caffeinate), survives this machine sleeping;
#       logs in runs/NAME.rank{0,1}.log on each machine
#
# QITA_RING0 and QITA_RING1 are the addresses the ranks use to reach each other (default: the ssh hosts).
# For an attached run, LAN addresses are faster. Detached processes on macOS could not open LAN connections
# in our measurements (local network privacy: they time out), so for --bg use addresses that are not on the
# LAN, e.g. Tailscale ones: a 65 MB all_sum took 2.8 s that way against 0.58 s on the LAN.
# QITA_DIR is the project folder on both machines, relative to the remote home.
# <cmd> runs inside single quotes on the remote side: it must not contain a single quote.
set -e
R0="${QITA_RANK0:?set QITA_RANK0 to the ssh host of rank 0}"
R1="${QITA_RANK1:?set QITA_RANK1 to the ssh host of rank 1}"
A0="${QITA_RING0:-$R0}"
A1="${QITA_RING1:-$R1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${QITA_DIR:-Documents/git/$(basename "$ROOT")}"
BG=""
PORT=41550
if [ "$1" = "--bg" ]; then
  BG="$2"; shift 2
  PORT=41560
fi
HOSTFILE="[[\"$A0:$PORT\"], [\"$A1:$((PORT + 1))\"]]"
prep="cd ~/$DIR && export PATH=\$HOME/.local/bin:\$PATH NO_PROXY='*' no_proxy='*' && echo '$HOSTFILE' > /tmp/qita_ring.json && mkdir -p runs && export MLX_HOSTFILE=/tmp/qita_ring.json"
run() {  # rank ssh_host command (exported, so every command of <cmd> sees the rank, not only the first)
  if [ -n "$BG" ]; then
    ssh "$2" "$prep && export MLX_RANK=$1 && nohup caffeinate -i -s sh -c '$3' > runs/$BG.rank$1.log 2>&1 < /dev/null & echo started rank $1"
  else
    ssh "$2" "$prep && export MLX_RANK=$1 && $3"
  fi
}
if [ -n "$BG" ]; then
  # Start both ranks at the same moment: a rank that dials a port not yet open waits for a
  # timeout (macOS drops the packets) instead of retrying.
  run 1 "$R1" "$1" &
  run 0 "$R0" "$1"
  wait
  exit 0
fi
run 1 "$R1" "$1" > /tmp/qita_ring_rank1.log 2>&1 &
run 0 "$R0" "$1"
wait
echo "--- rank 1:"; tail -5 /tmp/qita_ring_rank1.log
