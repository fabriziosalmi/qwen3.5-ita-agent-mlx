#!/bin/sh
# Runs the heavy work on another Mac (Apple Silicon, with uv installed), keeping this machine free.
#
#   QITA_HOST=<ssh host> scripts/remote.sh sync              copy code and eval sets there, then uv sync
#   QITA_HOST=<ssh host> scripts/remote.sh run <command>     run a command in the remote project (foreground)
#   QITA_HOST=<ssh host> scripts/remote.sh bg <name> <cmd>   run in the background, log in runs/<name>.log
#   QITA_HOST=<ssh host> scripts/remote.sh tail <name>       last lines of runs/<name>.log
#   QITA_HOST=<ssh host> scripts/remote.sh pull              copy the remote runs/ here (all of it)
#
# QITA_HOST is an ssh destination (alias, name or address). QITA_DIR is the project folder there, relative to
# the remote home (default: Documents/git/<this folder's name>). With bg, <cmd> runs inside single quotes on
# the remote side: it must not contain a single quote.
set -e
HOST="${QITA_HOST:?set QITA_HOST to an ssh destination}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${QITA_DIR:-Documents/git/$(basename "$ROOT")}"
ON_REMOTE="cd ~/$DIR && export PATH=\$HOME/.local/bin:\$PATH &&"
cmd="${1:-}"
[ $# -gt 0 ] && shift

case "$cmd" in
    sync)
        ssh "$HOST" "mkdir -p ~/$DIR/runs"
        rsync -a --exclude .venv --exclude '.venv-*' --exclude .pytest_cache --exclude .git --exclude /runs --exclude __pycache__ --exclude .DS_Store "$ROOT/" "$HOST:$DIR/"
        ssh "$HOST" "$ON_REMOTE uv sync -q" ;;
    run)
        ssh "$HOST" "$ON_REMOTE $*" ;;
    bg)
        name="$1"; shift
        ssh "$HOST" "$ON_REMOTE mkdir -p runs && nohup caffeinate -i -s sh -c '$*' > runs/$name.log 2>&1 < /dev/null & echo started $name" ;;
    tail)
        ssh "$HOST" "tail -${2:-20} ~/$DIR/runs/$1.log" ;;
    pull)
        mkdir -p "$ROOT/runs"
        rsync -a "$HOST:$DIR/runs/" "$ROOT/runs/" ;;
    *)  sed -n '2,14p' "$0"; exit 1 ;;
esac
