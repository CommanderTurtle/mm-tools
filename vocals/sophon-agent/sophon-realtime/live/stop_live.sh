#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; LIVE_PORT=${LIVE_PORT:-8780}
[ -f logs/live_$LIVE_PORT.pid ] && { kill -- -$(ps -o pgid= -p $(cat logs/live_$LIVE_PORT.pid) | tr -d ' ') 2>/dev/null || kill $(cat logs/live_$LIVE_PORT.pid) 2>/dev/null; rm -f logs/live_$LIVE_PORT.pid; echo stopped; }
