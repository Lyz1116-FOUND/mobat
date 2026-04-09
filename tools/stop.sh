#!/bin/sh
set -eu

cd /data/projects/gorge_chase
exec /data/projects/gorge_chase/tools/stop.sh "$@"
