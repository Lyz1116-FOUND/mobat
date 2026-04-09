#!/bin/sh
set -eu

cd /data/projects/gorge_chase
if [ -x /data/projects/gorge_chase/tools/stop_sim_python.sh ]; then
    exec /data/projects/gorge_chase/tools/stop_sim_python.sh "$@"
fi

# Some container images do not ship this helper. Keep cleanup best-effort.
exec /data/projects/gorge_chase/tools/stop.sh train_test
