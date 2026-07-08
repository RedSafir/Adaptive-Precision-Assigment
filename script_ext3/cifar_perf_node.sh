#!/bin/bash

#======#
# NOTE #
#======#
# - run on: every slave node (in the current bsub job).
# - run by: ./cifar_perf_bsub.sh (using lrun).
# - for quick test: nothing required.

#===========#
# set: vars #
#===========#
# set: NNODES, TRAIN_CMDS. [from args]
NNODES=1; TRAIN_CMDS="";
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-cmds) TRAIN_CMDS="$2";  shift; shift; ;;        
        --*) echo "Unknown option $1"; exit 1; ;;
    esac
done

# set: RANK, HOST, DATE. [from cur]
if [ -n "$LSB_JOBID" ]; then
    # for bsub.
    RANK=$OMPI_COMM_WORLD_RANK
else
    RANK=-1
fi
HOST=$(hostname)
DATE=$(date)

#==========#
# set: CMD #
#==========#
# set: CMD.
CMD="$TRAIN_CMDS"

#==========#
# run: CMD #
#==========#
# untar data.
echo "[node=$RANK/$NNODES]:" host=$HOST, date=$DATE
:

# run train.
echo "[node=$RANK/$NNODES]:" cmd=
echo $CMD; echo ""
eval $CMD

# remove data.
:
