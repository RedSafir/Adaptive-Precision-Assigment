#!/bin/bash

#======#
# NOTE #
#======#
# - run on: a single master node only (in the current bsub job)
# - run by: ./cifar_perf_main.sh (using bsub).

#===========#
# set: vars #
#===========#
# set: TRAIN_CMDS. [from args]
TRAIN_CMDS="";
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-cmds) TRAIN_CMDS="$2"; shift; shift; ;;        
        --*) echo "Unknown option $1"; exit 1; ;;
    esac
done

# set: CURDIR. [from cur]
CURDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#==========#
# set: CMD #
#==========#
# set: CMD.
CMD="bash $CURDIR/cifar_perf_node.sh \
       --train-cmds '$TRAIN_CMDS'"

#==========#
# run: CMD #
#==========#
# run: CMD.
echo ""
echo "---------------------------------"
echo "[bsub]:" cmd=
echo $CMD
echo "---------------------------------"
echo ""
eval $CMD
echo ""
echo "---------------------------------"
echo ""
