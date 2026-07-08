#!/bin/bash

#======#
# NOTE #
#======#
# - run on: the login node only (not in any bsub job).
# - run by:
#   - actual train:
#       bash script_ext3/cifar_perf_main_1_perf.sh --qtype batch --data cifar10  --model-wm 1.00
#       bash script_ext3/cifar_perf_main_1_perf.sh --qtype batch --data cifar100 --model-wm 1.00
#       bash script_ext3/cifar_perf_main_1_perf.sh --qtype batch --data cifar100 --model-wm 0.50
#       bash script_ext3/cifar_perf_main_1_perf.sh --qtype batch --data cifar100 --model-wm 0.25
#       bash script_ext3/cifar_perf_main_1_perf.sh --qtype batch --data cifar100 --model-wm 0.10
#   - only print cmd:
#       bash script_ext3/cifar_perf_main_1_perf.sh --qtype debug --data cifar10  --model-wm 1.00
#       bash script_ext3/cifar_perf_main_1_perf.sh --qtype debug --data cifar100 --model-wm 1.00
#       bash script_ext3/cifar_perf_main_1_perf.sh --qtype debug --data cifar100 --model-wm 0.50
#       bash script_ext3/cifar_perf_main_1_perf.sh --qtype debug --data cifar100 --model-wm 0.25
#       bash script_ext3/cifar_perf_main_1_perf.sh --qtype debug --data cifar100 --model-wm 0.10
# - config: change the parts surrounded by ----CONFIG---- below.

#===========#
# set: vars #
#===========#
# set: QTYPE, DATA, MODEL_WM. [from args]
QTYPE=""; DATA=""; MODEL_WM="";
while [[ $# -gt 0 ]]; do
    case $1 in
        --qtype)     QTYPE="$2";     shift; shift; ;;        
        --data)      DATA="$2";      shift; shift; ;;        
        --model-wm)  MODEL_WM="$2";  shift; shift; ;;        
        --*) echo "Unknown option $1"; exit 1; ;;
    esac
done

# set: CURDIR. [from cur]
CURDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) #dirname "$0"

#===========#
# set: CMDS #
#===========#
# set: CMD_MAIN.
#------------------------CONFIG------------------------
bsub_nnodes=1
bsub_w=360
bsub_o="./res/y-log-bsub/output_pbatch_"$DATA"_1_perf_%J.txt"
bsub_q="pbatch"
# bsub_stage="storage=200"
#------------------------------------------------------
CMD_MAIN="bash $CURDIR/cifar_perf_bsub.sh "

# set: CMD_ARGS.
inout_test="     --data-path ./dataset/ --workers 2 --quiet --logg-undovr-freq -1 --meas-no-fperr"
model_wm="       --model-wm $MODEL_WM"
opt_256="        --ep 200 --bs-tr 256 --lr 0.20 --mt 0.9 --wd 5e-4 --lr-schd cos" # for tot_batch_sz=256.
opt_128="        --ep 200 --bs-tr 128 --lr 0.10 --mt 0.9 --wd 5e-4 --lr-schd cos" # for tot_batch_sz=128.
lsc_none="       --lsc-init 1     --lsc-evry-nep -1"
lsc_dyn="        --lsc-init 65536 --lsc-evry-nep  1"
lsc_stc="        --lsc-init 1024  --lsc-evry-nep -1"
pa_upd_fp32="    --pa-def-dt FP32     FP32"
pa_upd_std="     --pa-def-dt FP_6_9_0 FP_6_9_0 --pa-upd-schm std      --pa-upd-dts FP_4_3_4 FP_4_3_4 FP_5_2_0 FP_6_9_0 --pa-upd-no-end" # pi_op.
pa_upd_ste="     --pa-def-dt FP_6_9_0 FP_6_9_0 --pa-upd-schm ste      --pa-upd-dts FP_4_3_4 FP_4_3_4 FP_5_2_0 FP_6_9_0 --pa-upd-no-end" # pi_op'.
pa_upd_topr_dec="--pa-def-dt FP_6_9_0 FP_6_9_0 --pa-upd-schm topr_dec --pa-upd-dts FP_4_3_4 FP_4_3_4 FP_5_2_0 FP_6_9_0" # pi_ours.
pa_ovr_thrs_050="    --pa-ovr-thrs 0.5"
pa_ovr_thrs_025="    --pa-ovr-thrs 0.25"
pa_ovr_thrs_010="    --pa-ovr-thrs 0.1"
pa_ovr_thrs_001="    --pa-ovr-thrs 0.01"
pa_ovr_thrs_0001="   --pa-ovr-thrs 0.001"
pa_ovr_thrs_00001="  --pa-ovr-thrs 0.0001"
#------------------------CONFIG------------------------
data_name=$DATA
inout="$inout_test"
# opt="$opt_256"
opt="$opt_128"
# pa_ovr_thrs="" # no ovr_thrs.
# pa_ovr_thrs="$pa_ovr_thrs_050"
# pa_ovr_thrs="$pa_ovr_thrs_025"
# pa_ovr_thrs="$pa_ovr_thrs_010"
pa_ovr_thrs="$pa_ovr_thrs_001"
# pa_ovr_thrs="$pa_ovr_thrs_0001"
# pa_ovr_thrs="$pa_ovr_thrs_00001"
CMD_ARGS=(
    #
    # spec: array of (DATA_NAME, TRAIN_ARGS).
    #
    # sqz.
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_none $pa_upd_fp32" # pi_unif: FP32.
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_std"  # pi_op  : FP16+FP8 (HFP8).
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_ste"  # pi_op' : FP16+FP8 (torch.amp).
    # "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.0 0.0" # pi_unif: FP16.
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 1.0 1.0" # pi_unif: FP8.
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.0 0.0 $pa_ovr_thrs"
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.1 0.2 $pa_ovr_thrs"
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.2 0.3 $pa_ovr_thrs"
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.3 0.4 $pa_ovr_thrs"
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.4 0.5 $pa_ovr_thrs"
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.5 0.6 $pa_ovr_thrs"
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.6 0.7 $pa_ovr_thrs"
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.7 0.8 $pa_ovr_thrs"
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.8 0.9 $pa_ovr_thrs"
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.9 1.0 $pa_ovr_thrs"
    "$data_name" "$inout --model sqz   $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 1.0 1.1 $pa_ovr_thrs"
    # shfv2.
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_none $pa_upd_fp32"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_std"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_ste"
    # "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.0 0.0"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 1.0 1.0"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.0 0.0 $pa_ovr_thrs"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.1 0.2 $pa_ovr_thrs"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.2 0.3 $pa_ovr_thrs"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.3 0.4 $pa_ovr_thrs"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.4 0.5 $pa_ovr_thrs"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.5 0.6 $pa_ovr_thrs"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.6 0.7 $pa_ovr_thrs"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.7 0.8 $pa_ovr_thrs"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.8 0.9 $pa_ovr_thrs"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.9 1.0 $pa_ovr_thrs"
    "$data_name" "$inout --model shfv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 1.0 1.1 $pa_ovr_thrs"
    # mblv2.
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_none $pa_upd_fp32"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_std"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_ste"
    # "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.0 0.0"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 1.0 1.0"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.0 0.0 $pa_ovr_thrs"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.1 0.2 $pa_ovr_thrs"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.2 0.3 $pa_ovr_thrs"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.3 0.4 $pa_ovr_thrs"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.4 0.5 $pa_ovr_thrs"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.5 0.6 $pa_ovr_thrs"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.6 0.7 $pa_ovr_thrs"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.7 0.8 $pa_ovr_thrs"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.8 0.9 $pa_ovr_thrs"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.9 1.0 $pa_ovr_thrs"
    "$data_name" "$inout --model mblv2 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 1.0 1.1 $pa_ovr_thrs"
    # res18.
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_none  $pa_upd_fp32"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_std"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_ste"
    # "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.0 0.0"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 1.0 1.0"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.0 0.0 $pa_ovr_thrs"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.1 0.2 $pa_ovr_thrs"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.2 0.3 $pa_ovr_thrs"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.3 0.4 $pa_ovr_thrs"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.4 0.5 $pa_ovr_thrs"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.5 0.6 $pa_ovr_thrs"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.6 0.7 $pa_ovr_thrs"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.7 0.8 $pa_ovr_thrs"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.8 0.9 $pa_ovr_thrs"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 0.9 1.0 $pa_ovr_thrs"
    "$data_name" "$inout --model res18 $model_wm $opt $lsc_dyn  $pa_upd_topr_dec --pa-upd-args 1.0 1.1 $pa_ovr_thrs"
)
#------------------------------------------------------

# set: CMDS.
CMDS=()
cur_ind=0; cur_cmds=":"; 
for ((i=0; i<${#CMD_ARGS[@]}; i+=2)); do
    # add new_cmd to cur_cmds.
    i0=$i; i1=$(($i+1));
    new_cmd="python3 run_train_ext3.py --cuda $cur_ind --data ${CMD_ARGS[$i0]} ${CMD_ARGS[$i1]}"
    cur_ind=$((cur_ind+1)); cur_cmds="$cur_cmds & $new_cmd"

    # add cur_cmds to CMDS.
    if (( $cur_ind == 1 )); then
        cur_cmds="$cur_cmds"
        CMD="$CMD_MAIN --train-cmds '$cur_cmds'"
        CMDS+=("$CMD")
        cur_ind=0; cur_cmds=":"
    fi
done
# add remaining cur_cmds to CMDS.
if (( $cur_ind > 0 )); then
    cur_cmds="$cur_cmds"
    CMD="$CMD_MAIN --train-cmds '$cur_cmds'"
    CMDS+=("$CMD")
    cur_ind=0; cur_cmds=":"
fi

#===========#
# run: CMDS #
#===========#
# run.
echo ""
echo "------------------------------"
echo "[main]: qtype=$QTYPE"
echo "------------------------------"
for ((i=0; i<${#CMDS[@]}; i++)); do
    CMD="${CMDS[$i]}"
    echo "[main]:" cmd=
    echo $CMD
    if [ "$QTYPE" == "batch" ]; then
        eval $CMD
    fi
    echo "------------------------------"
done
