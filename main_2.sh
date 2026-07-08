#!/bin/bash

# Figure 5, 16.
bash script_ext3/cifar_perf_main_2_heuris.sh --qtype batch --data cifar100 --model-wm 1.00 --schm topr_inc --seed -1
bash script_ext3/cifar_perf_main_2_heuris.sh --qtype batch --data cifar100 --model-wm 1.00 --schm rand     --seed 123
bash script_ext3/cifar_perf_main_2_heuris.sh --qtype batch --data cifar100 --model-wm 1.00 --schm rand     --seed 456
bash script_ext3/cifar_perf_main_2_heuris.sh --qtype batch --data cifar100 --model-wm 1.00 --schm rand     --seed 789
