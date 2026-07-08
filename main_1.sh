#!/bin/bash

# Figure 4, 10, 11, 12.
bash script_ext3/cifar_perf_main_1_perf.sh --qtype batch --data cifar10  --model-wm 1.00
bash script_ext3/cifar_perf_main_1_perf.sh --qtype batch --data cifar100 --model-wm 1.00
bash script_ext3/cifar_perf_main_1_perf.sh --qtype batch --data cifar100 --model-wm 0.50
bash script_ext3/cifar_perf_main_1_perf.sh --qtype batch --data cifar100 --model-wm 0.25
bash script_ext3/cifar_perf_main_1_perf.sh --qtype batch --data cifar100 --model-wm 0.10
