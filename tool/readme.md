# Quadapter
This is the official webpage for paper *Certified Quantization Strategy Synthesis for Neural Networks*. In this paper, we make the following main contributions:
- We introduce the first quantization strategy synthesis method for neural networks which provably preserves desired properties after quantization;
- WeproposeanovelMILP-basedmethod,tocomputeanunder-approximation of the preimage for each layer efficiently and effectively;
- We implement our methods into a tool Quadapter and conduct extensive experiments to demonstrate the application of the certified quantization for preserving robustness and backdoor-freeness properties.

## Benchmarks in Sections 5.1 & 5.2 & 5.3:

The 50 randomly selected inputs from the test set of the respective dataset (shown by IDs):

```
5346  8564  7059  371   6984  5782  2127  8517  4520  8685
3877  463   5446  7775  9623  5739  5010  7668  892   8825
3523  7997  8561  1613  6934  6781  5554  6301  6220  9873
9384  130   9033  8620  6066  4973  8870  5032  8911  5224
5369  1451  7766  5126  9498  1382  3932  8302  9566  5750
```


## Setup
Please install gurobypy from PyPI:

```shell script
$ pip install gurobipy
```

For MILP-based solving usage, please install Gurobi on your machine.

## Running Quadapter for Certified Robustness
```shell script
# Preimage Computation Mode: MILP-based ('--preimg_mode milp'), Abstr-based ('--preimg_mode abstr')
# If relaxed version of Quadapter: yes ('--if_relax 1'), no ('--if_relax 0')
# Input=5346, Attack=2, preimg_mode=milp, OutputFolder=./output/

python Quadapter_robustness_main.py --dataset mnist --arch 1blk_100 --sample_id 5346 --eps 2 --preimg_mode milp --if_relax 0 --outputPath ./output/
```

### Running Quadapter for Certified Backdoor-freeness
```shell script
# Backdoor Info:  --loc_row 1  --loc_col 1 --stamp_size 3 --targetCls 8 --originalCls 10
# Paras for Hypothesis Testing: --K 5 --delta 0.05

python Quadapter_backdoor_main.py --dataset mnist --arch 1blk_100 --bit_lb 2 --loc_row 1  --loc_col 1  --stamp_size 3 --targetCls 8 --originalCls 10 --K 5 --delta 0.05 --preimg_mode milp --ifRelax 1  --outputPath ./output/
```