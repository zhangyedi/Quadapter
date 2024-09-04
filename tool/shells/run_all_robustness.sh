#!/bin/bash
cd ../..

TO=7200

benchmark='5346  8564  7059  371   6984  5782  2127  8517  4520  8685
           3877  463   5446  7775  9623  5739  5010  7668  892   8825
           3523  7997  8561  1613  6934  6781  5554  6301  6220  9873
           9384  130   9033  8620  6066  4973  8870  5032  8911  5224
           5369  1451  7766  5126  9498  1382  3932  8302  9566  5750'

for arch in "1blk_100 2blk_100_100 3blk_100_100_100 2blk_512_512"
do

  for dataset in "mnist fashion-mnist"
  do

    for i in $benchmark
    do

      for eps in 1 2 3 4 5
      do

        timeout $TO python Quadapter_robustness_main.py --dataset $dataset --arch $arch --sample_id $i --eps $eps --preimg_mode milp --if_relax 1 --bit_lb 1 --bit_ub 16 --outputPath output_robustness/$dataset/$arch/eps_$eps/
        timeout $TO python Quadapter_robustness_main.py --dataset $dataset --arch $arch --sample_id $i --eps $eps --preimg_mode abstr --if_relax 1 --bit_lb 1 --bit_ub 16 --outputPath output_robustness/$dataset/$arch/eps_$eps/

      done

    done

  done

done


