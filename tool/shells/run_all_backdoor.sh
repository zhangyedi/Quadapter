#!/bin/bash
cd ../..

TO=7200

for arch in "1blk_100 2blk_100_100"
do

  for dataset in "mnist fashion-mnist"
  do

    for stampSize in 3 5
    do

      for i in 0 1 2 3 4 5 6 7 8 9
      do

        timeout $TO python Quadapter_backdoor_main.py --dataset $dataset --arch $arch --bit_lb 2 --loc_row 1  --loc_col 1  --stamp_size $stampSize --K 5 --preimg_mode milp --ifRelax 1 --delta 0.05 --targetCls $i --originalCls 10 --outputPath output_backdoor/$dataset/$arch"_largerK_Stamp"$stampSize/
        timeout $TO python Quadapter_backdoor_main.py --dataset $dataset --arch $arch --bit_lb 2 --loc_row 4  --loc_col 20 --stamp_size $stampSize --K 5 --preimg_mode milp --ifRelax 1 --delta 0.05 --targetCls $i --originalCls 10 --outputPath output_backdoor/$dataset/$arch"_largerK_Stamp"$stampSize/
        timeout $TO python Quadapter_backdoor_main.py --dataset $dataset --arch $arch --bit_lb 2 --loc_row 5  --loc_col 5  --stamp_size $stampSize --K 5 --preimg_mode milp --ifRelax 1 --delta 0.05 --targetCls $i --originalCls 10 --outputPath output_backdoor/$dataset/$arch"_largerK_Stamp"$stampSize/
        timeout $TO python Quadapter_backdoor_main.py --dataset $dataset --arch $arch --bit_lb 2 --loc_row 7  --loc_col 13 --stamp_size $stampSize --K 5 --preimg_mode milp --ifRelax 1 --delta 0.05 --targetCls $i --originalCls 10 --outputPath output_backdoor/$dataset/$arch"_largerK_Stamp"$stampSize/
        timeout $TO python Quadapter_backdoor_main.py --dataset $dataset --arch $arch --bit_lb 2 --loc_row 10 --loc_col 10 --stamp_size $stampSize --K 5 --preimg_mode milp --ifRelax 1 --delta 0.05 --targetCls $i --originalCls 10 --outputPath output_backdoor/$dataset/$arch"_largerK_Stamp"$stampSize/

      done

    done

  done

done


