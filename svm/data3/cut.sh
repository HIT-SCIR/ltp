# !bin/bash

train=-2543
test=-500

cat feature.txt | head $train > train.txt
cat feature.txt | tail $test > test.txt
cat raw_data.txt | head $train > train_raw.txt
cat raw_data.txt | tail $test > test_raw.txt
