#!/bin/sh

SCRIPT_DIR=`pwd`
ROOT_DIR=$SCRIPT_DIR/..
BIN_DIR=$ROOT_DIR/bin
DATA_DIR=$ROOT_DIR/test_data

EXE=./ltp_test
OPT=all

cd $BIN_DIR

echo "[Running]";
$EXE $OPT $DATA_DIR/test_gb.txt > $DATA_DIR/test_gb.tmp
echo "[Finished]";

cd $DATA_DIR
diff $DATA_DIR/test_gb.xml $DATA_DIR/test_gb.tmp

if [ $? == 0 ] then
    echo "[Failed] found diff between two version's output"
else
    echo "[Passed] no diff"
fi
