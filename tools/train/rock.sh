#!/bin/sh

#################################################
# THE GLOBAL SESSION                            #
#################################################

ROOT=
BUILD_DIR=build
CONF_DIR=conf
LOG_DIR=log

# create model output folder
if [ -d $BUILD_DIR ]; then 
    rm -rf $BUILD_DIR
fi

mkdir -p $BUILD_DIR

if [ -d $LOG_DIR ]; then
    rm -rf $LOG_DIR
fi

mkdir -p $LOG_DIR

#################################################
# THE SEGMENT SESSION                           #
#################################################

# cws dir config
CWS_MODEL_PATH=$BUILD_DIR/cws.model
CWS_LOG_TRAIN_PATH=$LOG_DIR/cws.train.log

CWS_EXE=./otcws

# execute the example training process 
$CWS_EXE learn --model $CWS_MODEL_PATH \
    --reference sample/seg/example-train.seg \
    --development sample/seg/example-holdout.seg \
    --max-iter 2

if [ ! -f $CWS_MODEL_PATH ]; then
    echo "[1] ERROR: CWS model is not detected!"
else
    echo "[1] TRACE: CWS train model test is passed."
fi

#################################################
# THE POSTAG SESSION                            #
#################################################

POS_MODEL_PATH=$BUILD_DIR/pos.model
POS_LOG_TRAIN_PATH=$LOG_DIR/pos.train.log

POS_EXE=./otpos

$POS_EXE learn --model $POS_MODEL_PATH \
    --reference sample/pos/example-train.pos \
    --development sample/pos/example-holdout.pos \
    --max-iter 2

if [ ! -f $CWS_MODEL_PATH ]; then 
    echo "[2] ERROR: POS model is not detected!"
else
    echo "[2] TRACE: POS train model test is passed."
fi

#################################################
# THE NER SESSION                               #
#################################################

# ner dir config
NER_MODEL_PATH=$BUILD_DIR/ner.model
NER_LOG_TRAIN_PATH=$LOG_DIR/ner.train.log

NER_EXE=./otner

# execute the example training process 
$NER_EXE learn --model $NER_MODEL_PATH \
    --reference sample/ner/example-train.ner \
    --development sample/ner/example-holdout.ner \
    --max-iter 2

if [ ! -f $NER_MODEL_PATH ]; then 
    echo "[3] ERROR: NER model is not detected!"
else
    echo "[3] TRACE: NER train model test is passed."
fi

#################################################
# THE PARSER O1 SESSION                         #
#################################################

PARSER_MODEL_PATH=$BUILD_DIR/parser.model
PARSER_LOG_TRAIN_PATH=$LOG_DIR/parser.train.log

PARSER_EXE=./nndepparser

./nndepparser learn \
    --model $PARSER_MODEL_PATH \
    --reference sample/parser/example-train.conll \
    --development sample/parser/example-holdout.conll \
    --embedding sample/parser/example.bin \
    --root HED \
    --max-iter 100

if [ ! -f $PARSER_MODEL_O2SIB_PATH ]; then 
    echo "[4] ERROR: neural network parser model is not detected!"
else
    echo "[4] TRACE: neural network parser model test is passed."
fi

#################################################
# THE SRL-PRG SESSION                           #
#################################################

SRL_PRG_MODEL_DIR=$BUILD_DIR/srl
SRL_PRG_MODEL_PATH=$SRL_PRG_MODEL_DIR/prg.model
SRL_PRG_INSTANCE_DIR=$SRL_PRG_MODEL_DIR/prg-instances.train

SRL_PRG_CONF_DIR=$CONF_DIR/srl
SRL_PRG_CONF_TRAIN_PATH=$SRL_PRG_CONF_DIR/srl-prg.cnf

SRL_PRG_LOG_DIR=$LOG_DIR/srl
SRL_PRG_LOG_TRAIN_PATH=$SRL_PRG_LOG_DIR/example-prg.train.log

mkdir -p $SRL_PRG_MODEL_DIR
mkdir -p $SRL_PRG_LOG_DIR
mkdir -p $SRL_PRG_INSTANCE_DIR

SRL_PRG_EXE=./lgsrl

$SRL_PRG_EXE $SRL_PRG_CONF_TRAIN_PATH

if [ ! -f $SRL_PRG_MODEL_PATH ]; then
    echo "[5.1] ERROR: PRG model is not detected!"
else
    echo "[5.1] TRACE: PRG train model test is passed."
fi

#################################################
# THE SRL-SRL SESSION                           #
#################################################

SRL_SRL_MODEL_DIR=$BUILD_DIR/srl
SRL_SRL_MODEL_PATH=$SRL_SRL_MODEL_DIR/srl.model
SRL_SRL_FEATURES_DIR=$SRL_SRL_MODEL_DIR/srl-features.train
SRL_SRL_INSTANCE_DIR=$SRL_SRL_MODEL_DIR/srl-instances.train

SRL_SRL_CONF_DIR=$CONF_DIR/srl
SRL_SRL_CONF_TRAIN_PATH=$SRL_SRL_CONF_DIR/srl-srl.cnf

SRL_SRL_LOG_DIR=$LOG_DIR/srl
SRL_SRL_LOG_TRAIN_PATH=$SRL_SRL_LOG_DIR/example-srl.train.log

mkdir -p $SRL_SRL_MODEL_DIR
mkdir -p $SRL_SRL_LOG_DIR
mkdir -p $SRL_SRL_FEATURES_DIR
mkdir -p $SRL_SRL_INSTANCE_DIR

SRL_SRL_EXE=./lgsrl

$SRL_SRL_EXE $SRL_SRL_CONF_TRAIN_PATH

if [ ! -f $SRL_SRL_MODEL_PATH ]; then
    echo "[5.2] ERROR: SRL model is not detected!"
else
    echo "[5.2] TRACE: SRL train model test is passed."
fi

