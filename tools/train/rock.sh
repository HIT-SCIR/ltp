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
CWS_MODEL_DIR=$BUILD_DIR/cws
CWS_MODEL_PATH=$CWS_MODEL_DIR/example-seg.0.model

CWS_CONF_DIR=$CONF_DIR/cws
CWS_CONF_TRAIN_PATH=$CWS_CONF_DIR/cws.cnf

CWS_LOG_DIR=$LOG_DIR/cws
CWS_LOG_TRAIN_PATH=$CWS_LOG_DIR/example-seg.train.log

CWS_EXE=./otcws

# create cws output dirs
mkdir -p $CWS_MODEL_DIR
mkdir -p $CWS_LOG_DIR

# execute the example training process 
$CWS_EXE $CWS_CONF_TRAIN_PATH >& $CWS_LOG_TRAIN_PATH

if [ ! -f $CWS_MODEL_PATH ]; then 
    echo "[1] ERROR: CWS model is not detected!"
else
    echo "[1] TRACE: CWS train model test is passed."
fi

#################################################
# THE POSTAG SESSION                            #
#################################################

POS_MODEL_DIR=$BUILD_DIR/pos
POS_MODEL_PATH=$POS_MODEL_DIR/example-pos.0.model

POS_CONF_DIR=$CONF_DIR/pos
POS_CONF_TRAIN_PATH=$POS_CONF_DIR/pos.cnf

POS_LOG_DIR=$LOG_DIR/pos
POS_LOG_TRAIN_PATH=$CWS_LOG_DIR/example-pos.train.log

POS_EXE=./otpos

# create pos output dirs
mkdir -p $POS_MODEL_DIR
mkdir -p $POS_LOG_DIR

$POS_EXE $POS_CONF_TRAIN_PATH >& $POS_LOG_TRAIN_PATH

if [ ! -f $CWS_MODEL_PATH ]; then 
    echo "[2] ERROR: POS model is not detected!"
else
    echo "[2] TRACE: POS train model test is passed."
fi

#################################################
# THE NER SESSION                               #
#################################################

# ner dir config
NER_MODEL_DIR=$BUILD_DIR/ner
NER_MODEL_PATH=$NER_MODEL_DIR/example-ner.0.model

NER_CONF_DIR=$CONF_DIR/ner
NER_CONF_TRAIN_PATH=$NER_CONF_DIR/ner.cnf

NER_LOG_DIR=$LOG_DIR/ner
NER_LOG_TRAIN_PATH=$NER_LOG_DIR/example-ner.train.log

NER_EXE=./otner

# create cws output dirs
mkdir -p $NER_MODEL_DIR
mkdir -p $NER_LOG_DIR

# execute the example training process 
$NER_EXE $NER_CONF_TRAIN_PATH >& $NER_LOG_TRAIN_PATH

if [ ! -f $NER_MODEL_PATH ]; then 
    echo "[3] ERROR: NER model is not detected!"
else
    echo "[3] TRACE: NER train model test is passed."
fi


