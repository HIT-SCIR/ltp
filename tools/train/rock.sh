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

#################################################
# THE PARSER O1 SESSION                         #
#################################################

PARSER_MODEL_DIR=$BUILD_DIR/parser
PARSER_MODEL_O1_PATH=$PARSER_MODEL_DIR/example-parser-o1.0.model

PARSER_CONF_DIR=$CONF_DIR/parser
PARSER_CONF_TRAIN_O1_PATH=$PARSER_CONF_DIR/parser-o1.cnf

PARSER_LOG_DIR=$LOG_DIR/parser
PARSER_LOG_TRAIN_O1_PATH=$PARSER_LOG_DIR/example-train.conll

PARSER_EXE=./lgdpj

mkdir -p $PARSER_MODEL_DIR
mkdir -p $PARSER_LOG_DIR

$PARSER_EXE $PARSER_CONF_TRAIN_O1_PATH >& $PARSER_LOG_TRAIN_O1_PATH

if [ ! -f $PARSER_MODEL_O1_PATH ]; then 
    echo "[4.1] ERROR: Parser-o1 model is not detected!"
else
    echo "[4.1] TRACE: Parser-o1 train model test is passed."
fi

#################################################
# THE PARSER O2 sibling SESSION                 #
#################################################

PARSER_MODEL_DIR=$BUILD_DIR/parser
PARSER_MODEL_O2SIB_PATH=$PARSER_MODEL_DIR/example-parser-o2sib.0.model

PARSER_CONF_DIR=$CONF_DIR/parser
PARSER_CONF_TRAIN_O2SIB_PATH=$PARSER_CONF_DIR/parser-o2sib.cnf

PARSER_LOG_DIR=$LOG_DIR/parser
PARSER_LOG_TRAIN_O2SIB_PATH=$PARSER_LOG_DIR/example-train.conll

PARSER_EXE=./lgdpj

$PARSER_EXE $PARSER_CONF_TRAIN_O2SIB_PATH >& $PARSER_LOG_TRAIN_O2SIB_PATH

if [ ! -f $PARSER_MODEL_O2SIB_PATH ]; then 
    echo "[4.2] ERROR: Parser-o2sib model is not detected!"
else
    echo "[4.2] TRACE: Parser-o2sib train model test is passed."
fi

#################################################
# THE PARSER O2 carreras SESSION                #
#################################################

PARSER_MODEL_DIR=$BUILD_DIR/parser
PARSER_MODEL_O2CARRERAS_PATH=$PARSER_MODEL_DIR/example-parser-o2carreras.0.model

PARSER_CONF_DIR=$CONF_DIR/parser
PARSER_CONF_TRAIN_O2CARRERAS_PATH=$PARSER_CONF_DIR/parser-o2carreras.cnf

PARSER_LOG_DIR=$LOG_DIR/parser
PARSER_LOG_TRAIN_O2CARRERAS_PATH=$PARSER_LOG_DIR/example-train.conll

PARSER_EXE=./lgdpj

$PARSER_EXE $PARSER_CONF_TRAIN_O2CARRERAS_PATH >& $PARSER_LOG_TRAIN_O2CARRERAS_PATH

if [ ! -f $PARSER_MODEL_O2SIB_PATH ]; then 
    echo "[4.3] ERROR: Parser-o2carreras model is not detected!"
else
    echo "[4.3] TRACE: Parser-o2carreras train model test is passed."
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

$SRL_PRG_EXE $SRL_PRG_CONF_TRAIN_PATH >& $SRL_PRG_LOG_TRAIN_PATH

if [ ! -f $SRL_PRG_MODEL_PATH ]; then
    echo "[1] ERROR: SRL model is not detected!"
else
    echo "[1] TRACE: SRL train model test is passed."
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

$SRL_SRL_EXE $SRL_SRL_CONF_TRAIN_PATH >& $SRL_SRL_LOG_TRAIN_PATH

if [ ! -f $SRL_SRL_MODEL_PATH ]; then
    echo "[1] ERROR: SRL model is not detected!"
else
    echo "[1] TRACE: SRL train model test is passed."
fi

