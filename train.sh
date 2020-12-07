#!/bin/bash 

UD_PATH=$1
UD_LANG=$2
UD_CORP=$3
ISO_CODE=$4
MS_PATH=$5
OUTPUT_PATH=$6

UD_CORP_DIR="UD_"$UD_LANG"-"$UD_CORP

TRAIN_FILE=`ls $UD_PATH/$UD_CORP_DIR/*-ud-train.conllu | head -n 1`
if [ ! -f $TRAIN_FILE ]; then
    echo "$ISO_CODE ERROR: can't find train file in $UD_PATH/$UD_CORP_DIR"
    exit 1
fi

DEV_FILE=`ls $UD_PATH/$UD_CORP_DIR/*-ud-dev.conllu | head -n 1`
if [ ! -f $DEV_FILE ]; then
    echo "$ISO_CODE ERROR: can't find dev file in $UD_PATH/$UD_CORP_DIR"
    exit 1
fi

TEST_FILE=`ls $UD_PATH/$UD_CORP_DIR/*-ud-test.conllu | head -n 1`
if [ ! -f $TEST_FILE ]; then
    echo "$ISO_CODE ERROR: can't find test file in $UD_PATH/$UD_CORP_DIR"
    exit 1
fi

if [ ! -f $MS_PATH/morphosyntax-$ISO_CODE.conf ]; then
    echo "$ISO_CODE ERROR: can't find morphosyntax config file in $MS_PATH"
    exit 1
fi

python3 train.py \
    -t $TRAIN_FILE \
    -d $DEV_FILE \
    -c $MS_PATH/morphosyntax-$ISO_CODE.conf \
    -m $OUTPUT_PATH/lemmatizer-$ISO_CODE \
    2> $OUTPUT_PATH/$ISO_CODE.train.err

python3 predict.py \
    -m $OUTPUT_PATH/lemmatizer-$ISO_CODE \
    -i $TEST_FILE \
    2> $OUTPUT_PATH/$ISO_CODE.pred.err \
    > $OUTPUT_PATH/$ISO_CODE-pred.conllu

python ~/lima/conll18_ud_eval.py -v \
    $TEST_FILE \
    $OUTPUT_PATH/$ISO_CODE-pred.conllu \
    > $OUTPUT_PATH/$ISO_CODE-pred.eval

RES=` grep 'Lemmas' $OUTPUT_PATH/$ISO_CODE-pred.eval `
echo $ISO_CODE $RES
