#!/bin/bash

# Set default parameters
MODEL_NAME="deepset/roberta-base-squad2"
TRAIN_FILE="covid-qa/covid-qa-train.json"
DEV_FILE="covid-qa/covid-qa-dev.json"
TEST_FILE="covid-qa/covid-qa-test.json"
OUTPUT_DIR="./adapter-qa-results"
BATCH_SIZE=16
EPOCHS=3
LEARNING_RATE=3e-5

# Run training
echo "Starting training on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python run_qa.py \
    --model_name_or_path $MODEL_NAME \
    --train_adapter \
    --train_file $TRAIN_FILE \
    --validation_file $DEV_FILE \
    --test_file $TEST_FILE \
    --do_train --do_eval --do_predict \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --num_train_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --overwrite_output_dir
