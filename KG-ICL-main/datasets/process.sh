#!/bin/bash

run_command() {
    echo "processing: $1"
    $1
    if [ $? -ne 0 ]; then
        echo "failed: $1"
        exit 1
    fi
}

run_command "python inductive/data_process.py"
run_command "python fully-inductive/data_process.py"
run_command "python transductive/data_process.py"
run_command "python ILPC/data_process.py"
run_command "python multi-hop/data_process.py"

echo "Finish！"
