#!/bin/bash

# Set variables
test_data_dir="SfPUEL_test_data/real1"
save_folder="sfpuel_real1"

# Run the Python test
python -m tools.test --data_dir_tst "$test_data_dir" --suffix "$save_folder"
