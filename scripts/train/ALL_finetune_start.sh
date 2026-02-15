#!/bin/bash

# Check if a path argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_scripts>"
  exit 1
fi

SCRIPT_DIR="$1"

# Define the 4 scripts to run
scripts=("start_8b_enchanced_batch.sh" "start_8b_normal_batch.sh" "start_32b_enchanced_batch.sh" "start_32b_normal_batch.sh")

for script in "${scripts[@]}"; do
  script_path="${SCRIPT_DIR}/${script}"
  if [ -f "$script_path" ]; then
    echo "Running $script_path ..."
    bash "$script_path"
  else
    echo "Warning: $script_path not found, skipping."
  fi
done

echo "All done."