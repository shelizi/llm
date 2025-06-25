#!/bin/bash

# Script to download Llama model
URL=$(cat "$(dirname "$0")/modelurl")
OUTPUT_FILE="nvidia_Llama-3_3-Nemotron-Super-49B-v1-Q5_K_M.gguf"

echo "Downloading $OUTPUT_FILE..."
curl -L -o "$OUTPUT_FILE" "$URL"

if [ $? -eq 0 ]; then
  echo "Download completed successfully!"
else
  echo "Download failed with error code $?"
fi
