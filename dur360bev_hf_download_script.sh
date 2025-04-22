#!/usr/bin/bash

################################################################################

# Dataset Downloader for Dur360BEV, Huggingface 
# (c) 2025 Li (Luis) Li, King's College London
# (c) 2025 Tom E, Durham University
# version v0.1.2 - 19 March, 2025

################################################################################

# === User Configuration (Modify as Needed) ===
DOWNLOAD_DIR="$HOME/Dur360BEV_download"  # Directory to store downloaded .tar parts
EXTRACT_DIR="$HOME/Dur360BEV" # Directory where extracted dataset will be stored
CLEANUP_AFTER_EXTRACTION=true  # Set to true to delete .tar files after extraction, false to keep them

# === Fixed Configuration (No Need to Modify) ===
HF_DATASET_ID="TomEeee/Dur360BEV"  # The dataset ID on Hugging Face

# === Display the Configuration ===
echo "====================================="
echo " Dur360BEV Dataset Download Configuration"
echo "====================================="
echo "Dataset ID                 : $HF_DATASET_ID"
echo "Download Directory         : $DOWNLOAD_DIR"
echo "Extraction Directory       : $EXTRACT_DIR"
echo "Cleanup After Extraction   : $CLEANUP_AFTER_EXTRACTION"
echo "====================================="

# # === Create necessary directories ===
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$EXTRACT_DIR"

# # === Authenticate with Hugging Face CLI ===
echo "Authenticating with Hugging Face..."
login_status=$(huggingface-cli whoami 2>&1)
if echo "$login_status" | grep -q "Not logged in"; then
    echo "ERROR: Not logged in to Hugging Face. Please run 'huggingface-cli login' first!"
else
    echo "Logged in successfully as $login_status."
fi

# # === Download the dataset ===
echo "Downloading dataset to $DOWNLOAD_DIR..."
huggingface-cli download "$HF_DATASET_ID" --repo-type dataset --local-dir "$DOWNLOAD_DIR" --resume
if [ $? -ne 0 ]; then
    echo "Dataset download failed. Please check your access permissions."
    exit 1
fi
echo "Download completed!"

# === Stream all parts into tar, extracting on the fly 
cd "$DOWNLOAD_DIR" || exit 1
echo "Streaming .tar parts directly to extractor..."
cat Dur360BEV_dataset.tar* | tar --use-compress-program="pigz -p $(nproc) -d" -xvf - -C "$EXTRACT_DIR"
if [ $? -ne 0 ]; then
    echo "Extraction failed! Please check the downloaded files."
    exit 1
fi

echo "Extraction completed successfully!"

# (Optional) Remove the original parts
if [ "$CLEANUP_AFTER_EXTRACTION" = true ]; then
    echo "Cleaning up downloaded .tar parts..."
    rm -f Dur360BEV_dataset.tar*
    echo "Cleanup completed!"
fi

# === Final Message ===
echo "Dataset is ready in: $EXTRACT_DIR"

################################################################################