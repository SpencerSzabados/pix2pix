#!/bin/bash

# Set the directory containing the image files
IMAGE_DIR="/home/sszabados/datasets/ct_pet/AB/val"

# Set the number of images to keep
NUM_TO_KEEP=1500

# Change directory to the image directory
cd "$IMAGE_DIR" || exit

# Count the total number of image files
TOTAL_IMAGES=$(ls -1 | wc -l)

# If there are fewer than or equal to 1500 images, do nothing
if [ "$TOTAL_IMAGES" -le "$NUM_TO_KEEP" ]; then
    echo "Total number of images is already $TOTAL_IMAGES which is less than or equal to $NUM_TO_KEEP. No images will be deleted."
    exit
fi

# Calculate the number of images to delete
NUM_TO_DELETE=$((TOTAL_IMAGES - NUM_TO_KEEP))

# Delete images at random
ls -1 | shuf -n "$NUM_TO_DELETE" | xargs rm -f

echo "Deleted $NUM_TO_DELETE random images."