import os
import random
import shutil

# Directory containing the images
source_directory = "/home/sszabados/datasets/lysto64_random_crop_pix2pix/AB/val"
destination_directory = "/home/sszabados/datasets/lysto64_random_crop_pix2pix/AB/val2"

# Function to copy selected images to a new folder
def copy_selected_images(directory, destination, num_to_keep=500):
    for class_num in range(3):  # Assuming classes are 0, 1, and 2
        class_files = [file for file in os.listdir(directory) if file.startswith(str(class_num) + "_")]
        random.shuffle(class_files)  # Shuffle the list of files
        files_to_copy = class_files[:num_to_keep]  # Files to copy
        os.makedirs(os.path.join(destination, str(class_num)), exist_ok=True)  # Create destination folder for the class
        for file in files_to_copy:
            shutil.copy(os.path.join(directory, file), os.path.join(destination, str(class_num), file))

# Call the function
copy_selected_images(source_directory, destination_directory)