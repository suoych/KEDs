import os
import shutil
import random
import multiprocessing

# Set the source and destination folder paths
image_source_folder = 'cc_image_feature_folder'  # Replace with the path to your source folder
image_destination_folder = 'cc_image_feature_database'  # Replace with the path to your destination folder

text_source_folder = 'cc_text_feature_folder'  # Replace with the path to your source folder
text_destination_folder = 'cc_text_feature_database'  # Replace with the path to your destination folder

# Get a list of all files in the source folder
all_files = os.listdir(text_source_folder)

# Randomly select 1,000,000 files from the list
selected_files = random.sample(all_files, 500000)


# Define a function to copy files
def copy_file(file_name):
    image_source_file_path = os.path.join(image_source_folder, file_name)
    image_destination_file_path = os.path.join(image_destination_folder, file_name)
    
    text_source_file_path = os.path.join(text_source_folder, file_name)
    text_destination_file_path = os.path.join(text_destination_folder, file_name)
    # Check if the file exists in the source folder before copying
    if os.path.isfile(image_source_file_path) and os.path.isfile(text_source_file_path):
        shutil.copy(image_source_file_path, image_destination_file_path)
        shutil.copy(text_source_file_path, text_destination_file_path)

# Create a pool of worker processes
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

# Use the pool to copy files in parallel
pool.map(copy_file, selected_files)

# Close the pool to free up resources
pool.close()
pool.join()

print(f"Successfully copied {len(selected_files)} files to the destination folder.")

