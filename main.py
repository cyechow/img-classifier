# Used to run through all images of every extracted persons from the processed video and aggregate the classification results.
# Ideal output:
#   ID | Frame | Label (male/female/none)
#
# Further processing to result in the following table:
#   ID | Label (%) | Total Images | Total Labelled
#
# This should give us, for every person, the most prevalent label (% out of total images for person) and the total number of
# images that were labelled by the model since some images may not have anything detected by the model.

import os

extraction_folder_path = '.\\extractions\\'
output_folder_path = '.\\results\\'

if not os.path.exists(output_folder_path):
    os.mkdir(output_folder_path)

    # Get all the folders with full path:
    persons_dir = [f.path for f in os.scandir(extraction_folder_path) if f.is_dir()]

    for d in persons_dir:
        # Get all the files:
        extracted_imgs = os.listdir(d)

        for img in extraction_folder_path:
            # Run model on this: