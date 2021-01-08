# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

extraction_folder_path = '.\\extractions\\'
output_folder_path = '.\\results\\'

# Get all the folders with full path:
persons_dir = [f.name for f in os.scandir(extraction_folder_path) if f.is_dir()]

dfc = pd.read_csv("labelled_data.csv")


# Aggregate per person and get the final labels:
#   ID | Label (%) | Total Images | Total Labelled
person_ids = []
person_labels = []
total_images = []
total_images_labelled = []
total_images_labelled_percentage = []
person_start_frame = []
person_end_frame = []
for id in persons_dir:
    dfp = dfc[dfc['id'] == int(id)]
 
    # Get total images in folder:
    person_path = os.path.join(extraction_folder_path, id)

    # Get all the files:
    nTotal = len(os.listdir(person_path))
    
    person_ids.append(int(id))
    total_images.append(nTotal)
    
    if dfp.empty:
        # No labels
        person_labels.append('NA')
        total_images_labelled.append(0)
        total_images_labelled_percentage.append(0)
    else:
        print('Aggregating {0}'.format(id))
        
        # Get start/end frames:
        start_frame_num = min(dfp['frame'])
        end_frame_num = max(dfp['frame'])
        person_start_frame.append(start_frame_num)
        person_end_frame.append(end_frame_num)

        nMale = sum(dfp['label'] == 'male')
        nFemale = sum(dfp['label'] == 'female')
        if nMale == 0 and nFemale == 0:
            person_labels.append('NA')
            total_images_labelled.append(0)
            total_images_labelled_percentage.append(0)
        elif nMale == nFemale:
            dfp_male = dfp[dfp['label'] == 'male']
            dfp_female = dfp[dfp['label'] == 'female']
            if max(dfp_male['probability']) > max(dfp_female['probability']):
                person_labels.append('male')
                total_images_labelled.append(nMale)
                total_images_labelled_percentage.append(nMale/nTotal)
            else:
                person_labels.append('female')
                total_images_labelled.append(nFemale)
                total_images_labelled_percentage.append(nFemale/nTotal)
        elif nMale > nFemale:
            person_labels.append('male')
            total_images_labelled.append(nMale)
            total_images_labelled_percentage.append(nMale/nTotal)
        else:
            person_labels.append('female')
            total_images_labelled.append(nFemale)
            total_images_labelled_percentage.append(nFemale/nTotal)

dfa = pd.DataFrame(data={'id':person_ids,'label':person_labels,'total_images':total_images,'total_images_labelled':total_images_labelled,'total_images_labelled_percentage':total_images_labelled_percentage,'start_frame':start_frame_num,'end_frame':end_frame_num})
dfa.to_csv(os.path.join(output_folder_path,'aggregated_data.csv'))
