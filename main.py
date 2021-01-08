# Used to run through all images of every extracted persons from the processed video and aggregate the classification results.
# Ideal output:
#   ID | Frame | Label (male/female/none) | Average Probability
#
# Further processing to result in the following table:
#   ID | Label (%) | Total Images | Total Labelled
#
# This should give us, for every person, the most prevalent label (% out of total images for person) and the total number of
# images that were labelled by the model since some images may not have anything detected by the model.

# Data is stored in the following structure under the extraction_folder_path:
# extractions
#       person_id_0
#               frame_0.jpg
#               frame_1.jpg
#               ...
#       person_id_1
#               frame_0.jpg
#               frame_1.jpg
#               ...
#       ...
#
# Note: Images can be of formats: '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'

# This assumes that the following files from the trained model are in this folder:
#   - model_frcnn.hdf5
#   - config.pickle
#
# ********************************************************************************** #

from __future__ import division

import os
import pandas as pd

extraction_folder_path = '.\\extractions\\'
output_folder_path = '.\\results\\'

# ---------------------------------------------------------------------------------- #
# Set up the model for use
# ---------------------------------------------------------------------------------- #
import os
import cv2
import numpy as np
import sys
import pickle
import time
import tensorflow as tf
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from tensorflow.python.keras.backend import set_session
from keras_frcnn import roi_helpers

sys.setrecursionlimit(40000)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

# Location to read the metadata related to the training (generated when training):
config_output_filename = "config.pickle"

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

if C.network == 'resnet50':
	import keras_frcnn.resnet as nn
elif C.network == 'vgg':
	import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

# ---------------------------------------------------------------------------------- #
# Need to pad the screenshots:
# color = (0, 0, 0) for black? it's in RGB
def add_padding(img, width, height, color):
    # read image
    ht, wd, cc= img.shape

    # create new image of desired size and color for padding
    result = np.full((height,width,cc), color, dtype=np.uint8)

    # compute center offset
    xx = (width - wd) // 2
    yy = (height - ht) // 2

    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = img

    return result

if not os.path.exists(output_folder_path):
    os.mkdir(output_folder_path)

# Get all the folders with full path:
persons_dir = [f.name for f in os.scandir(extraction_folder_path) if f.is_dir()]

# Set up arrays for collecting information:
person_ids = []
frame_number = []
person_label = []
person_label_probability = []

# Set up to run model:
class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

#Number of ROIs per iteration. Higher means more memory use:
C.num_rois = 32

if C.network == 'resnet50':
    num_features = 1024
elif C.network == 'vgg':
    num_features = 512

if K.image_data_format() == 'channels_first':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print(f'Loading weights from {C.model_path}')
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

bbox_threshold = 0.8

# Loop through every ID'd person's folder:
os.listdir()
for person_id in persons_dir:
    # Get person ID:
    person_path = os.path.join(extraction_folder_path, person_id)

    # Get all the files:
    extracted_imgs = os.listdir(person_path)

    # Loop through every image for the current person:
    for img_file in extracted_imgs:
        # Run model on this, this is taken and modified from test_frcnn.py in the kbardool/keras-frcnn repo:
        if not img_file.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue

        # The filename is the frame number, this is how the data is set up.
        curr_frame_num,_ = os.path.splitext(img_file)
        st = time.time()
        img_path = os.path.join(person_path,img_file)

        print('Processing ID{0} at frame {1}'.format(person_id, curr_frame_num))

        # Get output image path, skip drawing and saving if it already exists:
        img_full_path = os.path.join(output_folder_path,person_id)
        if not os.path.exists(img_full_path):
            os.mkdir(img_full_path)
        img_full_path = os.path.join(img_full_path, '{0}_result.png'.format(curr_frame_num))
        print(img_full_path)
        is_img_output_exists = os.path.exists(img_full_path)

        img = cv2.imread(img_path)
        try:
            if type(img) == None:
                continue
            # Pad to 1920x1080
            img = add_padding(img, 1920, 1080, (0,0,0))

            X, ratio = format_img(img, C)

            if K.image_data_format() == 'channels_last':
                X = np.transpose(X, (0, 2, 3, 1))

            # get the feature maps and output from the RPN
            [Y1, Y2, F] = model_rpn.predict(X)
            

            R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

            # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0]//C.num_rois + 1):
                ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0]//C.num_rois:
                    #pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

                for ii in range(P_cls.shape[1]):

                    if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        continue

                    cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                        tx /= C.classifier_regr_std[0]
                        ty /= C.classifier_regr_std[1]
                        tw /= C.classifier_regr_std[2]
                        th /= C.classifier_regr_std[3]
                        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []

            # Ideally there should only be one box because we're aiming to pass in an image of a single person, but the bbox in the crop
            # may include people in the back/foreground so we'll have to make a decision.
            #
            # We'll try this:
            #   Combination of confidence (probability) and larger bbox area
            key_data = pd.DataFrame()
            keys = []
            nKeys = []
            minProbKeys = []
            maxProbKeys = []
            avgProbKeys = []
            for key in bboxes:
                bbox = np.array(bboxes[key])

                # Get the bounding boxes and associated probabilites of all detection that falls under this 'key':
                new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)

                # Store this for decision-making:
                label = str(key)
                keys.append(label)
                nKeys.append(len(new_boxes))
                minProbKeys.append(min(new_probs))
                maxProbKeys.append(max(new_probs))
                avgProbKeys.append(sum(new_probs)/len(new_probs))

                # Draw the detection onto the images:
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk,:]

                    (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                    # Don't bother with drawing if this has already been done:
                    if not is_img_output_exists:
                        cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

                    textLabel = f'{key}: {int(100*new_probs[jk])}'
                    all_dets.append((key,100*new_probs[jk]))

                    # Don't bother with drawing if this has already been done:
                    if not is_img_output_exists:
                        (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                        textOrg = (real_x1, real_y1-0)

                        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                        cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                        cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

            key_data = pd.DataFrame(data={'key': keys, 'count': nKeys, 'min_prob': minProbKeys, 'max_prob': maxProbKeys, 'avg_prob': avgProbKeys})
            if not key_data.empty:
                # No classification, log
                prevalent_key = key_data[key_data['count'] == max(key_data['count'])]
                if prevalent_key.empty:
                    person_label.append('NA')
                    person_label_probability.append(-1)
                elif len(prevalent_key) == 1:
                    person_label.append(np.array(prevalent_key['key'])[0])
                    person_label_probability.append(float(prevalent_key['avg_prob']))
                else:
                    # Take the one with the higher average probability:
                    prevalent_key = prevalent_key[prevalent_key['avg_prob'] == max(prevalent_key['avg_prob'])]
                    person_label.append(np.array(prevalent_key['key'])[0])
                    person_label_probability.append(float(prevalent_key['avg_prob']))
                
                person_ids.append(person_id)
                frame_number.append(curr_frame_num)

            print(f'Elapsed time = {time.time() - st}')
            
            print(all_dets)
            
            # Write image to the output folder in the subdirectory with ID, append '_result' at the end:
            if not is_img_output_exists:
                cv2.imwrite(img_full_path,img)

            # Just save/update the csv:
            dfc = pd.DataFrame(data={'id': person_ids, 'frame': frame_number, 'label': person_label, 'probability': person_label_probability})
            dfc.to_csv(os.path.join(output_folder_path, 'labelled_data.csv'))
        except:
            #traceback.print_exc()
            continue


# Dump data into dataframe and save:
    
# Set up arrays for collecting information:
dfc = pd.DataFrame(data={'id': person_ids, 'frame': frame_number, 'label': person_label, 'probability': person_label_probability})
dfc.to_csv(os.path.join(output_folder_path, 'labelled_data.csv'))


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


# ----------------------------------------------------------- #
# Visualizations
# ----------------------------------------------------------- #
import matplotlib.pyplot as plt

nTotalAgents = len(persons_dir)

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

labels = 'Processed', 'Unprocessed'
sizes = [127, 2578]
explode = (0.1, 0)  # "explode" the processed slice. 
colors = ['#99ff99','#66b3ff']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=make_autopct(sizes),
        shadow=True, startangle=45)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Total Persons Detected')
plt.show()

labels = 'Female', 'Male'
sizes = [dfa['label'].value_counts()['female'], dfa['label'].value_counts()['male']]
explode = (0,0)
colors = []

# ----------------------------------------------------------- #