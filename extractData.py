import os
import xml.etree.ElementTree as ET
import cv2

import pandas as pd
import numpy as np
import random
from shutil import copyfile


def extract_persons(fullFilePath, outputFolderPath, idxStart):
    filename, ext = os.path.splitext(fullFilePath)

    # Read in image:
    img = cv2.imread(fullFilePath)

    # XML file containing the labels with bounding boxes:
    xmlfile = filename + '.xml'

    print(xmlfile)
    if os.path.exists(xmlfile):
        xmlTree = ET.parse(xmlfile)
        xmlRoot = xmlTree.getroot()


        for obj in xmlRoot.findall('object'):
            isMale = obj.find('name').text == 'male'
            isFemale = obj.find('name').text == 'female'

            if isMale or isFemale:
                bboxElement = obj.find('bndbox')
                xmin = int(bboxElement.find('xmin').text)
                xmax = int(bboxElement.find('xmax').text)
                ymin = int(bboxElement.find('ymin').text)
                ymax = int(bboxElement.find('ymax').text)

                labelledImg = img[ymin:ymax, xmin:xmax]
                if isMale:
                    imgPath = os.path.join(outputFolderPath, 'male')
                else:
                    imgPath = os.path.join(outputFolderPath, 'female')

                if not os.path.exists(imgPath):
                    os.mkdir(imgPath)

                imgPath = os.path.join(imgPath, str(idxStart) + '.jpg')
                cv2.imwrite(imgPath, labelledImg)
                idxStart += 1

    return idxStart

def aggregate_to_csv(folderPath, outputFolderPath, train_percentage):
    annotation_files = os.listdir(folderPath) 

    # Get total number of files to pull for training:
    nFiles = len(annotation_files)
    nTrain = round(train_percentage*nFiles)
    idxTrain = random.sample(range(0,nFiles-1),nTrain)

    #dir_paths_validate, image_names_validate, cell_type_validate, xmin_array_validate, xmax_array_validate, ymin_array_validate, ymax_array_validate = [], [], [], [], [], [], []
    #dir_paths_train, image_names_train, cell_type_train, xmin_array_train, xmax_array_train, ymin_array_train, ymax_array_train = [], [], [], [], [], [], []
    dir_paths_train, file_names_train, labels_train, xmin_array_train, xmax_array_train, ymin_array_train, ymax_array_train = [], [], [], [], [], [], []
    dir_paths_validate, file_names_validate, labels_validate, xmin_array_validate, xmax_array_validate, ymin_array_validate, ymax_array_validate = [], [], [], [], [], [], []

    # Get the parent dir:
    parentDir = os.path.dirname(folderPath.strip("\\"))

    # Make directories to store the separated images:
    trainImagesDir = os.path.join(outputFolderPath,'train_images')
    trainAnnotationsDir = os.path.join(outputFolderPath,'train_annotations')
    validateImagesDir = os.path.join(outputFolderPath,'validate_images')
    validateAnnotationsDir = os.path.join(outputFolderPath,'validate_annotations')
    if not os.path.exists(trainImagesDir):
        os.mkdir(trainImagesDir)
    if not os.path.exists(trainAnnotationsDir):
        os.mkdir(trainAnnotationsDir)
    if not os.path.exists(validateImagesDir):
        os.mkdir(validateImagesDir)
    if not os.path.exists(validateAnnotationsDir):
        os.mkdir(validateAnnotationsDir)

    i = 0
    for file in annotation_files:
        print('Extracting from {0}'.format(file))
        xmlTree = ET.parse(os.path.join(folderPath,file))
        xmlRoot = xmlTree.getroot()

        imgRootFolder = xmlRoot.find('folder').text
        imgName = xmlRoot.find('filename').text

        for obj in xmlRoot.findall('object'):
            label = obj.find('name').text
            # Only take the male and female labels:
            if label == 'male' or label == 'female':
                bboxElement = obj.find('bndbox')
                xmin = int(bboxElement.find('xmin').text)
                xmax = int(bboxElement.find('xmax').text)
                ymin = int(bboxElement.find('ymin').text)
                ymax = int(bboxElement.find('ymax').text)

                # Store info:
                if i in idxTrain:
                    dir_paths_train.append(imgRootFolder)
                    #image_names_train.append(imgName)
                    file_names_train.append(imgName)
                    #cell_type_train.append(label)
                    labels_train.append(label)
                    xmin_array_train.append(xmin)
                    xmax_array_train.append(xmax)
                    ymin_array_train.append(ymin)
                    ymax_array_train.append(ymax)
                else:
                    dir_paths_validate.append(imgRootFolder)
                    #image_names_validate.append(imgName)
                    file_names_validate.append(imgName)
                    #cell_type_validate.append(label)
                    labels_validate.append(label)
                    xmin_array_validate.append(xmin)
                    xmax_array_validate.append(xmax)
                    ymin_array_validate.append(ymin)
                    ymax_array_validate.append(ymax)
        
        # Copy the image into the respective directory:
        imgFolderPath = os.path.join(parentDir,imgRootFolder)
        if i in idxTrain:
            copyfile(os.path.join(imgFolderPath,imgName), os.path.join(trainImagesDir,imgName))
            copyfile(os.path.join(folderPath,file), os.path.join(trainAnnotationsDir,file))
        else:
            copyfile(os.path.join(imgFolderPath,imgName), os.path.join(validateImagesDir,imgName))
            copyfile(os.path.join(folderPath,file), os.path.join(validateAnnotationsDir,file))

        i = i + 1

    #d_train = {'dir_path': np.array(dir_paths_train), 'image_names': np.array(image_names_train), 'cell_type': np.array(cell_type_train), 'xmin': np.array(xmin_array_train), 'xmax': np.array(xmax_array_train), 'ymin': np.array(ymin_array_train), 'ymax': np.array(ymax_array_train) }
    #d_validate = {'dir_path': np.array(dir_paths_validate), 'image_names': np.array(image_names_validate), 'cell_type': np.array(cell_type_validate), 'xmin': np.array(xmin_array_validate), 'xmax': np.array(xmax_array_validate), 'ymin': np.array(ymin_array_validate), 'ymax': np.array(ymax_array_validate) }
    d_train = {'dir_path': np.array(dir_paths_train), 'file_name': np.array(file_names_train), 'label': np.array(labels_train), 'xmin': np.array(xmin_array_train), 'xmax': np.array(xmax_array_train), 'ymin': np.array(ymin_array_train), 'ymax': np.array(ymax_array_train) }
    d_validate = {'dir_path': np.array(dir_paths_validate), 'file_name': np.array(file_names_validate), 'label': np.array(labels_validate), 'xmin': np.array(xmin_array_validate), 'xmax': np.array(xmax_array_validate), 'ymin': np.array(ymin_array_validate), 'ymax': np.array(ymax_array_validate) }
    df_train = pd.DataFrame(data=d_train)
    df_validate = pd.DataFrame(data=d_validate)
    
    df_train.to_csv(os.path.join(outputFolderPath, 'train.csv'))
    df_validate.to_csv(os.path.join(outputFolderPath, 'validate.csv'))
    
    return df_train, df_validate


def save_formatted_data(df_train, filename, outputFolderPath):
    # For Faster R-CNN implemented in https://github.com/kbardool/keras-frcnn.git, the format is filepath,x1,y1,x2,y2,class_name
    data = pd.DataFrame()
    #data['format'] = df_train['image_names']
    data['format'] = df_train['file_name']

    # The images will be placed in a 'train_images' folder in the keras-frcnn clone repo:
    for i in range(data.shape[0]):
        data['format'][i] = 'train_images/' + data['format'][i]

    for i in range(data.shape[0]):
        data['format'][i] = data['format'][i] + ',' + str(df_train['xmin'][i]) + ',' + str(df_train['ymin'][i]) + ',' + str(df_train['xmax'][i]) + ',' + str(df_train['ymax'][i]) + ',' + df_train['label'][i]

    data.to_csv(os.path.join(outputFolderPath, filename), header=None, index=None, sep=' ')
    
    return data


def extract_roi():
    folderPath = 'C:\\Users\\reyl2\\Documents\\src\\arup\\screenshots'
    outputFolderPath = 'C:\\Users\\reyl2\\Documents\\src\\arup\\persons'

    # Make sure output folder exists:
    if not os.path.exists(outputFolderPath):
        os.mkdir(outputFolderPath)

    files = os.listdir(folderPath)

    # For every jpg or png in the folder, get the corresponding xml
    idxStart = 0
    for file in files:
        name, ext = os.path.splitext(file)
        filePath = os.path.join(folderPath, file)
        if ext == '.jpg' or ext == '.png':
            print('Extracting labelled persons from {0} starting at index {1}.'.format(filePath, idxStart))
            idxStart = extract_persons(filePath, outputFolderPath, idxStart)

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


def extract_detected_persons():
    csvFilePath = 'C:\\Users\\reyl2\\Documents\\src\\arup\\CentralFootbridge_190227-0723-0823_15fps_tracked.csv'
    vidFilePath = 'C:\\Users\\reyl2\\Documents\\src\\arup\\CentralFootbridge_190228-0723-0823_15fps_tracked.MP4'
    outputFolderPath = 'C:\\Users\\reyl2\\Documents\\src\\arup\\extractions\\'

    df = pd.read_csv(csvFilePath)

    # Columns
    # time,frame,id,confidence,mid-x,mid-y,width,height,speed,direction,density around ppl,zone,# of ppl in zone 1 and 2,# of ppl in zone 1,# of ppl in zone 2,# of ppl across line in zone 1,# of ppl across line in zone 2,# of ppl going downwards in zone 1,# of ppl going downwards in zone 2,all_time_mean_speed,time interval

    # We only want: time, frame, id, mid-x, mid-y, width, height
    df_sub = pd.DataFrame()
    df_sub['time'] = df['time']
    df_sub['frame'] = df['frame']
    df_sub['id'] = df['id']
    df_sub['mid-x'] = df['mid-x']
    df_sub['mid-y'] = df['mid-y']
    df_sub['width'] = df['width']
    df_sub['height'] = df['height']

    # Open up vid capture to get frame images:
    vid_cap = cv2.VideoCapture(vidFilePath)
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    curr_frame = 0
    while curr_frame <= total_frames:
        _, img_frame = vid_cap.read()

        frame_data = df_sub[df_sub['frame'] == curr_frame]

        if not frame_data.empty:
            for id in frame_data['id']:
                print('Cropping detected person {0} in frame {1}'.format(id, curr_frame))
                p = frame_data[frame_data['id'] == id]
                
                if len(p) > 1:
                    p = p.iloc[0]
                xmin = int(p['mid-x'] - 0.5*p['width'])
                xmax = int(p['mid-x'] + 0.5*p['width'])
                ymin = int(p['mid-y'] - 0.5*p['height'])
                ymax = int(p['mid-y'] + 0.5*p['height'])

                img_cropped = img_frame[ymin:ymax, xmin:xmax]
                img_path = os.path.join(outputFolderPath, str(id))

                if not os.path.exists(img_path):
                    os.mkdir(img_path)

                img_path = os.path.join(img_path, str(curr_frame) + '.jpg')
                cv2.imwrite(img_path, img_cropped)
        curr_frame = curr_frame + 1

    vid_cap.release()

if __name__ == '__main__':
    folderPath = 'C:\\Users\\reyl2\\Documents\\src\\arup\\Annotations\\'
    outputFolderPath = 'C:\\Users\\reyl2\\Documents\\src\\arup\\keras-fcrnn\\'

    df_train, df_validate = aggregate_to_csv(folderPath, outputFolderPath, 0.8)
    data_formatted = save_formatted_data(df_train, 'annotate.txt', outputFolderPath)
    
    df_train['label'].value_counts()
    df_validate['label'].value_counts()