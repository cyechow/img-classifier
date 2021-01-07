import os
import cv2

def is_cv2():
    (major, _, _) = cv2.__version__.split(".")
    return major == 2

def extract_images(fullFilePath, outputFolderPath, intervalInSeconds, startTimeInSeconds):
    if os.path.exists(fullFilePath):
        _, filename = os.path.split(fullFilePath)
        filename,_ = os.path.splitext(filename)
        
        vidCap = cv2.VideoCapture(fullFilePath)
        fps = vidCap.get(cv2.CAP_PROP_FPS)
        if is_cv2():
            totalFrames = int(vidCap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        else:
            totalFrames = int(vidCap.get(cv2.CAP_PROP_FRAME_COUNT))

        intervalInFrames = int(intervalInSeconds * fps) - 1

        startFrame = int(startTimeInSeconds * fps)
        currFrame = 0
        frameOfInterest = startFrame
        
        while currFrame <= totalFrames:
            _, frame = vidCap.read()

            if currFrame == frameOfInterest:
                print('Extracting screenshot at frame {0}'.format(currFrame))
                outputFileName = filename + '_Frame_' + str(currFrame) + '.jpg'
                outputFilePath = os.path.join(outputFolderPath, outputFileName)
                cv2.imwrite(outputFilePath, frame)
                frameOfInterest = currFrame + intervalInFrames
            
            currFrame = currFrame + 1
        
        vidCap.release()

if __name__ == '__main__':
    videoPath = 'C:\\Users\\reyl2\\Documents\\src\\arup\\CentralFootbridge_190228-0723-0823_15fps_tracked.mp4'
    outputFolderPath = 'C:\\Users\\reyl2\\Documents\\src\\arup\\screenshots\\'

    if not os.path.exists(outputFolderPath):
        os.mkdir(outputFolderPath)

    extract_images(videoPath, outputFolderPath, 5, 0)