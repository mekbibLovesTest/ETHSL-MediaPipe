import cv2
import os
import errno

def FrameCapture(video,a):
  
    # Path to video file
    vidObj = cv2.VideoCapture("./videos/{}/{}".format(a,video))
  
    # Used as counter variable
    count = 0
  
    # checks whether frames were extracted
    success = 1
  
    while vidObj.isOpened():
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        if image is None:
            break 
        # Saves the frames with frame-count
        cv2.imwrite("frames/{}/{}frame%d.jpg".format(a,video) % count, image)
  
        count += 1
    vidObj.release()
    cv2.destroyAllWindows()

try:
    os.mkdir("frames")
except OSError as e:
    if e.errno == errno.EEXIST:
        print('Directory already exists.')
    else:
        raise

actions = os.listdir('videos')
print(actions)
for a in actions:
    videos = os.listdir('videos/{}'.format(a))
    print(a)
    path = os.path.join('frames',a)
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    for video in videos:
        print(video)
        FrameCapture(video,a)

