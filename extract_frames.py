import cv2
import os
import errno
actions = os.listdir('videos')
print(actions)

def FrameCapture(video,a,path):
  
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
        cv2.imwrite("{}/{}{}frame%d.jpg".format(path,video,a) % count, image)
  
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

for a in actions:
    videos = os.listdir('videos/{}'.format(a))
    
    path = os.path.join('frames',a)
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    for video in videos:
        path = os.path.join('frames/{}'.format(a),video)
        print(path)
        try:
            os.mkdir(path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory already exists.')
            else:
                print(e.errno)
                raise
        FrameCapture(video,a,path)

