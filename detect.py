import cv2
import mediapipe as mp
import pandas as pd
import pickle
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

font_src = './washrab.ttf'
#For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

   while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

      # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      # mp_drawing.draw_landmarks(
      #     image,
      #     results.face_landmarks,
      #     mp_holistic.FACEMESH_CONTOURS,
      #     landmark_drawing_spec=None,
      #     connection_drawing_spec=mp_drawing_styles
      #     .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
      image,
      results.pose_landmarks,
      mp_holistic.POSE_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles
      .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
      image,
      results.left_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles
      .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
      image,
      results.right_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS,
      landmark_drawing_spec=mp_drawing_styles
      .get_default_pose_landmarks_style())
    try:
        # Extract Pose landmarks
        pose = list(np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4))  
        lh = list(np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3))
        rh = list(np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3))
                    
        # Concate rows
        row = pose +lh+rh
        # Make Detections
        X = pd.DataFrame([row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]
        print(body_language_class, np.argmax(body_language_prob))
        
        # Grab ear coords
        coords = tuple(np.multiply(
                      np.array(
                      (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                      results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                      , [640,480]).astype(int))
            
      
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        if not (body_language_class.split(' ')[0] == 'nothing'):
          image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          pil_img = Image.fromarray(image)

          draw = ImageDraw.Draw(pil_img)
          # font = ImageFont.truetype(<font-file>, <font-size>)
          font = ImageFont.truetype(font_src, 50)
          # draw.text((x, y),"Sample Text",(r,g,b))
          text = "{}  {}".format(body_language_class.split(' ')[0],str(round(body_language_prob[np.argmax(body_language_prob)],2)))
          draw.text((0, 0),text,(255,255,255),font=font)
          image= np.array(pil_img)
          image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    
        # # Get status box
        # cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
        # # Display Class
        # cv2.putText(image, 'CLASS'
        #     , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, body_language_class.split(' ')[0]
        #     , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        # # Display Probability
        # cv2.putText(image, 'PROB'
        #     , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
        #     , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)           
    except:
        pass
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
   cap.release()
