import numpy as np
import imutils
from matplotlib import pyplot as plt
import openai
import cv2
import pytesseract
import pyttsx3
import math
import requests
from requests.auth import HTTPBasicAuth
from cvzone.HandTrackingModule import HandDetector
from ultralytics import YOLO
import supervision as sv

# YOLO MODEL
# model = YOLO("weights/yolov8s.pt", "v8")

# For Optical Character Recognition & Text to Speech
speech = pyttsx3.init()
file_path='ProductCaptured.txt'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# OpenAI Key
openai.api_key = "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"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

detector = HandDetector(maxHands=1)
offset = 300
imgSize = 900
counter = 0

url = "http://192.0.0.2:8080/shot.jpg"
username = "ABC"
password = "abc"


''' From here onwards the commented code is for implementation of YOLOv8 model in the program -> Throwing errors for now.'''
# box_annotator = sv.BoxAnnotator(
#         thickness=2,
#         text_thickness=2,
#         text_scale=1
#     )

while True:
    success , image = cap.read()
    '''If I want to use my Phone as a Camera'''
    # img_resp = requests.get(url, auth = HTTPBasicAuth(username, password))
    # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    # image = cv2.imdecode(img_arr, -1)
    image = imutils.resize(image, width=640, height=480)
    
    hands= detector.findHands(img=image,draw=False)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = image[y-offset:y+h+offset, x-offset:x+w+offset]
        # detect_params = model.predict(source=imgCrop, save=False)
        imgCropShape = imgCrop.shape
        
        # result = model(imgCrop, agnostic_nms=True)[0]
        # detections = sv.Detections.from_yolov8(result)
        # labels = [
        #     f"{model.model.names[class_id]} {confidence:0.2f}"
        #     for _, confidence, class_id, _
        #     in detections
        # ]
        # frame = box_annotator.annotate(
        #     scene=image, 
        #     detections=detections, 
        #     labels=labels,
        #     skip_label=True
        # )
        # frame2 = box_annotator.annotate(scene=frame,detections=detections)
        # frame2 = cv2.resize(frame2,(640,480))
        aspectratio = h/w

        if aspectratio > 1:
            k = imgSize/h
            widthCal = math.ceil(k*w)
            try:
                imgResize = cv2.resize((imgCrop), (widthCal, imgSize))
            except cv2.error as e:
                print(f"Error resizing image: {e}")
                continue

            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgSize-widthCal)/2)
            imgWhite[:, widthGap:widthCal+widthGap] = imgResize
            
        else:
            k = imgSize/w
            heightCal = math.ceil(k*h)
            try:
                imgResize = cv2.resize((imgCrop), (imgSize, heightCal))
            except cv2.error as e:
                print(f"Error resizing image: {e}")
                continue
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgSize-heightCal)/2)
            imgWhite[heightGap:heightCal+heightGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("Imagewhite", imgWhite)
    
    cv2.imshow("Image", image)

    if cv2.waitKey(1) == ord('p'):
        imgS = cv2.cvtColor(imgCrop,cv2.COLOR_BGR2RGB)
        text=pytesseract.image_to_string(imgS)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if text in lines:
                continue # skip duplicate
        with open(file_path, 'a') as f:
            f.write(text + '\n')
    
    if cv2.waitKey(1) == ord('m'):
        with open(file_path, 'r') as f:
            gpttext = f.read()
            # For ChatGPT API
            print("OpenAI response is: ")
            prompt = f'''I will give you some text from a Item slip, find the hidden Item name in it.
                        give the Item name and its desciption in the following manner, keep it short
                        Item name: 
                        Descrpition:
                        the text is:   
                        {gpttext}'''
            response = openai.Completion.create(
                engine="text-davinci-003", prompt=prompt, max_tokens=3000
            )
            answer = response.choices[0]['text']
            print(answer)
            speech.say(answer)
            speech.runAndWait()
            with open(file_path, 'w') as f:
                f.write('')
    if cv2.waitKey(1) == ord('`'):
        break
  
cv2.destroyAllWindows()
