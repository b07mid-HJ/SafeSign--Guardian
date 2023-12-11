import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, Response, jsonify


model = keras.models.load_model('model.h5')

def returnRedness(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    return v

def threshold(img, T=150):
    _, img = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    return img

def findContour(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def findBiggestContour(contours):
    m = 0
    c = [cv2.contourArea(i) for i in contours]
    return contours[c.index(max(c))]

def boundaryBox(img, contours):
    x, y, w, h = cv2.boundingRect(contours)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    sign = img[y:(y + h), x:(x + w)]
    return img, sign, (x, y, w, h)

def preprocessingImageToClassifier(image=None, imageSize=32):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalize the histogram
    equalized_image = cv2.equalizeHist(gray_image)

    # Normalize the image
    normalized_image = equalized_image / 255.0

    # Resize the image to the desired size
    resized_image = cv2.resize(normalized_image, (imageSize, imageSize))

    # Add a channel dimension to match the input shape expected by the model
    final_image = resized_image.reshape(1, imageSize, imageSize, 1)

    return final_image

def predict(sign):
    img = preprocessingImageToClassifier(sign, imageSize=32)
    predictions = model.predict(img,verbose=0)
    max_confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)
    
    if max_confidence > 0.75:
        return predicted_class, max_confidence
    else:
        return None, None

#--------------------------------------------------------------------------
labelToText = {
    0: 'Speed Limit 20 km/h',
    1: 'Speed Limit 30 km/h',
    2: 'Speed Limit 50 km/h',
    3: 'Speed Limit 60 km/h',
    4: 'Speed Limit 70 km/h',
    5: 'Speed Limit 80 km/h',
    6: 'End of Speed Limit 80 km/h',
    7: 'Speed Limit 100 km/h',
    8: 'Speed Limit 120 km/h',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

cap = cv2.VideoCapture(0)

class VideoCamera:
    def __init__(self, model):
        self.model = model

    def get_frame(self):
        _, frame = cap.read()
        redness = returnRedness(frame)
        thresh = threshold(redness)
        try:
            contours = findContour(thresh)
            big = findBiggestContour(contours)
            if cv2.contourArea(big) > 2000:
                img, sign, bounding_box = boundaryBox(frame, big)
                prediction, confidence = predict(sign)
                if prediction is not None and confidence is not None and confidence > 0.75:
                    print("Now, I see:", labelToText[prediction], "with confidence:", confidence)
            else:
                img = frame
        except:
            img = frame

        _, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
    
    def get_prediction(self):
        _, frame = cap.read()
        _, frame = cap.read()
        redness = returnRedness(frame)
        thresh = threshold(redness)
        try:
            contours = findContour(thresh)
            big = findBiggestContour(contours)
            if cv2.contourArea(big) > 2000:
                img, sign, bounding_box = boundaryBox(frame, big)
                prediction, confidence = predict(sign)
        except:
            img = frame
        return prediction, confidence

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

model = keras.models.load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(model)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/get_prediction')
def getprediction():
    prediction, confidence = VideoCamera(model).get_prediction()
    if prediction is not None and confidence is not None:
        return jsonify({'prediction': labelToText[int(prediction)], 'confidence': float(confidence)})
    else:
        return jsonify({'prediction': None, 'confidence': None})


if __name__ == '__main__':
    app.run(debug=True)
