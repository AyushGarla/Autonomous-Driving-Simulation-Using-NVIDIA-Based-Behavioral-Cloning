import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()
#we need a middleware to dispatch to a socket.io web app


app = Flask(__name__) #'__main__'

#preprocessing images same as the training.
def img_preprocess(img):
  img = img[60:135, :,:]
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img, (3,3), 0)
  img = cv2.resize(img, (200,66))
  img = img/255
  return img

#we need a event handler
@sio.on('telemetry')
def telemetry(sid, data):
    image = Image.open(BytesIO(base64.b64decode(data['image']))) #buffer module to mimic our data as a normal file to process the data
    #converting to an array
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    send_control(steering_angle, 1.0)


#when there is a connection we want to fireoff an event handler

@sio.on('connect') #message and disconnect are 3 default
def connect(sid, environ):
    print('Connected')
    send_control(0, 0) # (steering angle, throttle)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {'steering_angle': steering_angle.__str__(), 'throttle': throttle.__str__()})

if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
