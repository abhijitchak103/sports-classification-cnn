#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

from utils import load_preprocess_img

#path = 'data/test/baseball/4.jpg'


classes = ['airhockey', 'amputefootball', 'archery', 'armwrestling', 'axethrowing', 'balancebeam', 'barellracing', 'baseball', 'basketball',
'batontwirling', 'bikepolo', 'billiards', 'bmx', 'bobsled', 'bowling', 'boxing', 'bullriding', 'bungeejumping', 'canoeslamon', 'cheerleading',
'chuckwagonracing', 'cricket', 'croquet', 'curling', 'discgolf', 'fencing', 'fieldhockey', 'figureskatingmen', 'figureskatingpairs', 'figureskatingwomen',
'flyfishing', 'football', 'formularacing', 'frisbee', 'gaga', 'giantslalom', 'golf', 'hammerthrow', 'hanggliding', 'harnessracing', 'highjump',
'hockey', 'horsejumping', 'horseracing', 'horseshoepitching', 'hurdles', 'hydroplaneracing', 'iceclimbing', 'iceyachting', 'jaialai', 'javelin',
'jousting', 'judo', 'lacrosse', 'logrolling', 'luge', 'motorcycleracing', 'mushing', 'nascarracing', 'olympicwrestling', 'parallelbar', 'poleclimbing',
'poledancing', 'polevault', 'polo', 'pommelhorse', 'rings', 'rockclimbing', 'rollerderby', 'rollerbladeracing', 'rowing', 'rugby', 'sailboatracing',
'shotput', 'shuffleboard', 'sidecarracing', 'skijumping', 'skysurfing', 'skydiving', 'snowboarding', 'snowmobileracing', 'speedskating', 'steerwrestling',
'sumowrestling', 'surfing', 'swimming', 'tabletennis', 'tennis', 'trackbicycle', 'trapeze', 'tugofwar', 'ultimate', 'unevenbars', 'volleyball', 'watercycling',
'waterpolo', 'weightlifting', 'wheelchairbasketball', 'wheelchairracing', 'wingsuitflying']

#url = 'https://images.pexels.com/photos/3628912/pexels-photo-3628912.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1'

interpreter = tflite.Interpreter(model_path='prediction.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def predict(url):

    X = load_preprocess_img(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_pred = preds[0].tolist()

    return sorted(zip(classes, float_pred), key=lambda x:-x[1])[:5]

# preds.squeeze().to_list()

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)

    return result