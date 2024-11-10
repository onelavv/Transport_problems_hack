import asyncio
import base64
import json
import multiprocessing
import pickle

import aiohttp
import cv2
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from flask import request
import threading
import imutils
import datetime


lock = asyncio.Lock()
app = Flask(__name__)
image_source = "0"
vs = VideoStream(src=image_source).start()
# vs=None
stop_flag = multiprocessing.Event()
stop_flag.clear()
t = []
loop = asyncio.new_event_loop()
neiro_enabled = ["0"]

@app.route("/", methods=['post', 'get'])
async def index():
    # return the rendered template
    global image_source, vs, t, stop_flag, neiro_enabled
    if request.method == 'POST':
        src = request.form.get('source')
        neiro_enabled = (request.form.getlist('options'))
        if not len(neiro_enabled): neiro_enabled = ['0']
        print(type(neiro_enabled), neiro_enabled)
        if src:
            image_source = src  # запрос к данным формы
        print(image_source, len(t))
        vs.stop()
        vs = VideoStream(src=image_source).start()
    return render_template("index.html")


async def get_predict_and_draw_box(frame):
    if int(neiro_enabled[0]):
        dict_to_send = {'frame': base64.b64encode(pickle.dumps(frame)).decode('utf-8')}
        async with aiohttp.ClientSession() as session:
            response = await session.post("http://localhost:64471", data=dict_to_send)
            dict_from_server = await response.json()

        print(dict_from_server)
        return dict_from_server
    else:
        return {}


async def prepare_image(orig_image, width):
    prepared = imutils.resize(orig_image, width=width)
    resp = await get_predict_and_draw_box(prepared)
    print(resp, type(resp))
    timestamp = datetime.datetime.now()
    cv2.putText(prepared, timestamp.strftime(
        "%A %d %B %Y %I:%M:%S%p"), (10, prepared.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.705, (0, 0, 0), 1)
    cv2.putText(prepared, timestamp.strftime(
        "%A %d %B %Y %I:%M:%S%p"), (10, prepared.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    if "predicts" in resp.keys():
        for prediction in resp['predicts']:
            cv2.putText(prepared, prediction['name'], (prediction['bbox'][0], prediction['bbox'][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.705, (0, 0, 0), 1)
            cv2.putText(prepared, prediction['name'], (prediction['bbox'][0], prediction['bbox'][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.rectangle(prepared, (prediction['bbox'][0], prediction['bbox'][1]), (prediction['bbox'][2], prediction['bbox'][3]), (0, 0, 255), 2)
    return prepared

async def generate(width: int):
    # grab global references to the output frame and lock variables
    global vs
    # loop over frames from the output stream
    while True:
        outputFrame = vs.read()
        # print(outputFrame)
        if outputFrame is None:
            continue
        # encode the frame in JPEG format
        image = await prepare_image(outputFrame, width)
        # print(image)
        (flag, encoded_image) = cv2.imencode(".jpg", image)
        # ensure the frame was successfully encoded
        if not flag:
            continue
    # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encoded_image) + b'\r\n')

loop = asyncio.get_event_loop()

# Backward compatible sync generator
def get_results(width):
    gen = generate(width)
    while True:
        try:
            yield loop.run_until_complete(gen.__anext__())
        except StopAsyncIteration:
            break

@app.route("/video_feed")
async def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(get_results(640),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

# @app.route("/video_feed_original")
# async def video_feed_original():
#     # return the response generated along with the specific media
#     # type (mime type)
#     return Response(await generate(3840),
#         mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # t.append(multiprocessing.Process(target=detect_motion, args=(vs, )))
    # t[0].daemon = True
    # t[0].start()
    # asyncio.run(detect_motion(vs))
    # start the flask app
    app.run(host="localhost", port=5000, debug=True, use_reloader=True, threaded=True)
    vs.stop()