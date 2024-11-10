import io
import time
import json
from aiohttp import web
from PIL import Image
from ultralytics import YOLO
import base64
import pickle


version = "0.0.1"
item_types = ['kids',
'life zone',
'No entry',
'Give Way',
'main road',
'autobus',
'uneven road',
'Movement Prohibition',
'Gas station',
'No stopping',
'trucks',
'Pedestrian crossing',
'Intersection with a bicycle path',
'Speed ​​limit 20',
'Speed ​​limit 40',
'bike road',
'Speed ​​hump']
model = YOLO("./best_cars.pt")

# GET /status
async def status(req):
    return web.json_response({
        "name": "name of detector",
        "type": "detector",
        "path": "/",
        "version": version,
        "output": {
            "types": item_types,
        },
    })
    
def extract_boxes(result_boxes):
    kmindex = "3.0000"
    _boxes = []
    for box in result_boxes:
        if kmindex in str(box.data):
            _boxes.append(box)
    return _boxes            

def json2im(file):
    """Convert a JSON string back to a Numpy array"""
    imdata = base64.b64decode(file)
    im = pickle.loads(imdata)
    return im

# POST /
async def parser(req):
    at = time.time()
    data = await req.post()
    print('start')
    file = data['frame']
    file = json2im(file)
    img = Image.fromarray(file)
    print(img.width, img.height)
    results = model.predict(img, task="detect", device='0')
    data = list()
    
    result_boxes = results[0].boxes

    print("------------------------------------------------------")
    for box in result_boxes:
        x1 = box.xywh[0][0].item()
        xs = int(x1)
        y1 = box.xywh[0][1].item()
        ys = int(y1)
        w = box.xywh[0][2].item()
        ws = int(w)
        h = box.xywh[0][3].item()
        hs = int(h)
        xs2=xs-ws/2
        ys2=ys-hs/2
        print(int(box.cls.cpu()))
        data.append({'name': item_types[int(box.cls.cpu())], 'bbox' : [xs,ys,xs + ws,ys + hs]})


    resp = {   
        "predicts": data,
        "version": version,
        "time": time.time() - at,
    }
    print(resp)
    return web.json_response(resp)


app = web.Application(client_max_size=1024**2*4)

app.add_routes([
    web.get('/status', status),
    web.post('/', parser),
])


if __name__ == '__main__':
    web.run_app(app, host="localhost", port=64471)
