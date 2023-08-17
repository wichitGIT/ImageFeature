# from fastapi import FastAPI , Request
# import cv2
# import numpy as np
# import base64
# from app.hog import gethog

# app = FastAPI()

# def read64(uri):
#     encoded_data = uri.split(',')[1]
#     nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
#     return img

# @app.get("/api/gethog")
# async def read_str(request : Request):
#     item = await request.json()
#     item_str = item['img']
#     img = read64(item_str)
#     hog = gethog(img)
#     return {"hog":hog.tolist()}


import cv2
import numpy as np
from fastapi import FastAPI
import base64

from pydantic import BaseModel

app = FastAPI()

# @app.get("/")
# def root():
#     return {"message": "This is my api"}

def decodeBase64(img_str):
    encoded_data = img_str.split(',')[1]
    img_byte = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(img_byte, cv2.IMREAD_GRAYSCALE)

    return img

@app.get("/api/")
def read_img ():
    return'hihihi'


    
# class ImageClass(BaseModel):
#     image_base64: str

@app.get("/api/genhog/")
def read_image(image_str):
    img = decodeBase64(image_str)
    # Load the image as grayscale
    img_gray = img
    img_new = cv2.resize(img_gray, (128,128), cv2.INTER_AREA)
    win_size = img_new.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9

    # Set the parameters of the HOG descriptor using the variables defined above
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride,cell_size, num_bins)
    # Compute the HOG Descriptor for the gray scale image
    hog_descriptor = hog.compute(img_new)

    return {'HOG Descriptor': hog_descriptor.tolist()}
    # return image