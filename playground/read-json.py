from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import cv2
import Image
import ImageDraw


#aachen_000000_000019_gtFine_polygons.json

with open('aachen_000000_000019_gtFine_polygons.json') as f:
	img_json = json.load(f)
f.close()

imgH = img_json['imgHeight']
imgW = img_json['imgWidth']
roads = img_json['objects']['label'=='car']['polygon']
#obj = img_json['objects']['label'=='car']
#for key, value in obj.items():
#	print(key, value)



print(roads)

img_poly = []
for i in roads:
	img_poly.append((i[0] , i[1]))

print(img_poly)
print(roads)
print(len(roads))

#back = Image.new('RGBA',(imgW,imgH), (0,0,0,255))
poly = Image.new('RGBA',(imgW,imgH), (0,0,0,255))
pdraw = ImageDraw.Draw(poly)
pdraw.polygon(img_poly, fill=(255,0,0,255))
#back.paste(poly,mask=poly)
#back.show()
plt.imshow(poly)
plt.pause(1000)
