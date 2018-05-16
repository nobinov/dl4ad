from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.transforms as transforms
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import json

#import Image
#import ImageDraw
import pandas as pd


#	print(key, value)

jsonpath = 'bochum_000000_000313_gtFine_polygons.json'
js = pd.read_json(jsonpath)
h, w = js['imgHeight'][0], js['imgWidth'][0]
polygon_road = []
for item in js.itertuples(index=True):
	label = getattr(item, 'objects')['label']
	if label=='road':
		polygon = getattr(item, 'objects')['polygon']
		tmp = []
		for i in polygon:
			tmp.append((i[0] , i[1]))
		polygon_road.append(tmp)
		print(label)
		print(polygon)



print(polygon_road)
print(len(polygon_road))


fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
#plt.plot([-w,w],[-h,h])
#ax.add_patch(patches.Rectangle((0,0),w,h))
#rectangle = plt.Rectangle((0,0), w, h, fc='black')
#plt.gca().add_patch(rectangle)


trans = transforms.Affine2D().rotate_deg(90) + ax.transData

x_r = [0,0,w,w]
y_r = [0,h,h,0]
rectangle = Polygon(np.c_[x_r,y_r], facecolor='black', edgecolor='black', transform = trans)
ax.add_patch(rectangle)
plt.fill(0,0,'black')

for pl in polygon_road:
	#pl_objo = Polygon(pl, facecolor='green', edgecolor='green') 
	pl_obj = Polygon(pl, facecolor='red', edgecolor='red', transform = trans)
	#ax.add_patch(pl_objo)
	ax.add_patch(pl_obj)

	#print(pl_objo.get_xy())
	#print(pl_obj.get_xy())
	#print('hehe')
	#print(pl_obj.get_xy()==pl_objo.get_xy())
	#plt.fill(pl,'blue')
	
ax.axis('off')
ax.invert_yaxis()

plt.tight_layout()

canvas = FigureCanvas(fig)
canvas.draw()
data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#print(data)


#ax.plot()

#plt.show()
#fig.canvas.draw()
#data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#print(data)
#plt.imshow(data)
#plt.pause(1000)

#poly_back = Image.new('RGBA',(w,h), (0,0,0,255))
#pdraw = ImageDraw.Draw(poly_back)
#for pl in polygon_road:
#	pdraw.polygon(pl, fill=(255,0,0,255))



print(data.shape)
plt.imshow(data)
#plt.show()
plt.pause(1000)