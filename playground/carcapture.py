import rospy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
import tf
from tf import ExtrapolationException, ConnectivityException
import time
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import os
import pandas as pd
import sys
import signal

#rosrun image_transport republish compreed in:=output/image raw out:=camera_out/image
 
class CameraPose(object):

	def __init__(self):
		self.tf_listener = tf.TransformListener()
		self.num = 0
		self.broken_amt = 0

		self.imgseq = []
		self.timestamp = []
		self.broken = []
		self.x = []
		self.y = []
		self.z = []
		self.w = []

	def callback_camerainfo(self,prop):
		#print('got image')
		self.gotimage = True
		print(prop.header.stamp)
		print(rospy.Time.now())
		self.num=self.num+1
		print(self.num)
		#(trans,rot) = self.tf_listener.lookupTransform('/mocap','front_cam',prop.header.stamp)
		#print(rot)

	def callback_camera(self,img):
		namefile = '{}{:06d}{}'.format('car02-frame',self.num,'.png')
		try:
			print('got image')
			self.num=self.num+1
			np_arr = np.fromstring(img.data, np.uint8)
			image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		except CvBridgeError as e:
			print(e)
		else:
			#cv2.imshow("window",image_np)
			#cv2.imwrite('test.jpg',image_np)
			#print("saving car01-frame%06i.png" % self.num)
			print("saving "+namefile)
			cv2.imwrite(os.path.join('/home/novian/catkin_ws/src/bagfile/car-02', namefile), image_np)

		rot = [0.0,0.0,0.0,0.0]
		broken = False
		try:
			(trans,rot) = self.tf_listener.lookupTransform('/mocap','front_cam',img.header.stamp)
		except ExtrapolationException as e:
			print(e)
			broken=True
			self.broken_amt+=1
		except ConnectivityException as e:
			print(e)
			broken=True
			self.broken_amt+=1


		self.imgseq.append(self.num)
		self.timestamp.append(img.header.stamp)
		self.broken.append(broken)
		self.x.append(rot[0])
		self.y.append(rot[1])
		self.z.append(rot[2])
		self.w.append(rot[3])
		print(self.broken_amt)

	def hook(self):
		print('shutdown')
		rawdata = {
			'imgseq':self.imgseq,
			'timestamp':self.timestamp,
			'broken':self.broken,
			'x':self.x,
			'y':self.y,
			'z':self.z,
			'w':self.w}
		df = pd.DataFrame(rawdata,columns=['imgseq','timestamp','broken','x','y','z','w'])
		df.to_csv('car-02.csv')

	def listener(self):
		rospy.init_node('listener',anonymous=True)
		#rospy.Subscriber("output/camera_info", CameraInfo, self.callback_camerainfo)
		rospy.Subscriber("output/image/compressed", CompressedImage, self.callback_camera)
		rospy.on_shutdown(self.hook)
		rospy.spin()


if __name__ == '__main__':
	cp = CameraPose()
	cp.listener()



