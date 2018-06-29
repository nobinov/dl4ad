import click
import rospy
import message_filters
from visualization_msgs.msg import *
from std_msgs.msg import String
from rospy_tutorials.msg import HeaderString
from geometry_msgs.msg import Polygon, PolygonStamped, Point32
from keyboard.msg import Key
import time

pub = rospy.Publisher('polygon_node', PolygonStamped, queue_size=10)

msg_poly = PolygonStamped()
poly = Polygon()

but = None

def callback(marker,button):
	print(button)
	key = button
	print(marker)
	if key=='a':
		print(marker.pose.position)
		
		point = Point32()

		msg_poly.header.stamp = rospy.Time.now()
		msg_poly.header.frame_id = "/mocap"
		point.x = marker.pose.position.x
		point.y = marker.pose.position.y
		point.z = marker.pose.position.z

		poly.points.append(point)

		msg_poly.polygon = poly

		print(msg_poly)
	pub.publish(msg_poly)

def callback1(marker):
	global but
	if(but==True):
		print(but)

def callback2(button):
	global but
	print(button.code)
	if button.code==32:
		but = True
		#print(but)
	but = False

def listener():
	rospy.init_node('listener',anonymous=True)
	#marker_sub = message_filters.Subscriber("wand", Marker)
	#button_sub = message_filters.Subscriber("keyboard/keyup", Key)
	#ts = message_filters.TimeSynchronizer([marker_sub,button_sub],10)
	#ts = message_filters.TimeSynchronizer([button_sub],10)
	#ts.registerCallback(callback2)

	rospy.Subscriber("wand", Marker, callback1)
	rospy.Subscriber("keyboard/keyup", Key, callback2)

	rospy.spin()


if __name__ == '__main__':
	print('press a to capture point')
	listener()

	
