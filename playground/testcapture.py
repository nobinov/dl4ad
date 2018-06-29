import click
import rospy
import message_filters
from visualization_msgs.msg import *
from std_msgs.msg import String
from rospy_tutorials.msg import HeaderString
from geometry_msgs.msg import PolygonStamped
import time

key = ''
arr = []
global datas
global getbutton
pub = rospy.Publisher('polygon_node', PolygonStamped)

def callback_marker(data):
	datas = data.pose
	#rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.id)	
	#print(datas)

def callback_button(data):
	getbutton = data.data
	if getbutton=='a':
		print('got signal')
		print(datas)	
	#print(getbutton)
	#print(datas)

def callback(marker):
	key = click.getchar()
	if key=='a':
		print(marker.pose)



def listener():
	rospy.init_node('listener',anonymous=True)
	marker_sub = message_filters.Subscriber("measure_point", Marker)
	#button_sub = message_filters.Subscriber("button", HeaderString)
	ts = message_filters.TimeSynchronizer([marker_sub],10)
	ts.registerCallback(callback)
	rospy.spin()


if __name__ == '__main__':
	listener()

	
