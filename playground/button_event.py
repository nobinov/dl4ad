import click
import rospy
from std_msgs.msg import String
from rospy_tutorials.msg import HeaderString

def button_talker():
	pub = rospy.Publisher('button',HeaderString, queue_size=10)
	rospy.init_node('talker', anonymous=True)
	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		key = click.getchar()
		if key=='a':	
			print('get')
			msg = HeaderString()
			msg.data = key
			msg.header.seq = 1
			msg.header.stamp = rospy.Time.now()
			pub.publish(msg)
			rate.sleep()

if __name__ == '__main__':
	button_talker()