#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

class Publisher:
    def __init__(self):
        # Internal variables
        self.rate = rospy.Rate(rospy.get_param('~rate', 2.0))

        # Publishers
        self.pub = rospy.Publisher('/message', String, queue_size=10)

        # Parameters
        self.message = rospy.get_param('~message', 'Hello World!')

    def run(self):
        while not rospy.is_shutdown():
            self.pub.publish(self.message)
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('publisher')
    node = Publisher()
    node.run()