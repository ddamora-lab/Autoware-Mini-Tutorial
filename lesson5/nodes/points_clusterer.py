#!/usr/bin/env python3

import rospy
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from sklearn.cluster import DBSCAN

from ros_numpy import numpify, msgify
from sensor_msgs.msg import PointCloud2


class PointsClusterer:
    def __init__(self):
        # Parameters
        self.cluster_epsilon = rospy.get_param('~cluster_epsilon')
        self.cluster_min_samples = rospy.get_param('~cluster_min_samples')

        self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_samples)

        # Publishers
        self.clustered_pub = rospy.Publisher('points_clustered', PointCloud2, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())

    def points_callback(self, msg):

        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        if len(points) == 0:
            return
        labels = self.clusterer.fit_predict(points)

        points_labeled = np.hstack((points, labels.reshape(-1, 1)))
        points_labeled = points_labeled[points_labeled[:, 3] != -1]

        data = unstructured_to_structured(points_labeled, dtype=np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('label', np.int32)
        ]))

        clustered_msg = msgify(PointCloud2, data)
        clustered_msg.header.stamp = msg.header.stamp
        clustered_msg.header.frame_id = msg.header.frame_id

        self.clustered_pub.publish(clustered_msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('points_clusterer', log_level=rospy.INFO)
    node = PointsClusterer()
    node.run()