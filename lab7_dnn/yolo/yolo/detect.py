import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
import cv2
import time
import os
import argparse
import numpy as np
from .dnn_detect import Dnn
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Detect(Node):
	def __init__(self,topic_name='text_message',nms=0.4):#here     
		super().__init__('detect')        
		self.subscription = self.create_subscription(Image,topic_name,self.callback,1)# change '/cam_front/raw' 
		self.subscription

		self.get_logger().info(f'Subscribed to: {topic_name}')
		self.dnn_img = Dnn(nms)	

	def callback(self, img_data):
		image_message = img_data
		bridge = CvBridge()
		cv_image = bridge.imgmsg_to_cv2(image_message, 'bgr8')
		cv2.waitKey(1)
		frame = cv_image
		classes, scores, boxes = self.dnn_img.detect(frame)
		self.dnn_img.draw(frame, classes, scores, boxes)
		return

def main(args=None):
	rclpy.init(args=args)
	parser = argparse.ArgumentParser(description='Image type arguments')
	parser.add_argument('--nms',      default=0.4,    dest='nms',     type=float, help="add int")
	parser.add_argument('topic_name',      default='/cam_front/raw',     type=str, help="Image topic to subscribe to")
	args, unknown = parser.parse_known_args() 
	if unknown: print('Unknown args:',unknown)
	nms = args.nms
	topic_name = args.topic_name
	dtct = Detect(topic_name=topic_name,nms=nms)
	rclpy.spin(dtct)






