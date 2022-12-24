#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np

class Obstacles(Node):
    def __init__(self, delta_r=0.25):
        super().__init__('bus_monitor')             
        self.delta_r = delta_r

        self.publisher = self.create_publisher(MarkerArray, 'obstacles', 1)    
        self.subscription = self.create_subscription(LaserScan, 'scan', self.lidar_centroids, 1)
        self.subscription

    def lidar_centroids(self, msg):       
        
        ranges = np.array(msg.ranges)  # Convert to Numpy array for vector operations
        if len(ranges)==0:
            return
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min # Angle of each ray
        x = ranges * np.cos(angles) # vector arithmatic is much faster than iterating
        y = ranges * np.sin(angles)

        # <...> Here is where you find the obstacles
        
        x_total = []
        y_total = []
        n_total = []
        sum_x = 0
        sum_y = 0
        count = 1
        in_range = ranges < np.inf
        r = 0
        
        for i in range(1,len(ranges)):
            if (ranges[i] != np.inf) and (ranges[i-1] != np.inf):
                if abs(ranges[i] - ranges[i-1]) < 0.25:
                    
                    sum_x+=x[i]
                    sum_y+=y[i]
                    count+=1
            else:
                if count != 1:
                    x_total.append(sum_x)
                    y_total.append(sum_y)
                    n_total.append(count)
                    sum_x = 0
                    sum_y = 0
                    count = 1
        xcen = []#np.array(x_total)/np.array(n_total)
        ycen = []#np.array(y_total)/np.array(n_total)
        for i in range(len(x_total)):
            xcen.append(x_total[i]/n_total[i])
            ycen.append(y_total[i]/n_total[i])
        '''
        on_obj = False
        for i in range(len(ranges)):
            if in_range[i] is True:
                if on_obj is True:
                    if abs(ranges[i]- r) < self.delta_r:
                        n_total[-1] = n_total[-1] + 1
                        x_total[-1] = x_total[-1] + x[i]
                        y_total[-1] = y_total[-1] + y[i]
                        
                    else:  
                        n_total.append(1)
                        x_total.append(x[i])
                        y_total.append(y[i])
                        
                else:
                    on_obj = True
                    n_total.append(1)
                    x_total.append(x[i])
                    y_total.append(y[i])
                    
                r = ranges[i]
            else:
                on_obj = False
        
        
        
        n = len(x_total)

        if on_obj and in_range[0]:
            if abs(ranges[0]- r) < self.delta_r:
                x_total[0] = x_total[0] + x_total[-1]
                y_total[0] = y_total[0] + y_total[-1]
                n_total[0] = n_total[0] + n_total[-1]
                n = len(x_total) - 1
        '''
        # Calculate xcen and ycen: two numpy arrays with coordinates of object centers

        #xcen = np.array(x_total)#/np.array(n_total)   # This makes an obstacle at the origin.  Change this to actual obstacle centroids
        #ycen = np.array(y_total)#/np.array(n_total)
        print("xcen----- ",xcen)
        print("ycen ------",ycen)
        # Convert to list of point vectors
        points = np.column_stack( (xcen, ycen, np.zeros_like(xcen)) ).tolist()

        ids = np.arange(len(points)).astype(int)

        self.pub_centroids(points, ids, msg.header)

      
    def pub_centroids(self, points, ids, header):

        ma = MarkerArray()

        for id, p in zip(ids, points):
            mark = Marker()            
            mark.header = header
            mark.id = id.item()
            mark.type = Marker.SPHERE
            mark.pose = Pose(position=Point(x=p[0],y=p[1],z=p[2]), orientation=Quaternion(x=0.,y=0.,z=0.,w=1.))
            mark.scale.x = 0.25
            mark.scale.y = 0.25
            mark.scale.z = 0.25
            mark.color.a = 0.75
            mark.color.r = 0.25
            mark.color.g = 1.
            mark.color.b = 0.25
            mark.lifetime = Duration(seconds=0.4).to_msg()
            ma.markers.append(mark)

        self.publisher.publish( ma )

def main(args=None):

    rclpy.init(args=args)

    node = Obstacles()
    rclpy.spin(node) 

