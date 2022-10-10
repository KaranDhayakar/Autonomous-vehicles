#!/usr/bin/env python
''' bot_monitor.py

    To save plot to bot_plot.png do:
     ros2 param set /bot_monitor bot_plot save
    To clear plot do:
     ros2 param set /bot_monitor bot_plot clear

    <...> Complete missing portions

    Copyright: Daniel Morris, 2020
'''
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import time

class PlotOdom(Node):
    ''' Reads and plots odometry '''

    def __init__(self):
        super().__init__('bot_monitor')        
        self.deltaPos = 0.05
        self.x = 0 
        self.y = 0
        # <...> initialize some parameters to store previously plotted points
        self.isinit = False
        self.declare_parameter('bot_plot','')

        # <...> Create a subscription:
        self.subscription = self.create_subscription(Odometry,"/odom",self.callback,10)
        self.subscription         
        self.get_logger().info("Subscribing to /odom")


    def callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # <...> extract the robot position
        doPlot = (x-self.x)**2 + (y-self.y)**2 > self.deltaPos**2
        # <...> set doPlot to be true if moved at least some minimum distance from previously plotted position
        
        if not self.isinit: 
            # Do plot initialization in callback to keep all plotting 
            # in the same thread            
            doPlot = True  # Always plot if doing initialization
            self.isinit = True 
            figsel = 'Odom'
            fig = plt.figure(figsel,figsize=(4,4))
            fig.clf()
            plt.subplots(1,1,num=figsel)    
            plt.grid(True, linestyle='--')
            plt.gca().set_aspect('equal', 'box')
            plt.gcf().canvas.flush_events()
            plt.show(block=False)
            plt.show(block=False) # For some reason calling this twice is necessary
            
        if doPlot:
            self.x = x
            self.y = y
	    # <...> store new position as previously plotting position 
            plt.plot(self.x, self.y, 'r.')
            # <...> plot current position

            plt.gcf().canvas.flush_events()
            plt.show(block=False)
            plt.show(block=False)
            time.sleep(0.01)
            print("do plot")
        
        # Handle saving plot to png and clearing plot
        val = self.get_parameter('bot_plot').get_parameter_value().string_value
        if val:   # if not empty parameter:
            if val=="save":
                plt.savefig("bot_plot.png")
                self.get_logger().info("Saving plot to bot_plot.png")
            elif val=="clear":
                plt.cla()
                plt.grid(True, linestyle='--')
                plt.gcf().canvas.flush_events()
                plt.show(block=False)
                plt.show(block=False) 
                self.get_logger().info("Clearing bot_monitor plot")
            else:
                self.get_logger().info(f"Unrecognized parameter value: {val}")
            # Set parameter to empty value:
            new_param_val = Parameter('bot_plot',rclpy.Parameter.Type.STRING,'')
            self.set_parameters([new_param_val])

def main(args=None):
    rclpy.init(args=args)

    bm = PlotOdom()
    rclpy.spin(bm)
