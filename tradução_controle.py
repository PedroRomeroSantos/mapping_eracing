#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
from fs_msgs.msg import ControlCommand

def callback(msg):
    #verifica 3 valores [steering, throttle, brake]
    if len(msg.data) >= 2:
        cmd = ControlCommand()
        cmd.header.stamp = rospy.Time.now()
        
        # índices corretos
        cmd.steering = float(msg.data[0])
        cmd.throttle = float(msg.data[1])
        cmd.brake = 0.0 # Force o freio em zero absoluto
        
        pub.publish(cmd)
        #conferir no terminal do ROS 1
        # rospy.loginfo("T: %f | S: %f", cmd.throttle, cmd.steering)

rospy.init_node('bridge_translator')
pub = rospy.Publisher('/fsds/control_command', ControlCommand, queue_size=1)
sub = rospy.Subscriber('/cmd_bridge', Float32MultiArray, callback)
rospy.spin()
