#!/usr/bin/env python3
"""
Simple action selection mechanism inspired by the K-bandit problem

Initially, MiRo performs one of the following actions on random, namely: 
wiggle ears, wag tail, rotate, turn on LEDs and simulate a Braitenberg Vehicle.

While an action is being executed, stroking MiRo's head will reinforce it, while  
stroking MiRo's body will inhibit it, by increasing or reducing the probability 
of this action being picked in the future.

NOTE: The code was tested for Python 2 and 3
For Python 2 the shebang line is
#!/usr/bin/env python
"""
# Adapted using COM3528 example code from https://github.com/AlexandrLucas/COM3528.git (k bandit)
# Original code by Professor Alexandr Lucas, COM3528, University of Sheffield
# referenced COM3528 and COM2009 github wikis for rospy and miro development information

#NOTE: cannot be run on the Python environment directly
# A MiRo robot and laptop setup with ROS from Diamond are needed
# Imports
##########################
import os
from re import X
import numpy as np

import rospy  # ROS Python interface
from std_msgs.msg import (
    Float32MultiArray,
    UInt32MultiArray,
    UInt16,
    #new
    Int16MultiArray,
)  # Used in callbacks
from geometry_msgs.msg import TwistStamped  # ROS cmd_vel (velocity control)
#new
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from tf.transformations import euler_from_quaternion

import miro2 as miro  # MiRo Developer Kit library
try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel  # Python 2

import random

# new
# from tabnanny import verbose
import speech_recognition as sr
# import pocketsphinx
# from pocketsphinx import LiveSpeech
import nlp_bigram_ver
from nlp_bigram_ver import parse
from math import pi
import sys
import os
from os import path
##########################

mode = "mic"

if __name__=="__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "text":
           mode = "text"
        if sys.argv[1] == "audio":
            mode = "audio"
        if sys.argv[1] == "mic":
            mode = "mic"
        if sys.argv[1] == "debug":
            mode = "debug"
    else:
        red_flag = True
        print_flag = False
        maxsim = 0.1
        mode = "mic"
    if len(sys.argv) > 2:
        print_flag = sys.argv[2] == 'True'
    else:
        red_flag = True
        print_flag = False
        maxsim = 0.1
    if len(sys.argv) > 3:
        red_flag = sys.argv[3] == "True"
    else:
        red_flag = True
        maxsim = 0.1
    if len(sys.argv) > 4:
        maxsim = float(sys.argv[4])
    else:
        maxsim = 0.1

class MiRoClient:

    # Script settings below
    TICK = 0.02  # Main loop frequency (in secs, default is 50Hz)
    RAW_DURATION = 0.5 # duration value in seconds as float
    ACTION_DURATION = rospy.Duration(RAW_DURATION)  # seconds
    VERBOSE = True  # Whether to print out values of Q and N after each iteration
    ##NOTE The following option is relevant in MiRoCODE
    NODE_EXISTS = False  # Disables (True) / Enables (False) rospy.init_node

    def __init__(self):
        """
        Class initialisation
        """
        print("Initialising the controller...")

        # Get robot name
        topic_root = "/" + os.getenv("MIRO_ROBOT_NAME")

        # Initialise a new ROS node to communicate with MiRo
        if not self.NODE_EXISTS:
            rospy.init_node("heymiro", anonymous=True)

        # Define ROS publishers
        self.pub_cmd_vel = rospy.Publisher(
            topic_root + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        self.pub_cos = rospy.Publisher(
            topic_root + "/control/cosmetic_joints", Float32MultiArray, queue_size=0
        )
        self.pub_illum = rospy.Publisher(
            topic_root + "/control/illum", UInt32MultiArray, queue_size=0
        )

        # Define ROS subscribers
        rospy.Subscriber(
            topic_root + "/sensors/light",
            Float32MultiArray,
            self.lightCallback,
        )
        #new
        rospy.Subscriber(
            topic_root + "/sensors/odom",
            Odometry,
            self.odomCallback,
        )
        rospy.Subscriber(
            topic_root + "/sensors/mics",
            Int16MultiArray,
            self.microphoneCallback,
        )
        rospy.Subscriber(
            topic_root + "/sensors/sonar",
            Range,
            self.sonarCallback,
        )
        rospy.Subscriber(
            topic_root + "/sensors/cliff",
            Float32MultiArray,
            self.cliffCallback,
        )

        # Initialise objects for data storage and publishing
        self.light_array = None
        self.velocity = TwistStamped()
        self.cos_joints = Float32MultiArray()
        self.cos_joints.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.illum = UInt32MultiArray()
        self.illum.data = [
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
        ]
        #new
        self.odom_array = None
        self.microphone_array = None
        self.sonar_range = None
        self.cliff_array = None

        # Utility enums
        self.tilt, self.lift, self.yaw, self.pitch = range(4)
        (
            self.droop,
            self.wag,
            self.left_eye,
            self.right_eye,
            self.left_ear,
            self.right_ear,
        ) = range(6)

        self.status_code = -1
        self.comms = []

        # Give it a sec to make sure everything is initialised
        rospy.sleep(1.0)

    def earWiggle(self, t0):
        print("MiRo wiggling ears")
        A = 1.0
        w = 2 * np.pi * 0.2
        f = lambda t: A * np.cos(w * t)
        i = 0
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.cos_joints.data[self.left_ear] = f(i)
            self.cos_joints.data[self.right_ear] = f(i)
            self.pub_cos.publish(self.cos_joints)
            i += self.TICK
            rospy.sleep(self.TICK)
        self.cos_joints.data[self.left_ear] = 0.0
        self.cos_joints.data[self.right_ear] = 0.0
        self.pub_cos.publish(self.cos_joints)

    def tailWag(self, t0):
        print("MiRo wagging tail")
        A = 1.0
        w = 2 * np.pi * 0.2
        f = lambda t: A * np.cos(w * t)
        i = 0
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.cos_joints.data[self.wag] = f(i)
            self.pub_cos.publish(self.cos_joints)
            i += self.TICK
            rospy.sleep(self.TICK)
        self.cos_joints.data[self.wag] = 0.0
        self.pub_cos.publish(self.cos_joints)

    def rotate(self, t0):
        print("MiRo rotating left")
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.velocity.twist.linear.x = 0
            self.velocity.twist.angular.z = (pi/4)/self.RAW_DURATION#2.1
            self.pub_cmd_vel.publish(self.velocity)
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)
    
    def rotate_alt(self, t0):
        print("MiRo rotating right")
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.velocity.twist.linear.x = 0
            self.velocity.twist.angular.z = -(pi/4)/self.RAW_DURATION#-2.1
            self.pub_cmd_vel.publish(self.velocity)
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)

    # def rotate(self, t0, init_yaw):
    #     print("MiRo rotating left")
    #     if self.odom_array == None:
    #         return
    #     #init_yaw = self.odom_array[5]
    #     while self.odom_array[5] > init_yaw-pi/2:
    #         self.velocity.twist.linear.x = 0
    #         self.velocity.twist.angular.z = 2.1
    #         self.pub_cmd_vel.publish(self.velocity)
    #     self.velocity.twist.linear.x = 0
    #     self.velocity.twist.angular.z = 0
    #     self.pub_cmd_vel.publish(self.velocity)
    
    # def rotate_alt(self, t0, init_yaw):
    #     print("MiRo rotating right")
    #     if self.odom_array == None:
    #         return
    #     #init_yaw = self.odom_array[5]
    #     while self.odom_array[5] > init_yaw-pi/2:
    #         self.velocity.twist.linear.x = 0
    #         self.velocity.twist.angular.z = -2.1
    #         self.pub_cmd_vel.publish(self.velocity)
    #     self.velocity.twist.linear.x = 0
    #     self.velocity.twist.angular.z = 0
    #     self.pub_cmd_vel.publish(self.velocity)

    def shine(self, t0):
        print("MiRo turning on LEDs")
        color = random.choices([
            0xFF00FF00,
            0xFFFF0000,
            0xFF0000FF,
            0xFFFF00FF,
            0xFF00FFFF,
            0xFFFFFF00
        ], k=6)
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            color = random.choices([
                0xFF00FF00,
                0xFFFF0000,
                0xFF0000FF,
                0xFFFF00FF,
                0xFF00FFFF,
                0xFFFFFF00
            ], k=6)
            self.illum.data = color
            self.pub_illum.publish(self.illum)
        self.illum.data = [0x00000000]*6
        self.pub_illum.publish(self.illum)
    

    # needs revision for more efficient implementation. Current version is modified from braitenberg2a
    def moveforward(self, t0):
        print("MiRo moves forward")
        wheel_speed = [1, 1]
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            #print("hi")
            if self.sonar_range <= 0.05 or self.cliff_array[0]<0.2 or self.cliff_array[1]<0.2:
                print("Emergency Stop!")
                print("sonar: ", self.sonar_range)
                print("cliff: ", self.cliff_array[0], " ", self.cliff_array[1])
                self.velocity.twist.linear.x = 0
                self.velocity.twist.angular.z = 0
                self.pub_cmd_vel.publish(self.velocity)
                rospy.sleep(t0 + self.ACTION_DURATION - rospy.Time.now())
            self.velocity.twist.linear.x = dr
            self.velocity.twist.angular.z = dtheta
            self.pub_cmd_vel.publish(self.velocity)
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)

    def lightCallback(self, data):
        """
        Get the frontal illumination
        """
        if data.data:
            self.light_array = data.data

    #new
    def microphoneCallback(self, data):
        """
        Get the microphone input in the form [left, right centre, tail]
        """
        if data.data:
            self.microphone_array = data.data
            #print("mic in: ", self.microphone_array)
    def sonarCallback(self, data):
        """
        get sonar range value
        normal values are in the range 0.03m to 1.0m
        0.0 means a very close echo, below 0.03m and not reliably accurate
        infinity means no echo received
        """
        if data:
            self.sonar_range = data.range
            #print("sonar: ", self.sonar_range)
    def cliffCallback(self, data):
        """
        Get the frontal illumination
        """
        if data.data:
            self.cliff_array = data.data
            #print("cliff sensor: ", self.cliff_array)
    
    def odomCallback(self, data):
        """
        Odometry information
        """
        if data:
            orientation_x = data.pose.pose.orientation.x
            orientation_y = data.pose.pose.orientation.y
            orientation_z = data.pose.pose.orientation.z
            orientation_w = data.pose.pose.orientation.w

            position_x = data.pose.pose.position.x
            position_y = data.pose.pose.position.y
            position_z = data.pose.pose.position.z
            
            (roll, pitch, yaw) = euler_from_quaternion([orientation_x, orientation_y, orientation_z, orientation_w], 'sxyz')

            robot_odom = [position_x, position_y, position_z, roll, pitch, yaw]
            #print("odom: ",robot_odom)
            #print(rospy.Time.now())

            self.odom_array = robot_odom
        
        

    def loop(self):
        """
        Main loop
        """
        print("Starting the loop")
        r = sr.Recognizer()

        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

        self.status_code = -1
        debug_val = 0

        while not rospy.core.is_shutdown():
            print("NEW LOOP")

            #recognize speech using Sphinx
            try:
                if self.status_code == -1:
                    print("status: ", self.status_code)
                    test_input = [
                    "go left and then right",
                    "go left then right then forward",
                    "do not go left and instead go right",
                    "after going left go forward and then right",
                    "do not go left and do not go right only go forward",
                    "go forward and then right not left",
                    "once you have made a left turn go forward and then turn right",
                    "turn left and then right but don't move forward",
                    "once you've gone left go right",
                    "turn right move forward then turn left",
                    "turn left turn left again move forward and then turn right and right again",
                    "turn left then right then left then right then do not go forward",
                    "move forward then turn left move forward again and then turn right and move forward again",
                    "turn right go forward and then go forward some more",
                    "move forward then make a left turn and go forward without turning right",
                    "shine the room please",
                    "I think you are good Miro",
                    "I think you are bad Miro"
                    ]
                    transcription = ""
                    if mode == "mic:":
                        print("adjusting for ambient noise, please do not talk")
                        with sr.Microphone() as source:
                                r.adjust_for_ambient_noise(source, duration=5)
                        print("ambient noise adjustment complete")
                        with sr.Microphone() as source:
                            print("Say something!")
                            #r.adjust_for_ambient_noise(source, duration=5)
                            audio = r.listen(source)
                        transcription = r.recognize_sphinx(audio)
                    elif mode == "audio":
                        audiofile = input("enter audio file name:\n")
                        #print(path.dirname(path.realpath(__file__)))
                        AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), audiofile)
                        print(AUDIO_FILE)
                        r = sr.Recognizer()
                        with sr.AudioFile(AUDIO_FILE) as source:
                            audio = r.record(source)  # read the entire audio file

                        # recognize speech using Sphinx
                        try:
                            transcription = r.recognize_sphinx(audio)
                        except sr.UnknownValueError:
                            print("Sphinx could not understand audio")
                        except sr.RequestError as e:
                            print("Sphinx error; {0}".format(e))
                    elif mode == "text":
                        print("enter your sentence:")
                        transcription = input()                    
                    elif mode == "debug":
                        transcription = random.choice(test_input)
                    else:
                        print("adjusting for ambient noise, please do not talk")
                        with sr.Microphone() as source:
                                r.adjust_for_ambient_noise(source, duration=5)
                        print("ambient noise adjustment complete")
                        with sr.Microphone() as source:
                            print("Say something!")
                            #r.adjust_for_ambient_noise(source, duration=5)
                            audio = r.listen(source)
                        transcription = r.recognize_sphinx(audio)
                    #transcription = debug_input[debug_val]
                    debug_val += 1
                    if debug_val > 1: debug_val = 0
                    #transcription = test_input[5]
                    print("you said " + transcription)
                    #if transcription == "goodbye": break # debug
                    self.comms = parse(transcription, print_flag=print_flag, red_check=red_flag, maxsimamount=maxsim)
                    self.status_code = 0
                elif self.status_code != -1:
                    print("status: ", self.status_code)
                    if self.status_code >= len(self.comms):
                        self.comms = []
                        self.status_code = -1
                        continue
                    print(self.comms)
                    start_time = rospy.Time.now()
                    c = self.comms[self.status_code]
                    print("c: ", c)
                    if c == "left":
                        self.rotate(start_time) # need to test and distinguish left and right as well as get a 90 degree turn
                    elif c == "right":
                        self.rotate_alt(start_time) # need to test and distinguish left and right as well as get a 90 degree turn
                    elif c == "forward":
                        self.moveforward(start_time)
                    elif c == "light":
                        self.shine(start_time)
                    elif c == "wagtail":
                        self.tailWag(start_time)
                    elif c == "ear":
                        self.earWiggle(start_time)
                    self.status_code += 1
                    rospy.sleep(self.ACTION_DURATION)
            except sr.UnknownValueError:
                print("could not understand audio")
            except sr.RequestError as e:
                print("error; {0}".format(e))


# This is run when the script is called directly
if __name__ == "__main__":
    main = MiRoClient()  # Instantiate class
    main.loop()  # Run the main control loop
