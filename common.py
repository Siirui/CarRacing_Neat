'''
Description: this file defines some common data and functions
Date: 2022-06-09 16:45:38
LastEditTime: 2022-06-11 16:25:59
'''

from locale import atoi
import string
from typing import List, Tuple
import pygame
import math


# the window default size
WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 720

# graphical settings
CAR_HEIGHT = 20
CAR_WIDTH = 10


GAME_FPS = 60

# about car properties
ANGLE_STEP = 0.03
ACCELERATION_FORWARD = 0.2
MAX_SPEED = 5
CNT_DETECT_LINE=5

#about the game properties
STARTING_POINT = (200,600)

#about the reward
REWARD_ONE_LAP=1000 # reward if the car can finish one lap
DEAD_SCORE=-500 # the dead score of the car



def rot_center(image, rect, angle):
    """rotate an image while keeping its center"""
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = rot_image.get_rect(center=rect.center)
    return rot_image, rot_rect

def is_boundary_color(color):
    if color[0] >= 150 and color[0]<=200 and color[1] >=0 and color[1]<=100 and color[2] == 255:
        return True
    else:
        return False

def calculate_angle(px1,py1,px2,py2):
    """
    calculate the angle between two points
    """
    dx = px2 - px1
    dy = py2 - py1
    rad = math.atan2(dy, dx)
    return rad



def load_map(dir,map_name):
    """
        load the map png file and the starting coordinate
    """
    image_surface=pygame.image.load(dir+map_name+'.png')
    starting_point=[0,0]
    point_list=[]
    #read the coordinate of 2 points
    with open(dir+map_name+'_start.txt','r') as f:
        for line in f:
            point_list.append(line.split())
    for i in range(len(point_list)):
        #transform into number
        starting_point[0]+=float(point_list[i][0])
        starting_point[1]+=float(point_list[i][1])
    starting_point[0]/=len(point_list)
    starting_point[1]/=len(point_list)
    # change to int
    starting_point[0]=int(starting_point[0])
    starting_point[1]=int(starting_point[1])
    return image_surface,starting_point



def auto_drive(speed:float,detect_dist:List)->List[float]:
    """
    this function is a example auto drive function,return the decision with tuple:
    (should_speed_up,should_speed_down,should_turn_left,should_turn_right)
    the input parameter:
        speed:current speed of the car, range:[0,MAX_SPEED=6]
        detect_dist:the distance to the detection line (size:5)
    the output parameter:
        decision:the decision of the car with tuple:
        0: should_speed_up:[0,0.2]
        1: should_speed_down:[0,0.2]
        2: should_turn_left:[0,0.03]
        3: should_turn_right:[0,0.03]
    """
    if speed < MAX_SPEED:
        return (ACCELERATION_FORWARD,0,0,0)
    else:
        return (0,ACCELERATION_FORWARD,0,0)

def simple_auto(speed:float,detect_dist:List)->Tuple[float,float,float,float]:
    decision=[0,0,0,0]

    high_speed=MAX_SPEED/2
    low_speed=MAX_SPEED/4

    dist_lim_high_low=300
    dist_lim_stop=20

    dist_sum=sum(detect_dist)

    if detect_dist[2]<dist_lim_stop:
        decision[0]=0
        decision[1]=ACCELERATION_FORWARD
    elif dist_sum>dist_lim_high_low:
        if speed<high_speed:
            decision[0]=ACCELERATION_FORWARD
            decision[1]=0
        elif speed>high_speed:
            decision[1]=ACCELERATION_FORWARD
            decision[0]=0
    else:
        if speed<low_speed:
            decision[0]=ACCELERATION_FORWARD
            decision[1]=0
        elif speed>low_speed:
            decision[1]=ACCELERATION_FORWARD
            decision[0]=0
    
    
    
    if detect_dist[0]+detect_dist[1]>detect_dist[3]+detect_dist[4]:
        decision[2]=ANGLE_STEP
        decision[3]=0
    else:
        decision[3]=ANGLE_STEP
        decision[2]=0
    return decision