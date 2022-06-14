'''
Description: this file use pygame to generate map in png format
Date: 2022-06-09 17:05:24
LastEditTime: 2022-06-10 16:57:53
'''

import random
import pygame
import math

MIDDLE_POINT=(540,360)
WINDOW_SIZE=(1080,720)
PADDING=10
BOUNDARY_COLOR=(150,41,255)

def is_out(pointx,pointy):
    """
        this function check whether the point is out of the window
    """
    if pointx<=PADDING or pointx>=WINDOW_SIZE[0]-PADDING or pointy<=PADDING or pointy>=WINDOW_SIZE[1]-PADDING:
        return True
    else:
        return False

def generate_random_points(cnt,dist_low,dist_high,last_dist_high=70):
    """
        this function generate random points around the middle point
    """

    angle_gap=2*math.pi/cnt
    points_inner=[]
    points_outer=[]
    last_dist=-1
    for i in range(cnt):
        angle=i*angle_gap
        #generate outer first
        while True:
            while True:
                dist=random.randint(dist_low*3,dist_high*3)
                if last_dist>0:
                    if abs(dist-last_dist)<last_dist_high:
                        break
                else:
                    break
            # generate outer points first
            px=MIDDLE_POINT[0]+dist*math.cos(angle)
            py=MIDDLE_POINT[1]+dist*math.sin(angle)
            print(px,py)
            if not is_out(px,py):
                break
        last_dist=dist
        points_outer.append((px,py))
        dist_inner=random.randint(max(dist_low,dist-dist_high),dist-dist_low)
        # generate inner points
        px=MIDDLE_POINT[0]+dist_inner*math.cos(angle)
        py=MIDDLE_POINT[1]+dist_inner*math.sin(angle)
        points_inner.append((px,py))
    return points_outer,points_inner

def generate_map(seed=None,points_cnt=20):
    
    if seed is not None:
        random.seed(seed)
    pygame.init()
    screen = pygame.display.set_mode((1080, 720), flags=pygame.HIDDEN)
    screen.fill((255, 255, 255))
    # pygame.draw.circle(screen, (0, 0, 0), MIDDLE_POINT, 10)
    point_list_outer,point_list_inner=generate_random_points(points_cnt,25,100)
    #draw all the points
    pygame.display.flip()
    # #draw the outer points
    # for point in point_list_outer:
    #     pygame.draw.circle(screen, BOUNDARY_COLOR, point, 5)
    #draw the lines between the outer points
    for i in range(len(point_list_outer)-1):
        pygame.draw.line(screen, BOUNDARY_COLOR, point_list_outer[i], point_list_outer[i+1], 5)
    pygame.draw.line(screen, BOUNDARY_COLOR, point_list_outer[-1], point_list_outer[0], 5)
    # #draw the inner points
    # for point in point_list_inner:
    #     pygame.draw.circle(screen, BOUNDARY_COLOR, point, 5)
    
    #draw the lines between the inner points
    for i in range(len(point_list_inner)-1):
        pygame.draw.line(screen, BOUNDARY_COLOR, point_list_inner[i], point_list_inner[i+1], 5)
    pygame.draw.line(screen, BOUNDARY_COLOR, point_list_inner[-1], point_list_inner[0], 5)
    
    pygame.display.flip()
    #output the map
    pygame.image.save(screen, '.\\maps\\map.png')
    pygame.quit()

    #save the starting point coordinates
    with open('.\\maps\\map_start.txt','w') as f:
        f.write(str(point_list_outer[0][0])+' '+str(point_list_outer[0][1])+'\n')
        f.write(str(point_list_inner[0][0])+' '+str(point_list_inner[0][1])+'\n')

if __name__ == '__main__':
    generate_map()
