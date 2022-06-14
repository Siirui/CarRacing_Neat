from pandas import date_range
import pygame
from common import *
import math
import numpy as np

car_speed = 2

class Car:
    def __init__(self,image,pos,angle,is_auto=False,midx=510,midy=360) -> None:
        self.image = image
        self.image_clean = image
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.speed = car_speed  # 固定速度为1
        self.speed_vec = (0,0)
        self.angle = angle
        self.x=pos[0]
        self.y=pos[1]
        self.is_alive=True
        # create a list of five angles
        self.detect_line_x=np.zeros(CNT_DETECT_LINE)
        self.detect_line_y=np.zeros(CNT_DETECT_LINE)
        self.detect_dist=np.zeros(CNT_DETECT_LINE)

        self.update_auto_func=None
        self.is_auto=is_auto
        
        #get the mid point of the map
        self.midx=midx
        self.midy=midy
        self.last_angle=self.calculate_angle()
        self.score=0
        self.accumulate_angle=0 # the true angle of the car that can accumulate each lap

    def reset(self,pos,angle,is_auto=False,midx=510,midy=360):
        self.is_alive=True
        self.x=pos[0]
        self.y=pos[1]
        self.angle=angle
        self.speed = car_speed
        self.speed_vec=(0,0)
        self.is_auto=is_auto
        self.midx=midx
        self.midy=midy
        self.last_angle=self.calculate_angle()
        self.score=0
        self.accumulate_angle=0


    def process_decision(self,forward,backward,left,right):
        self.speed += forward - backward
        self.angle += right - left
        if self.speed < 0:
            self.speed = 0
        if self.speed > MAX_SPEED:
            self.speed = MAX_SPEED

    def bind_auto_func(self,func):
        self.update_auto_func = func

    def update_auto(self,screen):
        if self.is_alive == False:
            return
        self.calculate_distance(screen)
        result=self.update_auto_func(self.speed,self.detect_dist)
        self.process_decision(*result)
        self.update_pos()
        
        

    def update_manual(self):
        if self.is_alive == False:
            return
        decision=[0,0,0,0]
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            decision[2]=ANGLE_STEP
        if keys[pygame.K_RIGHT]:
            decision[3]=ANGLE_STEP
        if keys[pygame.K_UP]:
            decision[0]=ACCELERATION_FORWARD
        if keys[pygame.K_DOWN]:
            decision[1]=ACCELERATION_FORWARD
        self.process_decision(*decision)
        self.update_pos()
    
    def update_by_step(self,screen,action:List[float]):
        """update by step

        Args:
            screen (pygame surface): _description_
            action (List[float]): _description_
        
        Returns:
            state
            reward
            cur_angle
        """
        if not self.is_alive:
            return None,0,True,"the car is already done"
        self.process_decision(*action)
        self.update_pos()
        d_angle=self.update_angle()
        self.calculate_distance(screen)
        return self.detect_dist,d_angle,self.accumulate_angle

    def update_pos(self):
        self.x+=self.speed*np.cos(self.angle)
        self.y+=self.speed*np.sin(self.angle)
        self.rect.center = (self.x,self.y)
        #rotate the image
        self.image,self.rect = rot_center(self.image_clean,self.rect,-self.angle*180/np.pi-90)
        
    
    def detect_track_boundary(self,screen):
        """
        this function use the pixel color to detect the track boundary
        """
        # if screen.get_at((int(self.x), int(self.y))) == (0, 0, 0, 255):
        try:
            pixel = screen.get_at((int(self.x), int(self.y)))
            if is_boundary_color(pixel):
                self.is_alive = False
        except:
            return

    def calculate_distance(self,screen):
        # calculate car's five angles' distances form the track boundary
        if not self.is_alive:
            return
        detect_angle=[self.angle-math.pi/2,self.angle-math.pi/4,self.angle,self.angle+math.pi/4,self.angle+math.pi/2]
        for i in range(CNT_DETECT_LINE):
            for diff in range(1000):
                cur_x=self.x+diff*np.cos(detect_angle[i])
                cur_y=self.y+diff*np.sin(detect_angle[i])
                if cur_x<0 or cur_x>=WINDOW_WIDTH or cur_y<0 or cur_y>=WINDOW_HEIGHT:
                    break
                # print(cur_x,cur_y)
                pixel = screen.get_at((int(cur_x), int(cur_y)))
                if is_boundary_color(pixel):
                    self.detect_line_x[i]=cur_x
                    self.detect_line_y[i]=cur_y
                    self.detect_dist[i]=diff
                    break
    
    def calculate_angle(self):
            #calculate the angle between the car and the mouse
            return calculate_angle(self.x,self.y,self.midx,self.midy)
    def update(self,screen):
        if self.is_auto:
            self.update_auto(screen)
        else:
            self.update_manual()
        #update the score
        self.update_score()

    def draw_dectect_line(self,screen):
        for i in range(CNT_DETECT_LINE):
            pygame.draw.line(screen,(255,0,0),(self.x,self.y),(self.detect_line_x[i],self.detect_line_y[i]),1)    


    def draw(self,screen):
        if self.is_alive:
            self.draw_dectect_line(screen)
            screen.blit(self.image,self.rect)

    def update_angle(self):
        d_angle=0
        # -pi to pi
        cur_angle=self.calculate_angle()
        if cur_angle<-3 and self.last_angle>3:
            d_angle=-(abs(cur_angle-self.last_angle)-2*math.pi)
        elif cur_angle>3 and self.last_angle<-3:
            d_angle=(abs(cur_angle-self.last_angle)-2*math.pi)
        else:
            d_angle=(cur_angle-self.last_angle)
        self.last_angle=cur_angle
        self.accumulate_angle+=d_angle
        print(self.accumulate_angle)
        return d_angle

    def update_score(self):
        # print(self.calculate_angle())
        # TODO: fixed the reverse problem
        # self.score+=abs(abs(self.last_angle)-abs(self.calculate_angle()))
        
        # print(self.last_angle,cur_angle,d_angle)
        d_angle=self.update_angle()
        self.score+=d_angle
        # print(self.last_angle)
        # print(self.score)

    def get_score(self):
        return self.score
