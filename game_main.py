'''
Description: 
Date: 2022-06-08 21:21:14
LastEditTime: 2022-06-11 15:22:35
'''
"""
Description: this program use the pygame to implement a car game
Date: 2022-06-08 21:21:14
LastEditTime: 2022-06-09 23:43:50
"""

from typing import List, Tuple
import pygame
import sys
import numpy as np
import math
from common import *
from car import *


class CarGame:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.car_image = pygame.image.load('car.png').convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image, (CAR_WIDTH, CAR_HEIGHT))
        # load the map
        self.background_img, self.start_point = load_map('./maps/', 'map')
        self.car_list = []
        self.car_list.append(Car(self.car_image, self.start_point, np.pi / 2, True))
        self.car_list[0].bind_auto_func(simple_auto)
        self.car_list.append(Car(self.car_image, self.start_point,np.pi/3,True))
        self.car_list[1].bind_auto_func(simple_auto)
        self.car_list.append(Car(self.car_image, self.start_point,np.pi*3/4,True))
        self.car_list[2].bind_auto_func(simple_auto)
        self.car_list.append(Car(self.car_image, self.start_point,np.pi/5,True))
        self.car_list[3].bind_auto_func(simple_auto)
        # self.car_list.append(Car(self.car_image, self.start_point,np.pi*3/2,True))
        # self.car_list[4].bind_auto_func(simple_auto)
        # self.car_list.append(Car(self.car_image, self.start_point,np.pi,True))
        # self.car_list[5].bind_auto_func(simple_auto)
        self.running = True

    def run(self):
        while self.running:
            self.clock.tick(GAME_FPS)
            self.handle_events()
            self.update()
            self.draw()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            # detect the mouse movement
            if event.type == pygame.MOUSEBUTTONDOWN:
                print(self.screen.get_at(event.pos))

    def update(self):
        # self.car.update(self.screen)
        for car in self.car_list:
            car.update(self.screen)
            # print(car.last_angle)

    def draw(self):
        # draw the background image
        # self.screen.fill((0, 0, 0))
        self.screen.blit(self.background_img, (0, 0))
        for car in self.car_list:
            car.detect_track_boundary(
                self.screen)  # for the function is using the pixel value to detect the track boundary, so it should be done before the car.draw()
            car.calculate_distance(self.screen)  # for the same reson, it should be done before the car.draw()
            car.draw(self.screen)
        # draw a circle
        # pygame.draw.circle(self.screen, (255, 0, 0), (int(510), int(360)), 10)
        pygame.display.flip()


if __name__ == '__main__':
    game = CarGame()
    game.run()
