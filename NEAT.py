import queue
from typing import List, Tuple
import pygame
import sys
import numpy as np
import math
from common import *
from car import *
import neat
import queue
import visualize
from environment import *
import time

CONFIG = './config-feedforward.txt'
ep_step = 300
generation_step = 1
Training = True
CHECKPOINT = 9

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def eval_genomes(genomes, config):
    env = Environment()
    env.start()
    net_array = {}
    fitness_history = {}
    for genome_id, genome in genomes:
        net_array[genome_id] = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness_history[genome_id] = []
        genome.fitness = 0
    for ep in range(generation_step):
        car_alive = len(genomes)
        env.reset(genomes)
        action_array = {}
        for genome_id, genome in genomes:
            action_array[genome_id] = ([0., 0., 0., 0.])
        speed_list, state_dist_array, reward_array, done_array, info = env.step(action_array, genomes, fitness_history)
        action_array = {}
        while True:
            done_cnt = 0
            for genome_id, genome in genomes:
                if done_array[genome_id]:
                    done_cnt += 1
            if done_cnt is len(genomes):
                break

            for genome_id, genome in genomes:
                if done_array[genome_id]:
                    action_array[genome_id] = [0, 0, 0, 0]
                    continue
                observation = state_dist_array[genome_id]
                action_value = net_array[genome_id].activate(observation)
                action_value.insert(0, 0)
                action_value.insert(0, 0)
                action_value.append(0)
                tmp = sigmoid(action_value[2]) - 0.5
                action_value[2] = (sigmoid(action_value[2]) - 0.8)
                if action_value[2] < 0:
                    action_value[3] = -action_value[2]
                    action_value[2] = 0
                action_array[genome_id] = action_value

            speed_list, state_dist_array, reward_array, done_array, info = env.step(action_array, genomes, fitness_history)
            for genome_id, genome in genomes:
                if done_array[genome_id]:
                    continue
                genome.fitness += reward_array[genome_id]
                fitness_history[genome_id].append(reward_array[genome_id])
                # print(genome.fitness)
    env.close()


def run():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, CONFIG)
    pop = neat.Population(config)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(5))

    winner = pop.run(eval_genomes, 50)
    nodes_name = {-1: "In0",
                  -2: "In1",
                  -3: "In2",
                  -4: "In3",
                  -5: "In4",
                  0: "act1"}
    visualize.draw_net(config, winner, True, node_names=nodes_name)
    visualize.draw_net(config, winner, True, node_names=nodes_name, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

def evaluation():
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % CHECKPOINT)
    winner = p.run(eval_genomes, 1)


    net = neat.nn.FeedForwardNetwork.create(winner, p.config)
    nodes_name = {-1: "In0",
                  -2: "In1",
                  -3: "In2",
                  -4: "In3",
                  -5: "In4",
                  0: "act1"}
    visualize.draw_net(p.config, winner, True, node_names=nodes_name)

    # env = Environment()
    # env.start()
    # fitness_history = {}


if __name__ == '__main__':
    if Training:
        run()
    else:
        evaluation()
