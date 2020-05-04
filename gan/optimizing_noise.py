import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.preproccesing import normalize

from gan.model_path import gen_path, disc_path

generator = tf.keras.models.load_model(gen_path)
discriminator = tf.keras.models.load_model(disc_path)

class DNA:

    def __init__(self, noise):
        self.noise = noise
        self.image = None
        self.fitness_score = None

    def create_image(self):
        if self.image is None:
            self.image = generator(self.noise)
        return self.image

    def show(self):
        plt.imshow(self.create_image()[0, :, :, 0], cmap='gray')

    def fitness(self):
        if self.fitness_score is None:
            self.fitness_score = 1 / (discriminator(self.create_image()) - 1) ** 2
        return self.fitness_score

    def mutate(self):
        self.noise = 0.99 * self.noise + 0.01 * tf.random.normal([1, 100])
        return self

    def copy(self):
        return DNA(self.noise)

    @staticmethod
    def random():
        return DNA(tf.random.normal([1, 100]))

    @staticmethod
    def crossover(dna1, dna2):
        return DNA((dna1.noise + dna2.noise) / 2)


population_size = 200
num_of_generations = 20
breed_percent = 0.3

population = [DNA.random() for _ in range(population_size)]

best = None

for gen in range(num_of_generations):
    cur_time = time.time()
    print("Generation", gen)
    population.sort(key=lambda x:-int(x.fitness()))
    parents = population[:int(population_size * breed_percent)]
    best = parents[0]
    total_fitness = sum([int(parent.fitness()) for parent in parents])
    p = [int(parent.fitness()) / total_fitness for parent in parents]
    def choose():
        return np.random.choice(parents, p=p)
    population = [DNA.crossover(choose(), choose()).mutate() if i < population_size * 0.8 else DNA.random() for i in range(population_size - 1)]
    population.append(best.copy())

print(discriminator(best.create_image()))
best.show()

np.save('./inputs/best_noise_input.npy', best.noise)

plt.show()