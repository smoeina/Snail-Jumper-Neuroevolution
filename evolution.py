import copy
import random

from player import Player
import numpy as np
import json

class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        fitness_results = {
                'min_fitness': [],
                'max_fitness': [],
                'mean_fitness': []
            }
        with open('fitness_results.json', 'w') as out_file:
                json.dump(fitness_results, out_file)
    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = player.nn
        new_player.fitness = player.fitness
        return new_player

    #By this function we calculate
    def calculate_cumulative_probabilities(self, players):
        total_fitness = 0
        for player in players:
            total_fitness += player.fitness
        probabilities = []
        for player in players:
            probabilities.append(player.fitness / total_fitness)
        # convert it to
        for i in range(1, len(players)):
            probabilities[i] += probabilities[i - 1]
        return probabilities
    def roulette_wheel(self, players, parent_numbers):
        probabilities = self.calculate_cumulative_probabilities(players)

        results = []
        for random_number in np.random.uniform(low=0, high=1, size=parent_numbers):
            for i, probability in enumerate(probabilities):
                if random_number <= probability:
                    results.append(self.clone_player(players[i]))
                    break

        return results

    def save_fitness_result(self, min_fitness, max_fitness, mean_fitness):
        fitness_results = {
                'min_fitness': [],
                'max_fitness': [],
                'mean_fitness': []
            }
        with open('fitness_results.json', 'r') as in_file:
                fitness_results = json.load(in_file)

        fitness_results['min_fitness'].append(min_fitness)
        fitness_results['max_fitness'].append(max_fitness)
        fitness_results['mean_fitness'].append(mean_fitness)

        with open('fitness_results.json', 'w') as out_file:
                json.dump(fitness_results, out_file)
    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        result = players
        selection_method = "SUS"

        if selection_method == "top-k":
            sorted_players = sorted(players, key=lambda player: player.fitness, reverse=True)
            result = sorted_players[: num_players]
        elif selection_method == 'roulette wheel':
            result = self.roulette_wheel(players, num_players)
        elif selection_method == "SUS":
            result = self.sus(players, num_players)

        fitness_list = [player.fitness for player in players]
        max_fitness = float(np.max(fitness_list))
        mean_fitness = float(np.mean(fitness_list))
        min_fitness = float(np.min(fitness_list))
        self.save_fitness_result(min_fitness, max_fitness, mean_fitness)
        return result
    def sus(self, players, num_players):
        # Create Intervals
        interval_length = 1 - 1 / num_players
        intervals = np.linspace(0, interval_length, num_players)
        random_number = np.random.uniform(0, 1 / num_players, 1)
        intervals += random_number

        probabilities = self.calculate_cumulative_probabilities(players)

        result = []
        for interval in intervals:
            for i, probability in enumerate(probabilities):
                if interval < probability:
                    result.append(self.clone_player(players[i]))
                    break

        return result
    def mutate_child(self, child):
        number_of_layer = random.randint(0, len(child.nn.weights)-1)
        number_of_weight = random.randint(0, len(child.nn.weights[number_of_layer])-1)
        child.nn.weights[number_of_layer][number_of_weight] = random.random()
        return child

    def q_tournament(self, players, q):
        q_selected = np.random.choice(players, q)
        return max(q_selected, key=lambda player: player.fitness)

    def crossover(self, child1_array, child2_array, parent1_array, parent2_array):
        row_size, column_size = child1_array.shape
        section_1, section_2, section_3 = int(row_size / 3), int(2 * row_size / 3), row_size

        random_number = np.random.uniform(0, 1, 1)
        if random_number > 0.5:
            child1_array[:section_1, :] = parent1_array[:section_1:, :]
            child1_array[section_1:section_2, :] = parent2_array[section_1:section_2, :]
            child1_array[section_2:, :] = parent1_array[section_2:, :]

            child2_array[:section_1, :] = parent2_array[:section_1:, :]
            child2_array[section_1:section_2, :] = parent1_array[section_1:section_2, :]
            child2_array[section_2:, :] = parent2_array[section_2:, :]
        else:
            child1_array[:section_1, :] = parent2_array[:section_1:, :]
            child1_array[section_1:section_2, :] = parent1_array[section_1:section_2, :]
            child1_array[section_2:, :] = parent2_array[section_2:, :]

            child2_array[:section_1, :] = parent1_array[:section_1:, :]
            child2_array[section_1:section_2, :] = parent2_array[section_1:section_2, :]
            child2_array[section_2:, :] = parent1_array[section_2:, :]

    def add_gaussian_noise(self, array, threshold):
        random_number = np.random.uniform(0, 1, 1)
        if random_number < threshold:
            array += np.random.normal(size=array.shape)
    def mutate(self, child):
        # child: an object of class `Player`
        threshold = 0.2
        self.add_gaussian_noise(child.nn.W1, threshold)
        self.add_gaussian_noise(child.nn.W2, threshold)
        self.add_gaussian_noise(child.nn.b1, threshold)
        self.add_gaussian_noise(child.nn.b2, threshold)
    def reproduction(self, parent1, parent2):
        child1 = Player(self.game_mode)
        child2 = Player(self.game_mode)
        self.crossover(child1.nn.W1, child2.nn.W1, parent1.nn.W1, parent2.nn.W1)
        self.crossover(child1.nn.W2, child2.nn.W2, parent1.nn.W2, parent2.nn.W2)
        self.crossover(child1.nn.b1, child2.nn.b1, parent1.nn.b1, parent2.nn.b1)
        self.crossover(child1.nn.b2, child2.nn.b2, parent1.nn.b2, parent2.nn.b2)

        self.mutate(child1)
        self.mutate(child2)
        return child1, child2

    def generate_new_population(self, num_players, prev_players=None):

        if prev_players is None:
            return [Player(self.game_mode) for _ in range(num_players)]

        else:
            method = "Q tournament"
            children = []
            parents = []

            if method == 'roulette wheel':
                parents = self.roulette_wheel(prev_players, num_players)
            elif method == "Q tournament":
                for _ in range(num_players):
                    parents.append(self.q_tournament(prev_players, q=3))

            for i in range(0, len(parents), 2):
                child1, child2 = self.reproduction(parents[i], parents[i + 1])
                children.append(child1)
                children.append(child2)
            return children
