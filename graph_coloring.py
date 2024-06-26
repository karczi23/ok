from itertools import count, filterfalse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import random
import time
from statistics import fmean

class GraphColoring:
    def __init__(self, graph_data, size=100, mutation_rate=0.05, crossover_rate=0.5, time_to_run=300) -> None:
        self.length = graph_data[0]
        self.graph = np.array(graph_data[1], dtype=int)
        self.graph_colors = np.zeros((self.length, self.length))
        self.size = size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.time_to_run = time.time() + time_to_run
        self.time_to_run_greedy = []

        self.colors_matrix = np.zeros(self.length)

    def get_smallest_color(self, num):
        colors = self.graph_colors[num]
        color = 1
        while color in colors:
            color += 1
        return color

    def greedy(self, permutation):
        colors_used = 0
        start = time.time()
        # permutation = list(random.permutation(number))
        # permutation = [i for i in range(number)]
        # number = len(permutation)
        self.graph_colors = eye_matrix(self.length)
        for num in permutation:
            # print(num)
            # print(graph[num])
            # print(graph_colors[num])
            # print()
            color = next(filterfalse(set(self.graph_colors[num]).__contains__, count(1)))
            if color > colors_used:
                colors_used = color
            for line in range(self.length):
                self.graph_colors[line][num] = color if self.graph[line][num] == 1 else 0
        # print(colors_used)
                
        # print (time.time() - start)
        self.time_to_run_greedy.append(time.time() - start)
        return colors_used, permutation
# read_graph()
    def generate_individual(self):
        permutation = np.random.permutation(self.length)
        return self.greedy(permutation)

    def generate_population(self):
        population = np.empty(self.size, dtype=object)
        for i in range(self.size):
            population[i] = self.generate_individual()
        return population
    
    def mutate(self, individual):
        # print(individual)
        # print(len(individual))
        # print(self.length)
        permutation = individual.copy()
        for node in range(self.length):
            if random.random() < self.mutation_rate:
                position_to_swap = random.randrange(0, self.length)
                permutation[node], permutation[position_to_swap] = permutation[position_to_swap], permutation[node]
        return self.greedy(permutation)
    
    def crossover(self, parent1, parent2):
    # def crossover(self):
        # print(("start", parent1, parent2))
        permutation = np.zeros(self.length, dtype=np.uint16)
        # print(parent2)
        # parent1 = list(np.permutation(23))
        # parent2 = np.permutation(23)
        # print(parent1)
        position_to_swap = random.randrange(0, self.length)
        np_parent1 = np.array(parent1[1][:position_to_swap])
        np_parent2 = np.array(parent2[1])
        # np_parent1 = np.zeros(position_to_swap)
        # np_parent2 = np.zeros(self.length - position_to_swap)
        for i in range(position_to_swap):
            permutation[i] = parent1[1][i]
        # permutation = np.array(parent1[1][:position_to_swap])
        np_parent2 = np_parent2[~np.isin(parent2[1], np_parent1)]
        # print((parent2, np_parent2))
        # for i in range(self.length - position_to_swap):
        #     np_parent2[i] = parent2[i]
        # permutation = parent1[1].copy()[:position_to_swap]
        # parent2 = list(setdiff1d(parent2.copy(), permutation))
        # print(parent2)
        i = 0
        # print((position_to_swap,len(np_parent2), self.length, parent1,  permutation, np_parent2))
        for k in range(position_to_swap, self.length):
            permutation[k] = np_parent2[i]
            i += 1
        # permutation.extend(list(parent2))
        # print(permutation)
        # print((parent1[1], parent2, permutation))
        # for node in range(self.length):
        #     if random.random() < self.crossover_rate:
        #         permutation.append(parent1[1][node])
        #     else:
        #         permutation.append(parent2[1][node])
        return permutation
    
    def evolve(self, population, executor):
        # new_population = []
        fitness_scores = [(individual[0], individual[1]) for individual in population]
        fitness_scores.sort(key=lambda x : x[0])
        # for fit in fitness_scores:
        #     print(fit[0])

        self.parents_half = fitness_scores[:(self.length // 2)]

        parents_list = []
        for _ in range(int(self.size * 0.97)):
            # parents = random.sample(parents_half, 2)
            p1 = random.randint(0, len(self.parents_half) - 1)
            p2 = random.randint(0, len(self.parents_half) - 1)
            parents_list.append((p1, p2))


        new_population = self.run_in_parallel(parents_list, executor)

        old_pop = self.size - len(parents_list)
        for i in range(old_pop):
            new_population[self.size - 1 - i] = fitness_scores[i]
        # print(parents_list)
        # print(len(self.parents_half))
        return new_population
    
    def changes(self, tupla):
            (p1, p2) = tupla
            # print(p1, p2)
            child = self.crossover(self.parents_half[p1], self.parents_half[p2])
            child = self.mutate(child)

            return child
            # return min([self.parents_half[p1], self.parents_half[p2], child], key=lambda x : x[0])
    
    def run_in_parallel(self, parents, executor):
        # print(len(parents))
        new_pop = executor.map(self.changes, parents)
        print("po popie")
        # i = 0
        # for _ in new_pop:
        #     i += 1
        # print(i)
        arr = np.zeros(self.size, dtype=object)
        for i, item in enumerate(new_pop):
            arr[i] = item
        return arr
        # return np.fromiter(new_pop, dtype=object)
    
    def run(self):
        population = self.generate_population()
        # print(population)
        g = 0
        best_individual, _ = min([(individual[0], individual[1]) for individual in population],
                                            key=lambda x: x[0])
        print(best_individual)
        with ProcessPoolExecutor() as executor:
            while time.time() < self.time_to_run:
                # print("g")
                g += 1
                population = self.evolve(population, executor)
                # print(population)
                best_individual, _ = min([(individual[0], individual[1]) for individual in population],
                                                key=lambda x: x[0])
                data = []
                for individual in population:
                    data.append(individual[0])
                
                print("best: " + str(min(data)))
                print(sorted(data))
                # print([(individual[0], _) for individual in population])

        best_individual, _ = min([(individual[0], individual[1]) for individual in population],
                                            key=lambda x: x[0])
        

        print(min(self.time_to_run_greedy))
        print(max(self.time_to_run_greedy))
        print(fmean(self.time_to_run_greedy))
        print(best_individual)
        print(g)

def read_graph():
    with open("data/gc1000.txt", "r") as f:
        data = f.readlines()
        length = int(data[0])
        # greedy(length)
        graph = eye_matrix(length)
        for item in data[1:]:
            itemSep = item.split(' ')
            e1 = int(itemSep[0]) - 1
            e2 = int(itemSep[1]) - 1
            graph[e1][e2] = 1
            graph[e2][e1] = 1
        return length, graph
        # return greedy(length)

def eye_matrix(number):
    return np.zeros((number, number))

alg = GraphColoring(read_graph())
# colors, _ = alg.greedy([i for i in range(36)])
# print(colors)
# alg.crossover()
if __name__ == "__main__":
    alg.run()

# best = 1000gg4444
# for i in range(100):
#     num = read_graph()
#     graph = []
#     graph_colors = []
#     if num < best:
#         best = num
    
#     print(best)