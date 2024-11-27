import numpy
import pygad
import pygad.nn
import pygad.gann
import pygad.kerasga

import keras
import pickle
from utils import GraphModel, calc_single_fitness_score, model_build
from settings import TRAINING_TARGETS_LIST
from concurrent.futures import ThreadPoolExecutor
from threading import Lock


def fitness_func(ga_instance, solution, sol_idx):
    global GANN_instance, model, thread_lock

    # make sure only one fitness function is running at a time
    with thread_lock:
        model_weights_matrix = pygad.kerasga.model_weights_as_matrix(
            model=model, weights_vector=solution
        )

        model.set_weights(weights=model_weights_matrix)
        graph_model = GraphModel(model=model)

        score, targets, total_time = 0, 0, 0

        inputs = [(graph_model, targets) for targets in TRAINING_TARGETS_LIST]
        with ThreadPoolExecutor(max_workers=10) as executor:
            for result in executor.map(calc_single_fitness_score, inputs):
                single_score, simulation, time = result

                score += single_score
                targets += simulation.score
                total_time += time

        # average score
        # score /= len(TRAINING_TARGETS_LIST)

        print(
            f"Generation: {ga_instance.generations_completed + 1 + initial_population_idx}, Solution: {sol_idx}, Collected Targets: {targets}, Fitness score: {score}, Time: {total_time}"
        )

    return score


def callback_generation(ga_instance):
    global GANN_instance, last_fitness

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    last_fitness = solution_fitness.copy()

    print(
        f"Generation = {ga_instance.generations_completed+initial_population_idx} BEST FITNESS = {solution_fitness} BEST SOLUTION = {solution_idx}"
    )
    # save the best model for each generation

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(
        model=model, weights_vector=solution
    )
    model.set_weights(weights=model_weights_matrix)
    model.save(
        f"models-2-dim/model-{ga_instance.generations_completed+initial_population_idx}.h5"
    )

    # save population weights to file
    pickle.dump(
        ga_instance.population,
        open(
            f"weights-2-dim/population_{ga_instance.generations_completed+initial_population_idx}-weights.pkl",
            "wb",
        ),
    )


thread_lock = Lock()

# Holds the fitness value of the previous generation.
last_fitness = 0

# Creating the Keras model.
# input_layer = keras.layers.Input(shape=(6,))
# dense_layer1 = keras.layers.Dense(32, activation="relu")
# dense_layer2 = keras.layers.Dense(16, activation="relu")
# output_layer = keras.layers.Dense(2, activation="sigmoid")

# model = keras.Sequential([input_layer, dense_layer1, dense_layer2, output_layer])
model = model_build(6, 2)

# Creating an initial population of neural networks. The return of the initial_population() function holds references to the networks, not their weights. Using such references, the weights of all networks can be fetched.
num_solutions = 100  # A solution or a network can be used interchangeably.

GANN_instance = pygad.kerasga.KerasGA(model=model, num_solutions=num_solutions)

# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.

# initial_population_idx = 230
initial_population_idx = 0

if initial_population_idx > 0:
    population_weights = pickle.load(
        open(f"weights-2-dim/population_{initial_population_idx}-weights.pkl", "rb")
    )

    # reshape weights matrix to match number of solutions
    initial_population = []
    for i in range(0, num_solutions):
        initial_population.append(population_weights[i % len(population_weights[0])])

    initial_population = numpy.array(initial_population)
else:
    initial_population = GANN_instance.population_weights

num_parents_mating = (
    25  # Number of solutions to be selected as parents in the mating pool.
)

num_generations = 5000  # Number of generations.

mutation_percent_genes = 10  # mutate 10% of genes - to preserve diversity
mutation_type = "random"  # mutate randomly

parent_selection_type = (
    "rws"  # tournament selection - select parents by tournament selection
)

crossover_type = "single_point"  # single point crossover - to preserve structure

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    initial_population=initial_population,
    fitness_func=fitness_func,
    mutation_percent_genes=mutation_percent_genes,
    parent_selection_type=parent_selection_type,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    keep_parents=10,  # Keep 25 of parents
    on_generation=callback_generation,
)


if __name__ == "__main__":
    # run on multiple threads
    ga_instance.run()

    # # save the best model
    # solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # model_weights_matrix = pygad.kerasga.model_weights_as_matrix(
    #     model=model, weights_vector=solution
    # )
    # model.set_weights(weights=model_weights_matrix)
    # model.save(f"model-{num_generations}.h5")

    # # save population weights to file
    # pickle.dump(
    #     ga_instance.population,
    #     open(f"weights/population_{num_generations}-weights.pkl", "wb"),
    # )
