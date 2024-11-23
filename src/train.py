import time
import copy
import numpy
import pygad
import pygad.nn
import pygad.gann
import pygad.kerasga

import keras
import pickle
from main import Simulation
from utils import GraphModel
from settings import Settings, TRAINING_TARGETS


def calc_fitness_score(graph_model):
    simulation = Simulation(
        use_pygame=False, settings=Settings(), targets=copy.deepcopy(TRAINING_TARGETS)
    )

    # initial distance to target
    min_distance = starting_distance = simulation.calculate_target_distance()

    max_time, max_target_score = 30, 10

    # max score that can be achieved
    max_score = max_target_score * 3  # for each target at most 3 points

    score = 0
    score_travel = 0
    current_target_iterations = 1

    for frame_num in range(1, max_time * simulation.FPS + 1):

        prev_score = simulation.score

        # increase score_travel only if drone is not flipped
        if not simulation.drone.is_flipped:

            # compute travel score, add reward if distance to target is decreasing
            # add penalty if distance to target is increasing
            if simulation.calculate_target_distance() <= min_distance:
                # update min distance
                min_distance = simulation.calculate_target_distance()

                # add reward from range 0.5 to 1
                score_travel += 1 - (
                    simulation.calculate_target_distance() / (min_distance * 2)
                )
            else:
                # add penalty from range 0 to 0.25
                score_travel += min_distance / (
                    simulation.calculate_target_distance() * 4
                )
        else:
            # add penalty for flipping the drone
            score_travel -= score_travel / current_target_iterations

        # get predictions from the model
        predictions = graph_model.predict(numpy.array([simulation.nn_input]))
        keys = [1 if p >= 0.5 else 0 for p in predictions[0]]
        simulation.next(keys)

        # player out of bounds
        if not simulation.running:
            # add a penalty for losing

            # reduce score by 15%
            score *= 0.85
            # reduce travel score by 90%
            score += (score_travel / current_target_iterations) * 0.1
            score_travel = 0
            current_target_iterations = 1

            break

        # player reached the target
        if simulation.score > prev_score:
            # calculate final travel score
            score += score_travel / current_target_iterations
            score_travel = 0

            # add a reward for winning
            seconds = current_target_iterations // simulation.FPS
            target_score = (
                starting_distance - seconds * (starting_distance / simulation.FPS)
            ) / starting_distance
            score += target_score

            # update starting distance and reset current target iterations
            current_target_iterations = 1
            starting_distance = min_distance = simulation.calculate_target_distance()

            # stop if reached max score
            if simulation.score >= max_target_score:
                break

            continue

        current_target_iterations += 1

    # if player not lost, calculate only travel score
    if simulation.running and current_target_iterations > 1:
        # calculate only travel score
        score += score_travel / current_target_iterations

    # if player not lost and still has time
    if simulation.running and frame_num < max_time * simulation.FPS:
        # add reward from 0 to 10 based on remaining time
        score += (
            (max_time * simulation.FPS - frame_num) / (max_time * simulation.FPS)
        ) * 10

    # normalize score (from 0 to 1)
    score = max(score, 0) / max_score
    return score, frame_num / simulation.FPS, simulation


def fitness_func(ga_instance, solution, sol_idx):
    global GANN_instance, model

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(
        model=model, weights_vector=solution
    )

    model.set_weights(weights=model_weights_matrix)

    # 4 retries
    for i in range(5):
        try:
            graph_model = GraphModel(model=model)
            score, time, simulation = calc_fitness_score(graph_model)
            break
        except Exception as e:
            if i == 4:
                raise e
            else:
                print(e)

    print(
        f"Generation: {ga_instance.generations_completed + 1 + initial_population_idx}, Solution: {sol_idx}, Collected Targets: {simulation.score}, Fitness score: {score}, Time: {time}"
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
        f"models/model-{ga_instance.generations_completed+initial_population_idx}.h5"
    )

    # save population weights to file
    pickle.dump(
        ga_instance.population,
        open(
            f"weights/population_{ga_instance.generations_completed+initial_population_idx}-weights.pkl",
            "wb",
        ),
    )


# Holds the fitness value of the previous generation.
last_fitness = 0

# Creating the Keras model.
input_layer = keras.layers.Input(shape=(6,))
dense_layer1 = keras.layers.Dense(9, activation="tanh")
dense_layer2 = keras.layers.Dense(9, activation="tanh")
output_layer = keras.layers.Dense(4, activation="sigmoid")

model = keras.Sequential([input_layer, dense_layer1, dense_layer2, output_layer])

# Creating an initial population of neural networks. The return of the initial_population() function holds references to the networks, not their weights. Using such references, the weights of all networks can be fetched.
num_solutions = 50  # A solution or a network can be used interchangeably.

GANN_instance = pygad.kerasga.KerasGA(model=model, num_solutions=num_solutions)

# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.

initial_population_idx = 100
# initial_population_idx = 0

if initial_population_idx is not None:
    population_weights = pickle.load(
        open(f"weights/population_{initial_population_idx}-weights.pkl", "rb")
    )

    # reshape weights matrix to match number of solutions
    initial_population = []
    for i in range(0, num_solutions):
        initial_population.append(population_weights[i % 20])

    initial_population = numpy.array(initial_population)
else:
    initial_population = GANN_instance.population_weights

num_parents_mating = (
    15  # Number of solutions to be selected as parents in the mating pool.
)

num_generations = 5000  # Number of generations.

mutation_percent_genes = 5  # mutate 5% of genes - to preserve diversity
mutation_type = "random"  # mutate randomly

parent_selection_type = "tournament"  # Tournament selection - Selects the best solution by a tournament competition.

crossover_type = "single_point"  # Single-point crossover - A single point is selected on the parent solutions where the genes are swapped between the parents to produce children.

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    initial_population=initial_population,
    fitness_func=fitness_func,
    mutation_percent_genes=mutation_percent_genes,
    parent_selection_type=parent_selection_type,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    keep_parents=5,  # Keep 5 of parents
    keep_elitism=3,  # Keep 3 best solutions
    suppress_warnings=True,
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
