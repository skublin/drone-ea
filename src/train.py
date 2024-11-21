import copy
import numpy
import pygad
import pygad.nn
import pygad.gann
import pygad.kerasga
import keras
import pickle
from main import Simulation
from settings import Settings, TRAINING_TARGETS


def fitness_func(ga_instance, solution, sol_idx):
    global GANN_instance, model

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(
        model=model, weights_vector=solution
    )

    model.set_weights(weights=model_weights_matrix)

    simulation = Simulation(
        use_pygame=False, settings=Settings(), targets=copy.deepcopy(TRAINING_TARGETS)
    )

    # initial distance to target
    starting_distance = simulation.calculate_target_distance()

    max_time, max_target_score = 30, 10

    # max score that can be achieved
    max_score = max_target_score * 3  # for each target at most 3 points

    score = 0
    score_travel = 0
    current_target_iterations = 1

    for frame_num in range(1, max_time * simulation.FPS + 1):

        prev_score = simulation.score

        # compute travel score
        if simulation.calculate_target_distance() <= starting_distance:
            # add reward from range 0.5 to 1
            score_travel += 1 - (
                simulation.calculate_target_distance() / (starting_distance * 2)
            )
        else:
            # add penalty from range 0 to 0.25
            score_travel += starting_distance / (
                simulation.calculate_target_distance() * 4
            )

        predictions = model.predict(numpy.array([simulation.nn_input]), verbose=0)
        keys = [1 if p >= 0.5 else 0 for p in predictions[0]]
        simulation.next(keys)

        # player out of bounds
        if not simulation.running:
            # calculate final travel score

            # reduce score by 15%
            score *= 0.85
            # reduce travel score by 95%
            score += (score_travel / current_target_iterations) * 0.05
            score_travel = 0
            current_target_iterations = 1

            # add a penalty for losing
            break

        # player reached the target
        if simulation.score > prev_score:
            # calculate final travel score
            score += score_travel / current_target_iterations
            score_travel = 0

            # add a reward for winning
            seconds = current_target_iterations // simulation.FPS
            score += (
                starting_distance - seconds * (starting_distance / simulation.FPS)
            ) / starting_distance

            # update starting distance and reset current target iterations
            current_target_iterations = 1
            starting_distance = simulation.calculate_target_distance()

            # stop if reached max score
            if simulation.score >= max_target_score:
                break

            continue

        current_target_iterations += 1

    # if player not lost
    if simulation.running and current_target_iterations > 0:
        # calculate only travel score
        score += score_travel / current_target_iterations

    # if player not lost and still has time
    if simulation.running and frame_num < max_time * simulation.FPS:
        # add reward from 0 to 10 based on remaining time
        score += (
            (max_time * simulation.FPS - frame_num) / (max_time * simulation.FPS)
        ) * 10

    # normalize score
    score = max(score, 0) / max_score
    print(
        f"Generation: {ga_instance.generations_completed + 1}, Solution: {sol_idx}, Collected Targets: {simulation.score}, Fitness score: {score}"
    )

    return score


def callback_generation(ga_instance):
    global GANN_instance, last_fitness

    last_fitness = ga_instance.best_solution()[1].copy()
    print(
        "Generation = {generation}".format(generation=ga_instance.generations_completed)
    )
    # save the best model for each generation

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(
        model=model, weights_vector=solution
    )
    model.set_weights(weights=model_weights_matrix)
    model.save(f"models/model-{ga_instance.generations_completed}.h5")

    # save population weights to file
    pickle.dump(
        ga_instance.population,
        open(
            f"weights/population_{ga_instance.generations_completed}-weights.pkl",
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
num_solutions = 20  # A solution or a network can be used interchangeably.

GANN_instance = pygad.kerasga.KerasGA(model=model, num_solutions=num_solutions)

# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
# population_weights_file = "weights/population_1000-weights.pkl"
population_weights_file = None

if population_weights_file is not None:
    population_weights = pickle.load(open(population_weights_file, "rb"))
    initial_population = population_weights
else:
    initial_population = GANN_instance.population_weights

num_parents_mating = (
    10  # Number of solutions to be selected as parents in the mating pool.
)

num_generations = 1000  # Number of generations.

mutation_percent_genes = [
    5,
    10,
]  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.

parent_selection_type = "sss"  # Type of parent selection.

crossover_type = "single_point"  # Type of the crossover operator.

mutation_type = "adaptive"  # Type of the mutation operator.

keep_parents = 1  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

init_range_low = -2
init_range_high = 5

ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    initial_population=initial_population,
    fitness_func=fitness_func,
    mutation_percent_genes=mutation_percent_genes,
    init_range_low=init_range_low,
    init_range_high=init_range_high,
    parent_selection_type=parent_selection_type,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    keep_parents=keep_parents,
    suppress_warnings=True,
    on_generation=callback_generation,
    parallel_processing=["thread", 10],
)


# run on multiple threads
ga_instance.run()

# save the best model
solution, solution_fitness, solution_idx = ga_instance.best_solution()
model_weights_matrix = pygad.kerasga.model_weights_as_matrix(
    model=model, weights_vector=solution
)
model.set_weights(weights=model_weights_matrix)
model.save(f"model-{num_generations}.h5")


# save population weights to file
pickle.dump(
    ga_instance.population,
    open(f"weights/population_{num_generations}-weights.pkl", "wb"),
)
