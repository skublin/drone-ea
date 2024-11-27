import tensorflow as tf
import numpy as np
from deap import base, creator, tools, algorithms
from deap.algorithms import varAnd
import pickle
import numpy
from utils import GraphModel
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from settings import TRAINING_TARGETS_LIST
from utils import calc_single_fitness_score, model_build, model_weights_as_matrix


in_dimen = 6
out_dimen = 2
thread_lock = Lock()


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        with open(f"models-2-dim-deap/model-{gen}.pkl", "wb") as cp_file:
            pickle.dump(hof.items[0], cp_file)

    return population, logbook


def graph_model(weights):
    model = model_build(in_dimen, out_dimen)
    model.set_weights(model_weights_as_matrix(model, weights))
    return GraphModel(model=model)


# fitness function
def evaluate(individual, award=0):

    # make sure only one fitness function is running at a time
    with thread_lock:
        _graph_model = graph_model(weights=individual)

        targets, total_time = 0, 0

        inputs = [(_graph_model, targets) for targets in TRAINING_TARGETS_LIST]
        with ThreadPoolExecutor(max_workers=10) as executor:
            for result in executor.map(calc_single_fitness_score, inputs):
                single_score, simulation, time = result

                award += single_score
                targets += simulation.score
                total_time += time

        print(
            f"Collected Targets: {targets}, Fitness score: {award}, Time: {total_time}"
        )

    return (award,)


# Define the genetic algorithm

model = model_build(in_dimen, out_dimen)
ind_size = model.count_params()


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register(
    "weight_bin", np.random.uniform, -1, 1
)  # Initiate weights from uniform distribution between -1 to +1
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.weight_bin, n=ind_size
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("Mean", np.mean)
stats.register("Max", np.max)
stats.register("Min", np.min)


pop = toolbox.population(n=100)  # n = No. of individual in a population

hof = tools.HallOfFame(1)


if __name__ == "__main__":

    pop, log = eaSimple(
        pop,
        toolbox,
        cxpb=0.8,
        mutpb=0.2,
        ngen=500,
        halloffame=hof,
        stats=stats,
        verbose=True,
    )

    with open("model.pkl", "wb") as cp_file:
        pickle.dump(hof.items[0], cp_file)

    # env = gym.make("LunarLanderContinuous-v3", render_mode="human")

    # with open("lunarlander_model.pkl", "rb") as cp_file:
    #     weights = pickle.load(cp_file)

    # demo(env, weights, render=True)
