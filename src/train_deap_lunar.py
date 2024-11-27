import tensorflow as tf
import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import demo_heuristic_lander
import Box2D
import random
from deap import base, creator, tools, algorithms
import pickle
import numpy
from utils import GraphModel


# Init the gym environment
env = gym.make("LunarLanderContinuous-v3")
env = env.env
env.reset()
in_dimen = env.observation_space.shape[0]
out_dimen = env.action_space.shape[
    0
]  # Total no. of possible actions. In this case it can take 2 continous values ranging between -1 to +1


def model_weights_as_matrix(model, weights_vector):
    """
    Reshapes the PyGAD 1D solution as a Keras weight matrix.

    Parameters
    ----------
    model : TYPE
        The Keras model.
    weights_vector : TYPE
        The PyGAD solution as a 1D vector.

    Returns
    -------
    weights_matrix : TYPE
        The Keras weights as a matrix.

    """
    weights_matrix = []

    start = 0
    for layer_idx, layer in enumerate(model.layers):  # model.get_weights():
        # for w_matrix in model.get_weights():
        layer_weights = layer.get_weights()
        if layer.trainable:
            for l_weights in layer_weights:
                layer_weights_shape = l_weights.shape
                layer_weights_size = l_weights.size

                layer_weights_vector = weights_vector[
                    start : start + layer_weights_size
                ]
                layer_weights_matrix = numpy.reshape(
                    layer_weights_vector, newshape=(layer_weights_shape)
                )
                weights_matrix.append(layer_weights_matrix)

                start = start + layer_weights_size
        else:
            for l_weights in layer_weights:
                weights_matrix.append(l_weights)

    return weights_matrix


# Function to create model
def model_build(in_dimen=in_dimen, out_dimen=out_dimen):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, input_dim=in_dimen, activation="relu"))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(8, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(out_dimen))

    # Just like before, compilation is not required for the algorithm to run
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model


def graph_model(weights):
    model = model_build()
    model.set_weights(model_weights_as_matrix(model, weights))
    return GraphModel(model=model)


# fitness function
def evaluate(individual, award=0):
    env.reset()
    obs1 = env.reset()[0]
    model = graph_model(individual)
    # model = model_build()
    # model.set_weights(model_weights_as_matrix(model, individual))
    done = False
    step = 0
    while (done == False) and (step <= 1000):
        obs2 = np.expand_dims(obs1, axis=0)
        obs3 = []
        for i in range(in_dimen):
            obs3.append(obs2[0][i])
        obs4 = np.array(obs3).reshape(-1)
        obs = np.expand_dims(obs4, axis=0)
        selected_move1 = model.predict(obs, retries=4)
        _step = env.step(selected_move1[0])
        obs2, reward, terminated, truncated, *_ = _step

        award += reward
        step = step + 1
        obs1 = obs2

        if terminated or truncated:
            done = True

    return (award,)


# Define the genetic algorithm

model = model_build()
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
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("Mean", np.mean)
stats.register("Max", np.max)
stats.register("Min", np.min)


pop = toolbox.population(n=100)  # n = No. of individual in a population
hof = tools.HallOfFame(1)


if __name__ == "__main__":

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.8,
        mutpb=0.2,
        ngen=30,
        halloffame=hof,
        stats=stats,
        verbose=True,
    )

    with open("lunarlander_model.pkl", "wb") as cp_file:
        pickle.dump(hof.items[0], cp_file)

    # env = gym.make("LunarLanderContinuous-v3", render_mode="human")

    # with open("lunarlander_model.pkl", "rb") as cp_file:
    #     weights = pickle.load(cp_file)

    # demo(env, weights, render=True)
