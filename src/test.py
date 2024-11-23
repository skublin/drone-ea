from train import calc_fitness_score
import glob
import copy
from settings import TRAINING_TARGETS
from main import Game
from utils import GraphModel
import time

# def get_model(model_name):
#     print("GETTING MODEL")
#     input_layer = tf.keras.layers.Input(shape=(6,))
#     dense_layer1 = tf.keras.layers.Dense(9, activation="tanh")
#     dense_layer2 = tf.keras.layers.Dense(9, activation="tanh")
#     output_layer = tf.keras.layers.Dense(4, activation="sigmoid")

#     model = tf.keras.Sequential([input_layer, dense_layer1, dense_layer2, output_layer])
#     if model_name:
#         model.load_weights(model_name)

#     return model


def test_models(start, end, step=1):
    for i in range(start, end, step):
        model = GraphModel(f"models/model-{i}.h5")
        score, time, _ = calc_fitness_score(model)
        print(f"MODEL: {i} Score: {score} Time: {time}")


def display_games(use_pygame, n):
    """
    n - models step, get each n-th model
    """

    models, max_idx = {}, 0
    for model in glob.glob("models/*.h5"):
        idx = int(model.split("-")[-1].replace(".h5", ""))
        max_idx = max(idx, max_idx)

        models[idx] = model

    for i in range(1, max_idx + 1, n):
        print(f"GENERATION: {i}")

        if use_pygame:
            game = Game(model_name=models[i], targets=copy.deepcopy(TRAINING_TARGETS))
            game.run()
        else:
            score, time, simulation = calc_fitness_score(model)
            print(f"MODEL: {i} Score: {score}, Time: {time}")


if __name__ == "__main__":
    # display_games(use_pygame=True, n=10)
    test_models(100, 120)
