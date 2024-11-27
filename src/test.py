import glob
import copy
from settings import TRAINING_TARGETS_LIST, TRAINING_TARGETS
from main import Game, Settings
from simulation import Simulation
from utils import GraphModel, calc_fitness_score, calc_single_fitness_score
import random

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


def generate_training_targets(samples, targets_per_sample):
    simulation = Simulation(use_pygame=False, settings=Settings())

    def euclidean_distance(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    training_targets_list = []
    all_targets = []

    # to make equal distribution of targets in each quarter
    def in_quarter(x, y, q):
        if q == 1:
            return x < simulation.BOARD_WIDTH / 2 and y < simulation.BOARD_HEIGHT / 2
        elif q == 2:
            return x > simulation.BOARD_WIDTH / 2 and y < simulation.BOARD_HEIGHT / 2
        elif q == 3:
            return x < simulation.BOARD_WIDTH / 2 and y > simulation.BOARD_HEIGHT / 2
        elif q == 4:
            return x > simulation.BOARD_WIDTH / 2 and y > simulation.BOARD_HEIGHT / 2

    # generate first target for each sample (to make sure even distribution)
    first_targets = []
    for s in range(samples):
        _target = simulation._generate_target()

        while (
            # Make sure far enough from any other first targets
            any(
                euclidean_distance(_target.x, _target.y, target[0], target[1]) < 200
                for target in first_targets
            )
            # make sure in proper quarter (to make sure even distribution)
            or not in_quarter(_target.x, _target.y, s % 4 + 1)
        ):
            _target = simulation._generate_target()

        first_targets.append([_target.x, _target.y])
        all_targets.append([_target.x, _target.y])

    # generate targets for each sample
    for s in range(samples):
        training_targets = [first_targets[s]]

        for i in range(targets_per_sample + 1):
            _target = simulation._generate_target()
            while (
                # Make sure far engough from previous target
                (
                    i != 0
                    and euclidean_distance(
                        _target.x,
                        _target.y,
                        training_targets[i - 1][0],
                        training_targets[i - 1][1],
                    )
                    < random.randint(200, 500)
                )
                # Make sure far enough from any other targets
                or any(
                    euclidean_distance(_target.x, _target.y, target[0], target[1]) < 100
                    for target in all_targets
                )
            ):
                _target = simulation._generate_target()

            training_targets.append([_target.x, _target.y])
            all_targets.append([_target.x, _target.y])
            print(f"{s}-{i}")

        training_targets_list.append(training_targets)

    return training_targets_list


def test_models(start, end, step=1):
    best = (0, 0)
    for i in range(start, end, step):
        try:
            model = GraphModel(f"models/model-{i}.pkl")
            inputs = [
                (model, copy.deepcopy(targets)) for targets in TRAINING_TARGETS_LIST
            ]

            score, time, targets = 0, 0, 0
            for model, training_targets in inputs:
                score_, simulation, time_ = calc_single_fitness_score(
                    (model, training_targets)
                )
                score += score_
                time += time_
                targets += simulation.score

            best = max(best, (score, i))
            print(
                f"MODEL: {i}, Collected Targets: {targets}, Score: {score}, Time: {time}"
            )
        except FileNotFoundError:
            break

    print(f"BEST MODEL: {best[1]} Score: {best[0]}")


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
            print(
                f"MODEL: {i}, Collected Targets: {simulation.score}, Score: {score}, Time: {time}"
            )


if __name__ == "__main__":
    # display_games(use_pygame=True, n=10)
    # test_models(1, 500)

    training_targets_list = generate_training_targets(samples=20, targets_per_sample=3)

    print(training_targets_list)
