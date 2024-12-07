import tensorflow as tf
from simulation import Simulation
from settings import Settings
import numpy
import copy
import pickle


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
def model_build(in_dimen, out_dimen):
    input_layer = tf.keras.layers.Input(shape=(in_dimen,))
    dense_layer1 = tf.keras.layers.Dense(8, activation="tanh")
    dense_layer2 = tf.keras.layers.Dense(8, activation="tanh")
    output_layer = tf.keras.layers.Dense(out_dimen, activation="sigmoid")

    model = tf.keras.Sequential([input_layer, dense_layer1, dense_layer2, output_layer])

    # Just like before, compilation is not required for the algorithm to run
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    return model


class GraphModel:
    """Class used to convert keras model into TensorFlow graph (boosting inference speed significantly)"""

    def __init__(self, model_name=None, model=None):
        self.model = self.get_model(model_name, model)

    def get_model(self, model_name: str | None, model: tf.keras.Model | None):
        if model is None and model_name is None:
            raise ValueError("Provide either model or model_name")

        if model:
            return model

        # input_layer = tf.keras.layers.Input(shape=(6,))
        # dense_layer1 = tf.keras.layers.Dense(9, activation="tanh")
        # dense_layer2 = tf.keras.layers.Dense(9, activation="tanh")
        # output_layer = tf.keras.layers.Dense(2, activation="sigmoid")

        # input_layer = tf.keras.layers.Input(shape=(6,))
        # dense_layer1 = tf.keras.layers.Dense(32, activation="relu")
        # output_layer = tf.keras.layers.Dense(2, activation="sigmoid")
        # model = tf.keras.Sequential([input_layer, dense_layer1, output_layer])

        model = model_build(6, 2)

        if model_name.endswith(".h5"):
            model.load_weights(model_name)
        elif model_name.endswith(".pkl"):
            weights = pickle.load(open(model_name, "rb"))
            model.set_weights(model_weights_as_matrix(model, weights))
        else:
            raise ValueError("Invalid model file format")

        return model

    def on_retry_fail(self, e):
        print(f"Failed to predict: {e}")
        return [[0, 0]]

    @tf.function
    def _predict(self, nn_input):
        return self.model(nn_input, training=False)

    def predict(self, nn_input, retries=1):
        for i in range(retries + 1):
            try:
                return self._predict(nn_input)
            except Exception as e:
                if i == retries:
                    return self.on_retry_fail(e)


def calc_fitness_score(graph_model, training_targets):
    simulation = Simulation(
        use_pygame=False, settings=Settings(), targets=copy.deepcopy(training_targets)
    )

    sim_time = 6  # 5 seconds
    iterations = sim_time * simulation.FPS

    target_reward, survive_reward, loss_penalty, time_penalty, flip_penalty = (
        300,
        2,
        1000,
        800,
        100,  # - 100 per each second
    )

    # max score that can be achieved
    score = loss_penalty * 2  # initial score
    max_score = (
        score + target_reward * len(training_targets) + sim_time * survive_reward
    )

    cur_target_iterations = 1

    for frame_num in range(1, iterations + 1):
        prev_score = simulation.score

        # add survive reward per each survived second
        score += survive_reward / simulation.FPS

        # add penalty based on distance to target
        # score will be decreasing if drone not moving towards the target
        score -= simulation.calculate_target_distance() / (
            simulation.target_margin * simulation.FPS
        )

        # add penalty if drone is flipped
        if simulation.drone.is_flipped:
            score -= flip_penalty / simulation.FPS

        # get predictions from the model
        predictions = graph_model.predict(numpy.array([simulation.nn_input]), retries=4)
        simulation.next(predictions=[float(p) for p in predictions[0]])

        # player out of bounds
        if not simulation.running:
            score -= loss_penalty
            # time penalty
            score -= ((iterations - frame_num) / iterations) * time_penalty
            break

        # player reached the target
        if simulation.score > prev_score:
            # based on time to reach the target add 50 to 100 % of target reward
            score += target_reward * (
                (2 * iterations - cur_target_iterations) / (2 * iterations)
            )
            cur_target_iterations = 0
            if simulation.score == len(training_targets):
                break

        cur_target_iterations += 1

    # # normalize score (from 0 to 1)
    score = max(score, 0) / max_score
    return score, frame_num / simulation.FPS, simulation


def calc_single_fitness_score(_input):
    graph_model, training_targets = _input

    single_score, time, simulation = calc_fitness_score(graph_model, training_targets)

    return single_score, simulation, time
