import tensorflow as tf


class GraphModel:
    """Class used to convert keras model into TensorFlow graph (boosting inference speed significantly)"""

    def __init__(self, model_name=None, model=None):
        self.model = self.get_model(model_name, model)

    def get_model(self, model_name: str | None, model: tf.keras.Model | None):
        if model is None and model_name is None:
            raise ValueError("Provide either model or model_name")

        if model:
            return model

        input_layer = tf.keras.layers.Input(shape=(6,))
        dense_layer1 = tf.keras.layers.Dense(9, activation="tanh")
        dense_layer2 = tf.keras.layers.Dense(9, activation="tanh")
        output_layer = tf.keras.layers.Dense(4, activation="sigmoid")

        model = tf.keras.Sequential(
            [input_layer, dense_layer1, dense_layer2, output_layer]
        )
        if model_name:
            model.load_weights(model_name)

        return model

    @tf.function
    def predict(self, nn_input):
        return self.model(nn_input, training=False)
