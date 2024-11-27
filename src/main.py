import os
import sys
import pygame
import keras
import copy
import numpy
from utils import GraphModel
from settings import Settings, TRAINING_TARGETS_LIST
from simulation import Simulation
import pickle
from train import model_weights_as_matrix, model_build


class Game:
    def __init__(
        self,
        model_name: str,
        time: int | None = None,
        targets: list[list[int]] | None = None,
    ):
        pygame.init()
        pygame.display.set_caption("Drone Simulation")

        self._model = None
        self.model_name = model_name

        self.settings = Settings()

        # setup pygame window
        bg_img = pygame.image.load(os.path.join(self.assets_path, "bg.jpg"))
        self.bg = pygame.transform.scale(
            bg_img, (self.settings.BOARD_WIDTH, self.settings.BOARD_HEIGHT)
        )

        self.window = pygame.display.set_mode(
            (self.settings.WIDTH, self.settings.HEIGHT)
        )

        self.clock = pygame.time.Clock()

        self.time = time

        # setup game simulation
        self.simulation = Simulation(
            use_pygame=True, settings=self.settings, targets=targets
        )

    @property
    def assets_path(self):
        return os.path.join(os.path.dirname(__file__), "assets")

    @property
    def model(self):
        if self._model is None and self.model_name:
            # input_layer = keras.layers.Input(shape=(6,))
            # dense_layer1 = keras.layers.Dense(32, activation="relu")
            # dense_layer2 = keras.layers.Dense(16, activation="relu")
            # output_layer = keras.layers.Dense(2, activation="sigmoid")

            # model = keras.Sequential(
            #     [input_layer, dense_layer1, dense_layer2, output_layer]
            # )

            # model = keras.Sequential([input_layer, dense_layer1, output_layer])

            model = model_build(6, 2)

            if self.model_name.endswith(".h5"):
                model.load_weights(self.model_name)
            elif self.model_name.endswith(".pkl"):
                weights = pickle.load(open(self.model_name, "rb"))
                model.set_weights(model_weights_as_matrix(model, weights))
            else:
                raise ValueError("Invalid model file format")

            self._model = GraphModel(model=model)

        return self._model

    @property
    def keys(self) -> list[int]:
        keys = pygame.key.get_pressed()
        return [keys[pygame.K_w], keys[pygame.K_s], keys[pygame.K_a], keys[pygame.K_d]]

    @property
    def predictions(self) -> list[float]:
        predictions = self.model.predict(numpy.array([self.simulation.nn_input]))
        return [float(p) for p in predictions[0]]

    def run(self):
        i = 0
        while self.simulation.running:
            i += 1
            if self.time and i > self.time * self.settings.FPS:
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.clock.tick(self.settings.FPS)  # Limit the frame rate to 60 FPS

            if self.model:
                self.simulation.next(predictions=self.predictions)
            else:
                self.simulation.next(self.keys)

            self.draw()

        pygame.quit()

    def draw(self):
        # Clear the screen
        self.window.fill((162, 210, 255))  # Fill the window with white

        drone, target, BOARD_WIDTH, BOARD_HEIGHT = (
            self.simulation.drone,
            self.simulation.target,
            self.settings.BOARD_WIDTH,
            self.settings.BOARD_HEIGHT,
        )

        bg_x, bg_y = (
            -(BOARD_WIDTH - self.window.get_width()) / 2,
            -(BOARD_HEIGHT - self.window.get_height()) / 2,
        )

        # update camera position
        if (
            drone.x > self.window.get_width() / 2
            and drone.x < BOARD_WIDTH - self.window.get_width() / 2
        ):
            bg_x -= drone.x - BOARD_WIDTH / 2
        elif drone.x <= self.window.get_width() / 2:
            bg_x = 0
        elif drone.x >= BOARD_WIDTH - self.window.get_width() / 2:
            bg_x = -BOARD_WIDTH + self.window.get_width()

        if (
            drone.y > self.window.get_height() / 2
            and drone.y < BOARD_HEIGHT - self.window.get_height() / 2
        ):
            bg_y -= drone.y - BOARD_HEIGHT / 2
        elif drone.y <= self.window.get_height() / 2:
            bg_y = 0
        elif drone.y >= BOARD_HEIGHT - self.window.get_height() / 2:
            bg_y = -BOARD_HEIGHT + self.window.get_height()

        self.window.blit(self.bg, (bg_x, bg_y))

        # Draw the drone and the target
        drone.draw(self.window, BOARD_WIDTH, BOARD_HEIGHT)
        target.draw(self.window, drone, BOARD_WIDTH, BOARD_HEIGHT)

        # Update the display
        pygame.display.flip()


if __name__ == "__main__":

    for targets in TRAINING_TARGETS_LIST:
        game = Game(
            model_name="models-2-dim-deap/model-37.pkl",
            # model_name=None,
            time=6,
            targets=copy.deepcopy(targets),
        )
        game.run()

    # for _ in range(20):
    #     game = Game(
    #         model_name="models/model-39.pkl",
    #         # model_name=None,
    #         time=20,
    #         # targets=copy.deepcopy(TRAINING_TARGETS),
    #     )
    #     game.run()
