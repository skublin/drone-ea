import os
import sys
import copy
import math
import pygame
import random
import keras
import numpy
from models import Drone
from models import Target
from utils import GraphModel
from settings import Settings, TRAINING_TARGETS


class Simulation:
    def __init__(
        self,
        use_pygame: bool,
        settings: Settings,
        targets: list[list[int]] | None = None,
    ):

        self.settings = settings

        # Create instances of Drone and Target
        self.drone = Drone(
            use_pygame=use_pygame,
            assets_path=self.assets_path,
            x=self.BOARD_WIDTH / 2,
            y=self.BOARD_HEIGHT / 2,
        )

        self.targets = targets
        self.target = self._generate_target()

        self.score = 0
        self.running = True

    @property
    def FPS(self):
        return self.settings.FPS

    @property
    def WIDTH(self):
        return self.settings.WIDTH

    @property
    def HEIGHT(self):
        return self.settings.HEIGHT

    @property
    def BOARD_WIDTH(self):
        return self.settings.BOARD_WIDTH

    @property
    def BOARD_HEIGHT(self):
        return self.settings.BOARD_HEIGHT

    @property
    def assets_path(self):
        return os.path.join(os.path.dirname(__file__), "assets")

    @property
    def nn_input(self):
        """
        Return the input to the neural network
            - distance to target by X axis
            - distance to target by Y axis
            - drone speed by X axis
            - drone speed by Y axis
            - drone angular speed
            - drone angle
        """

        return [
            self.drone.x - self.target.x,
            self.drone.y - self.target.y,
            self.drone.x_speed,
            self.drone.y_speed,
            self.drone.angular_speed,
            self.drone.angle,
        ]

    def _generate_target(self) -> Target:
        if self.targets:
            position = self.targets.pop(0)
            return Target(position[0], position[1])

        # add padding 1.5 times the width and height of the drone
        x = random.randint(
            int(self.drone.width * 1.5), self.BOARD_WIDTH - int(self.drone.width * 1.5)
        )
        y = random.randint(
            int(self.drone.height * 1.5),
            self.BOARD_HEIGHT - int(self.drone.height * 1.5),
        )

        return Target(x, y)

    def calculate_target_distance(self):
        return (
            (self.drone.x - self.target.x) ** 2 + (self.drone.y - self.target.y) ** 2
        ) ** 0.5

    def out_of_bounds(self):

        if (
            self.drone.x < 0
            or self.drone.x > self.BOARD_WIDTH
            or self.drone.y < 0
            or self.drone.y > self.BOARD_HEIGHT
        ):
            return True

        return False

    def next(self, keys: list[int]):
        """
        Update the game state based on the keys pressed by the player
            - keys: list of 4 integers representing the keys pressed by the player - [W, S, A, D]
        """

        # Move the drone towards the target
        self.drone.move(keys=keys)

        if self.calculate_target_distance() < 40:  # 40 = margin
            # Generate a new target if the drone has reached the current target
            self.score += 1
            self.target = self._generate_target()

        if self.out_of_bounds():
            self.running = False


class Game:
    def __init__(self, model_name: str, targets: list[list[int]] | None = None):
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
            input_layer = keras.layers.Input(shape=(6,))
            dense_layer1 = keras.layers.Dense(9, activation="tanh")
            dense_layer2 = keras.layers.Dense(9, activation="tanh")
            output_layer = keras.layers.Dense(4, activation="sigmoid")

            model = keras.Sequential(
                [input_layer, dense_layer1, dense_layer2, output_layer]
            )
            model.load_weights(self.model_name)
            self._model = GraphModel(model=model)

        return self._model

    @property
    def keys(self) -> list[int]:
        if self.model:
            predictions = self.model.predict(numpy.array([self.simulation.nn_input]))
            return [1 if p >= 0.5 else 0 for p in predictions[0]]

        keys = pygame.key.get_pressed()
        return [keys[pygame.K_w], keys[pygame.K_s], keys[pygame.K_a], keys[pygame.K_d]]

    def run(self):
        while self.simulation.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.clock.tick(self.settings.FPS)  # Limit the frame rate to 60 FPS
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
    game = Game(
        model_name="models/model-48.h5", targets=copy.deepcopy(TRAINING_TARGETS)
    )
    game.run()
