import os
import random
from models import Drone
from models import Target
from settings import Settings


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

        self.target_margin = 40  # Margin of error for the drone to reach the target

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

    def next(
        self, keys: list[int] | None = None, predictions: list[float] | None = None
    ):
        """
        Update the game state based on the keys pressed by the player
            - keys: list of 4 integers representing the keys pressed by the player - [W, S, A, D]
        """

        # Move the drone towards the target
        self.drone.move(keys=keys, predictions=predictions)

        if self.calculate_target_distance() < self.target_margin:
            # Generate a new target if the drone has reached the current target
            self.score += 1
            self.target = self._generate_target()

        if self.out_of_bounds():
            self.running = False
