import os
import math
import pygame


class Drone:
    def __init__(self, use_pygame, assets_path, x, y, width=150):
        self.x = x  # X position
        self.y = y  # Y position

        self.HEIGHT_WIDTH_RATIO = 0.22
        self.width = width
        self.height = self.HEIGHT_WIDTH_RATIO * self.width

        self.use_pygame = use_pygame

        if self.use_pygame:
            imp1 = pygame.image.load(
                os.path.join(assets_path, "drone-1.png")
            ).convert_alpha()
            imp2 = pygame.image.load(
                os.path.join(assets_path, "drone-2.png")
            ).convert_alpha()

            # resize the drone image
            self.imp1 = pygame.transform.scale(imp1, (width, self.height))
            self.imp2 = pygame.transform.scale(imp2, (width, self.height))
            self.img = 1

        # Drone properties (needed for physics)
        self.mass, self.drone_size, self.gravity = 1, 30, 0.1

        # force of left and right thrusters
        self.left_thruster, self.right_thruster = 0.04, 0.04

        self.thruster_amplitude = 0.04  # Amplitude with which speed is increased
        self.thruster_diff_amplitude = (
            0.002  # Amplitude with which speed between thrusters is decreased
        )

        # angle, speed responsible for drone rotation
        self.angle, self.angular_speed = 0, 0

        # x, y drone movement
        self.x_speed = 0
        self.y_speed = 0

    @property
    def x_acceleration(self):
        return (
            -(self.left_thruster + self.right_thruster)
            * math.sin(self.angle * math.pi / 180)
            / self.mass
        )

    @property
    def y_acceleration(self):
        return (
            -(self.left_thruster + self.right_thruster)
            * math.cos(self.angle * math.pi / 180)
            / self.mass
        ) + self.gravity

    @property
    def angular_acceleration(self):
        return (
            (self.right_thruster - self.left_thruster) * self.drone_size
        ) / self.mass

    def display_x(self, surface, board_width):
        # X position of the drone on the display

        if self.x <= surface.get_width() / 2:
            return self.x
        elif self.x >= board_width - surface.get_width() / 2:
            return self.x - (board_width - surface.get_width())
        else:
            return surface.get_width() / 2

    def display_y(self, surface, board_height):
        if self.y <= surface.get_height() / 2:
            return self.y
        elif self.y >= board_height - surface.get_height() / 2:
            return self.y - (board_height - surface.get_height())
        else:
            return surface.get_height() / 2

    def _update_thruster_force(self, keys: list[int]):
        """
        There can be multiple keys pressed at the same time
            - keys: list of 4 integers representing the keys pressed - [W, S, A, D]
        """

        # Set thrusters to default value
        self.left_thruster, self.right_thruster = (
            self.thruster_amplitude,
            self.thruster_amplitude,
        )

        if keys[0]:  # W
            # Increase both thrusters force
            self.left_thruster += self.thruster_amplitude
            self.right_thruster += self.thruster_amplitude

        if keys[1]:  # S
            # Decrease both thrusters force
            self.left_thruster -= self.thruster_amplitude
            self.right_thruster -= self.thruster_amplitude

        if keys[2]:  # A
            # Decrease left thruster force
            self.left_thruster -= self.thruster_diff_amplitude

        if keys[3]:  # D
            # Decrease right thruster force
            self.right_thruster -= self.thruster_diff_amplitude

    def move(self, keys: list[int]):
        self._update_thruster_force(keys)

        self.x_speed += self.x_acceleration
        self.y_speed += self.y_acceleration
        self.angular_speed += self.angular_acceleration

        self.x += self.x_speed
        self.y += self.y_speed
        self.angle += self.angular_speed

    def draw(self, surface, board_width, board_height):
        # draw drone and rotate it based on angle

        if not self.use_pygame:
            return

        if self.img == 1:
            drone = pygame.transform.rotate(self.imp1, self.angle)
            self.img = 2
        else:
            drone = pygame.transform.rotate(self.imp2, self.angle)
            self.img = 1

        display_x = self.display_x(surface, board_width)
        display_y = self.display_y(surface, board_height)

        # get center of the drone
        drone_x, drone_y = (
            display_x - drone.get_width() // 2,
            display_y - drone.get_height() // 2,
        )

        surface.blit(drone, (drone_x, drone_y))
        pygame.draw.circle(surface, (0, 0, 0), (display_x, display_y), 10)
