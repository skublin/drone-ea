import math
import pygame


class Drone:
    def __init__(self, x, y, width=150):
        self.x = x  # X position
        self.y = y  # Y position

        imp = pygame.image.load("assets/drone-1.png").convert_alpha()
        # resize the drone image
        height = int(imp.get_height() * (width / imp.get_width()))
        self.imp = pygame.transform.scale(imp, (width, height))

        # Drone properties (needed for physics)
        self.mass, self.drone_size, self.gravity = 1, 25, 0.1

        # force of left and right thrusters
        self.left_thruster, self.right_thruster = 0.04, 0.04

        self.thruster_amplitude = 0.04  # Amplitude with which speed is increased
        self.thruster_diff_amplitude = (
            0.003  # Amplitude with which speed between thrusters is decreased
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

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                # Increase both thrusters force
                self.left_thruster += self.thruster_amplitude
                self.right_thruster += self.thruster_amplitude
            elif event.key == pygame.K_s:
                # Decrease both thrusters force
                self.left_thruster -= self.thruster_amplitude
                self.right_thruster -= self.thruster_amplitude
            elif event.key == pygame.K_a:
                # Decrease left thruster force
                self.left_thruster -= self.thruster_diff_amplitude
            elif event.key == pygame.K_d:
                # Decrease right thruster force
                self.right_thruster -= self.thruster_diff_amplitude
        if event.type == pygame.KEYUP and event.key in (
            pygame.K_w,
            pygame.K_s,
            pygame.K_a,
            pygame.K_d,
        ):
            # Reset thrusters to default values
            self.left_thruster, self.right_thruster = 0.04, 0.04

    def move(self):
        self.x_speed += self.x_acceleration
        self.y_speed += self.y_acceleration
        self.angular_speed += self.angular_acceleration

        self.x += self.x_speed
        self.y += self.y_speed
        self.angle += self.angular_speed

    def draw(self, surface):
        # draw drone and rotate it based on angle
        drone = pygame.transform.rotate(self.imp, self.angle)
        surface.blit(drone, (self.x, self.y))
        # pygame.draw.rect(
        #     surface, self.color, (self.x, self.y, self.size[0], self.size[1])
        # )
