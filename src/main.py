import os
import pygame
import random
from models import Drone
from models import Target


class Simulation:
    def __init__(self, width=800, height=600):
        pygame.init()
        pygame.display.set_caption("Drone Simulation")

        self.WIDTH = width
        self.HEIGHT = height

        self.BOARD_WIDTH, self.BOARD_HEIGHT = 2 * self.WIDTH, 2 * self.HEIGHT
        bg_img = pygame.image.load(os.path.join(self.assets_path, "bg.jpg"))
        self.bg = pygame.transform.scale(bg_img, (self.BOARD_WIDTH, self.BOARD_HEIGHT))

        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        # Create instances of Drone and Target
        self.drone = Drone(
            assets_path=self.assets_path,
            x=self.BOARD_WIDTH / 2,
            y=self.BOARD_HEIGHT / 2,
        )
        self.target = self._generate_target()

        self.clock = pygame.time.Clock()
        self.running = True

    @property
    def assets_path(self):
        return os.path.join(os.path.dirname(__file__), "assets")

    def _generate_target(self) -> Target:
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

    def run(self):
        while self.running:
            self.clock.tick(60)  # Limit the frame rate to 60 FPS
            self.handle_events()
            self.update()
            self.draw()

        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        # Move the drone towards the target
        self.drone.move(pressed=pygame.key.get_pressed())

        if self.calculate_target_distance() < 40:
            # Generate a new target if the drone has reached the current target
            self.target = self._generate_target()

    def draw(self):
        # Clear the screen
        self.window.fill((162, 210, 255))  # Fill the window with white

        bg_x, bg_y = (
            -(self.BOARD_WIDTH - self.window.get_width()) / 2,
            -(self.BOARD_HEIGHT - self.window.get_height()) / 2,
        )

        if (
            self.drone.x > self.window.get_width() / 2
            and self.drone.x < self.BOARD_WIDTH - self.window.get_width() / 2
        ):
            bg_x -= self.drone.x - self.BOARD_WIDTH / 2
        elif self.drone.x <= self.window.get_width() / 2:
            bg_x = 0
        elif self.drone.x >= self.BOARD_WIDTH - self.window.get_width() / 2:
            bg_x = -self.BOARD_WIDTH + self.window.get_width()

        if (
            self.drone.y > self.window.get_height() / 2
            and self.drone.y < self.BOARD_HEIGHT - self.window.get_height() / 2
        ):
            bg_y -= self.drone.y - self.BOARD_HEIGHT / 2
        elif self.drone.y <= self.window.get_height() / 2:
            bg_y = 0
        elif self.drone.y >= self.BOARD_HEIGHT - self.window.get_height() / 2:
            bg_y = -self.BOARD_HEIGHT + self.window.get_height()

        self.window.blit(self.bg, (bg_x, bg_y))

        # Draw the drone and the target
        self.drone.draw(self.window, self.BOARD_WIDTH, self.BOARD_HEIGHT)
        self.target.draw(self.window, self.drone, self.BOARD_WIDTH, self.BOARD_HEIGHT)

        # Update the display
        pygame.display.flip()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
