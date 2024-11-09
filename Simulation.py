import pygame
import Drone
import Target
import random


class Simulation:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.WIDTH = width
        self.HEIGHT = height
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("2D Drone Simulation")

        # Create instances of Drone and Target
        self.drone = Drone.Drone(x=(self.WIDTH // 2) - 75, y=self.HEIGHT // 2)
        self.target = self._generate_target()

        self.clock = pygame.time.Clock()
        self.running = True

    def _generate_target(self) -> Target.Target:
        # add 50 padding
        x = random.randint(50, self.WIDTH - 50)
        y = random.randint(50, self.HEIGHT - 50)
        return Target.Target(x, y)

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
            # elif event.type == pygame.MOUSEBUTTONDOWN:
            #     if event.button == 1:  # Left mouse button
            #         # Update target position to where the mouse was clicked
            #         pos = pygame.mouse.get_pos()
            #         x, y = pos
            #         x = min(max(x, 0), self.WIDTH)
            #         y = min(max(y, 0), self.HEIGHT)
            #         self.target.x, self.target.y = pos
            #         print(f"Target moved to {pos}")

    def update(self):
        # Move the drone towards the target
        self.drone.move(pressed=pygame.key.get_pressed())

        if self.calculate_target_distance() < 40:
            # Generate a new target if the drone has reached the current target
            self.target = self._generate_target()

    def draw(self):
        # Clear the screen
        self.window.fill((162, 210, 255))  # Fill the window with white

        # Draw the drone and the target
        self.drone.draw(self.window)
        self.target.draw(self.window)

        # Update the display
        pygame.display.flip()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
