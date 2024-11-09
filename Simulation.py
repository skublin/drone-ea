import pygame
import Drone
import Target


class Simulation:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.WIDTH = width
        self.HEIGHT = height
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("2D Drone Simulation")

        # Create instances of Drone and Target
        self.drone = Drone.Drone(x=(self.WIDTH // 2) - 75, y=self.HEIGHT // 2)

        # self.target = Target.Target(x=self.WIDTH // 4, y=self.HEIGHT // 4)
        self.clock = pygame.time.Clock()
        self.running = True

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

    def draw(self):
        # Clear the screen
        self.window.fill((162, 210, 255))  # Fill the window with white

        # Draw the drone and the target
        self.drone.draw(self.window)

        # Update the display
        pygame.display.flip()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
