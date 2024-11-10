import pygame


class Target:
    def __init__(self, x, y, size=10, color=(255, 0, 0)):
        self.x = x  # X position
        self.y = y  # Y position
        self.size = size  # Size of the target (for drawing)
        self.color = color  # Color of the target

    def display_x(self, surface, drone, board_width):
        return drone.display_x(surface, board_width) + self.x - drone.x

    def display_y(self, surface, drone, board_height):
        return drone.display_y(surface, board_height) + self.y - drone.y

    def arrow_x(self, surface, drone):
        vector = self.vector(drone)
        x = (vector[0] / vector[1]) * (surface.get_height() / 2)
        return (
            min(abs(x), surface.get_width() / 2) * (1 if vector[0] > 0 else -1)
            + surface.get_width() / 2
        )

    def arrow_y(self, surface, drone):
        vector = self.vector(drone)
        y = (vector[1] / vector[0]) * (surface.get_width() / 2)
        return (
            min(abs(y), surface.get_height() / 2) * (1 if vector[1] > 0 else -1)
            + surface.get_height() / 2
        )

    def vector(self, drone):
        # Return the vector from the drone to the target
        return (self.x - drone.x, self.y - drone.y)

    def draw(self, surface, drone, board_width, board_height):
        x = self.display_x(surface, drone, board_width)
        y = self.display_y(surface, drone, board_height)

        if x < 0 or x >= surface.get_width() or y < 0 or y >= surface.get_height():
            # Draw the arrow pointing to the target if target is out of bounds
            x = self.arrow_x(surface, drone)
            y = self.arrow_y(surface, drone)
            pygame.draw.circle(surface, self.color, (x, y), self.size)
        else:
            pygame.draw.circle(surface, self.color, (x, y), self.size)
