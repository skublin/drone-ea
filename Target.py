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

    def draw(self, surface, drone, board_width, board_height):
        x = self.display_x(surface, drone, board_width)
        y = self.display_y(surface, drone, board_height)
        pygame.draw.circle(surface, self.color, (x, y), self.size)
