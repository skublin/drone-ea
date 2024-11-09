import pygame

class Target:
    def __init__(self, x, y, size=10, color=(255, 0, 0)):
        self.x = x  # X position
        self.y = y  # Y position
        self.size = size  # Size of the target (for drawing)
        self.color = color  # Color of the target

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.size)
