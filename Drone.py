import math
import pygame

class Drone:
    def __init__(self, x, y, size=10, color=(0, 255, 0)):
        self.x = x  # X position
        self.y = y  # Y position
        self.size = size  # Size of the drone (for drawing)
        self.color = color  # Color of the drone
        self.speed = 1  # Movement speed

    def move_up(self):
        self.y -= self.speed
        print(f"Drone moves up to ({self.x}, {self.y})")

    def move_down(self):
        self.y += self.speed
        print(f"Drone moves down to ({self.x}, {self.y})")

    def move_left(self):
        self.x -= self.speed
        print(f"Drone moves left to ({self.x}, {self.y})")

    def move_right(self):
        self.x += self.speed
        print(f"Drone moves right to ({self.x}, {self.y})")

    def move_towards(self, target_x, target_y):
        # Calculate the angle to the target
        angle = math.atan2(target_y - self.y, target_x - self.x)
        dx = self.speed * math.cos(angle)
        dy = self.speed * math.sin(angle)

        # Update position
        self.x += dx
        self.y += dy

        # Check if the drone is close enough to the target
        distance = math.hypot(target_x - self.x, target_y - self.y)
        if distance < self.speed:
            self.x = target_x
            self.y = target_y

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (int(self.x), int(self.y), self.size, self.size))
