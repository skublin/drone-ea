class Settings:
    def __init__(self, width=800, height=600):
        self.FPS = 60

        self.WIDTH = width
        self.HEIGHT = height

        self.BOARD_WIDTH, self.BOARD_HEIGHT = 2 * self.WIDTH, 2 * self.HEIGHT


TRAINING_TARGETS = [
    [992, 203],
    [741, 853],
    [849, 99],
    [1046, 915],
    [815, 313],
    [1328, 292],
    [731, 295],
    [298, 141],
    [1211, 628],
    [523, 1151],
]
