import numpy as np


class GameObject:
    def __init__(self, _game, pos=(0, 0, 0), color=None, name="object"):
        if len(pos) != 3:
            raise "pos of object is not 3-dimensional"
        self._game = _game
        self._game.obj_types[type(self)] += 1
        self.pos = np.array(pos, float)
        self.color = color
        self.name = name

    def update(self):
        pass

    def move(self, speed):
        self.pos += speed

    def collision(self):
        own_size = np.array([0, 0, 0])
        if hasattr(self, 'size'):
            own_size = self.size / 2
        for obj in self._game.objects:
            if obj == self:
                continue
            other_size = np.array([0, 0, 0])
            if hasattr(obj, 'size'):
                other_size = obj.size / 2
            if (obj.pos[0] + other_size[0] >= self.pos[0] - own_size[0]
                    and obj.pos[0] - other_size[0] <= self.pos[0] + own_size[0]
                    and obj.pos[1] + other_size[1] >= self.pos[1] - own_size[1]
                    and obj.pos[1] - other_size[1] <= self.pos[1] + own_size[1]
                    and obj.pos[2] + other_size[2] >= self.pos[2] - own_size[2]
                    and obj.pos[2] - other_size[2] <= self.pos[2] + own_size[2]):
                return True
        return False

    def set_pos(self, pos):
        self.pos = pos


class Cube(GameObject):
    def __init__(self, _game, pos, size, color=(20, 20, 20), name="block"):
        if len(size) != 3:
            raise "size of block is not 3-dimensional"
        super().__init__(_game, pos, color, name)
        self.size = np.array(size, float)


class Floor(GameObject):
    def __init__(self, _game, pos, size, color=(40, 40, 40), name="floor"):
        if len(size) != 3 or size[2] != 0:
            raise "size of floor is not 3-dimensional or floor is not flat"
        super().__init__(_game, pos, color, name)
        self.size = np.array(size, float)


class MovingObject(GameObject):
    def __init__(self, _game, pos, speed=(0, 0, 0), acceleration=(0, 0, 0), color=None, name="moving_object"):
        if len(speed) != 3 or len(acceleration) != 3:
            raise "speed or acceleration of moving object is not 3-dimensional"
        super().__init__(_game, pos, color, name)
        self.speed = np.array(speed, float)
        self.acceleration = np.array(acceleration, float)

    def update(self):
        self.accelerate(self.acceleration)

    def accelerate(self, acceleration):
        self.speed += acceleration

    def stop(self):
        self.acceleration = np.zeros(3)
        self.speed = np.zeros(3)

    def stay_down(self):
        self.acceleration[2] = 0
        self.speed[2] = 0

    def fall(self):
        self.acceleration[2] = self._game.gravity


class Player(MovingObject):
    def __init__(self, _game, pos=(0, 0, 0), size=(5, 5, 10), strength=50,
                 speed=(0, 0, 0), acceleration=(0, 0, 0), color=None, name="player"):
        if len(size) != 3:
            raise "size of player is not 3-dimensional"
        super().__init__(_game, pos, speed, acceleration, color, name)
        self._game = _game
        self.size = np.array(size, float)
        self.strength = strength

    def update(self):
        if self._game.gamemode == "fall":
            if not self.collision():  # in z
                self.fall()
            elif self.speed[2] > 0:
                self.fall()
                self.speed[2] = 0  # fall from up
        super().update()
        if self._game.gamemode == "fall" and self.collision() and self.speed[2] <= 0:
            self.stay_down()

    def jump(self):
        self.speed[2] = self.strength


''' movements = {"right": (1, 0, 0), "left": (-1, 0, 0), "up": (0, 1), "down": (0, -1, 0),
                 "forward": (0, 0, 1), "backward": (0, 0, -1)} '''
