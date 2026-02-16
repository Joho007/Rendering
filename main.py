import pygame
import gameObjects
import rendering


class Game:
    def __init__(self, size, color, fps, gravity, jump):
        # graphics
        pygame.init()
        self.clock = pygame.time.Clock()
        self.fps = fps
        # objects
        self.objects = []
        self.obj_types = Game.objectCounter()
        self.player = gameObjects.Player(self, strength=jump / fps)
        self.load_objects()
        # mechanics
        self.gravity = -gravity / fps
        self.gamemodes = ["fly", "fall"]
        self.gamemode = "fly"
        self.camera = rendering.Camera((0, 0, 20), (0, 0, 0), 50)
        self.renderer = rendering.Renderer(self.camera, self.objects, size[0], size[1], color)
        self.handler = EventHandler(self)
        self.running = True
        self.run()  # start game

    def run(self):
        while self.running:
            for obj in self.objects:
                obj.update()
            self.handler.check_events()
            # move the camera by the speed of the player and update the player's position
            self.camera.set_center(self.camera.center + self.player.speed)
            self.player.set_pos(self.camera.center)
            # render and tick the clock
            self.renderer.update()
            self.clock.tick(self.fps) / 1000
        pygame.quit()

    def jump(self):
        if self.player.collision():  # in z
            self.player.jump()

    def change_gamemode(self):
        self.gamemode = self.gamemodes[(self.gamemodes.index(self.gamemode) + 1) % len(self.gamemodes)]
        if self.gamemode == "fly":
            self.player.stop()

    def close_game(self):
        self.running = False

    def load_objects(self): # configurable
        self.objects.append(self.player)
        for i in range(-2, 2):
            for j in range(-2, 2):
                self.objects.append(gameObjects.Floor(self, (10 * i, 10 * j, 0), (10, 10, 0)))
        self.objects.append(gameObjects.Cube(self, (0, 30, 20), (20, 20, 20), (127, 255, 0)))
        self.objects.append(gameObjects.Cube(self, (0, 0, 50), (20, 20, 20), (179, 238, 58)))
        self.objects.append(gameObjects.Cube(self, (-40, 20, 0), (20, 20, 20), (0, 238, 118)))

    def create_object(self, obj_type):
        if obj_type in globals():
            self.objects.append(globals()[obj_type]())
        raise ValueError(f"unnamed object type: {obj_type}")

    @staticmethod
    def objectCounter(current=gameObjects.GameObject):
        obj_types = {current: 0}
        for subclass in current.__subclasses__():
            obj_types[subclass] = 0
            obj_types.update(Game.objectCounter(subclass))
        return obj_types

    def allObjects(self):
        return self.objects

    def numberOfInstances(self, subclass):
        return self.obj_types[subclass]


class EventHandler:
    def __init__(self, _game):
        self._game = _game
        self.events = {27: self._game.close_game,  # esc
                       49: self._game.renderer.switch_screen,  # 1
                       50: self._game.camera.change_outlook,  # 2
                       51: self._game.change_gamemode,  # 3
                       52: self._game.camera.reset}  # 4
        self.events_fall = {32: self._game.jump}  # space

    def check_events(self):
        for event in pygame.event.get():  # check events
            if event.type == pygame.QUIT:
                self._game.running = False
            elif event.type == pygame.KEYDOWN:
                self.keys(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mousedown()
            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouseup()
            elif event.type == pygame.MOUSEWHEEL:
                self.mousewheel(event.y)
        self.pressed(pygame.key.get_pressed())

    def keys(self, key):
        if key in self.events:
            self.events[key]()
        if self._game.gamemode == "fall" and key in self.events_fall:
            self.events_fall[key]()

    def pressed(self, keys):
        move_speed = 120 / self._game.fps
        turn_speed = 420 / self._game.fps
        zoom_speed = 5 / self._game.fps
        if keys[pygame.K_LCTRL]:
            move_speed *= 2.5
        # move
        if keys[pygame.K_a]:
            self._game.camera.move("left", move_speed)
        elif keys[pygame.K_d]:
            self._game.camera.move("right", move_speed)
        if keys[pygame.K_w]:
            self._game.camera.move("forward", move_speed)
        elif keys[pygame.K_s]:
            self._game.camera.move("backward", move_speed)
        if self._game.gamemode == "fly":
            if keys[pygame.K_LSHIFT]:
                self._game.camera.move("down", move_speed)
            elif keys[pygame.K_SPACE]:
                self._game.camera.move("up", move_speed)
        # turn
        if keys[pygame.K_LEFT]:
            self._game.camera.turn("left", turn_speed)
        elif keys[pygame.K_RIGHT]:
            self._game.camera.turn("right", turn_speed)
        if keys[pygame.K_DOWN]:
            self._game.camera.turn("down", 0.75 * turn_speed)
        elif keys[pygame.K_UP]:
            self._game.camera.turn("up", 0.75 * turn_speed)
        # zoom
        if self._game.camera.outlook != "first_person":
            if keys[pygame.K_KP_MINUS]:
                self._game.camera.zoom("-", zoom_speed)
            elif keys[pygame.K_KP_PLUS]:
                self._game.camera.zoom("+", zoom_speed)

    def mousedown(self):
        pass

    def mouseup(self):
        pass

    def mousewheel(self, y):
        if self._game.camera.outlook != "first_person":
            self._game.camera.zoom("-", 8 * y / self._game.fps)


game = Game([800, 600], (126, 192, 238), 144, 0.8, 100)
