import pyglet
import util


class Button:
    def __init__(self):
        self.position = util.Vector()
        self.size = util.Vector()
        self.corner_radius = 10
        self.outline_width = 7

        self.base = pyglet.shapes.RoundedRectangle(0, 0, 0, 0, color = (180, 180, 180),
                                                   radius = self.corner_radius, segments = 8)
        self.outline = pyglet.shapes.RoundedRectangle(0, 0, 0, 0, color = (170, 170, 170),
                                                      radius = self.corner_radius + self.outline_width, segments = 8)

        self.visible = True
        self.state_active = True
        self.state_hover = False
        self.state_down = False
        self.pressed = False
        self.released = False

    def set_visible(self, visible):
        self.visible = visible
        self.base.visible = self.visible
        self.outline.visible = self.visible
        return self

    def set_size(self, size):
        self.size = size
        self.base.width, self.base.height = self.size.decode()
        self.outline.width, self.outline.height = (self.size + self.outline_width * 2).decode()
        return self

    def set_position(self, position):
        self.position = position
        self.base.position = self.position.decode()
        self.outline.position = (self.position - self.outline_width).decode()
        return self

    def set_center(self, position):
        self.set_position(position - self.size // 2)
        return self

    def set_geometry(self, position, size):
        self.set_position(position)
        self.set_size(size)
        return self

    def set_corner_radius(self, radius):
        self.corner_radius = radius
        self.base.radius = self.corner_radius
        self.outline.radius = self.corner_radius + self.outline_width
        return self

    def set_outline_width(self, width):
        self.outline_width = width
        self.outline.width, self.outline.height = self.size.decode() + self.outline_width * 2
        self.outline.position = self.position.decode() - self.outline_width
        return self

    def collide_point(self, position):
        return self.position.x < position.x < self.position.x + self.size.x and self.position.y < position.y < self.position.y + self.size.y

    def refresh_colour(self):
        if not self.state_active:
            self.base.color = (180, 180, 180)
            self.outline.color = (170, 170, 170)
        elif self.state_down:
            self.base.color = (170, 170, 170)
            self.outline.color = (165, 165, 165)
        elif self.state_hover:
            self.base.color = (185, 185, 185)
            self.outline.color = (175, 175, 175)
        else:
            self.base.color = (180, 180, 180)
            self.outline.color = (170, 170, 170)

    def update(self, mouse_position, pressed, released):
        self.pressed = False
        self.released = False

        if not self.visible:
            return

        if not self.state_active:
            self.state_hover = False
            self.state_down = False
            self.refresh_colour()
            return

        if self.collide_point(mouse_position):
            self.state_hover = True
            if pressed:
                self.state_down = True
                self.pressed = True
        else:
            self.state_hover = False

        if self.state_down and released:
            self.state_down = False
            if self.collide_point(mouse_position):
                self.released = True

        self.refresh_colour()


class TextButton(Button):
    def __init__(self):
        super().__init__()
        self.text = "Button"
        self.font_name = "arial"
        self.font_size = 25
        self.font_buffer = util.Vector(20, 10)

        self.label = pyglet.text.Label(self.text, 0, 0, anchor_x = "center", anchor_y = "center",
                                       color = (255, 255, 255),
                                       font_name = self.font_name, font_size = self.font_size)
        self.fit_text()

    def set_visible(self, visible):
        super().set_visible(visible)
        self.label.visible = self.visible
        return self

    def fit_text(self):
        self.set_size(util.Vector(self.label.content_width, self.label.content_height) + self.font_buffer * 2)
        return self

    def set_text(self, text):
        self.text = text
        self.label.text = self.text
        return self

    def set_font(self, font_name = None, font_size = None):
        self.font_name = font_name or self.font_name
        self.font_size = font_size or self.font_size
        self.label.font_name = self.font_name
        self.label.font_size = self.font_size

    def set_size(self, size):
        super().set_size(size)
        self.label.x, self.label.y = (self.position + self.size // 2).decode()
        return self

    def set_position(self, position):
        super().set_position(position)
        self.label.x, self.label.y = (self.position + self.size // 2).decode()
        return self

    def to_batch(self, batch):
        self.outline.batch = batch
        self.base.batch = batch
        self.label.batch = batch
        return self

    def refresh_colour(self):
        super().refresh_colour()
        if not self.state_active:
            self.label.color = (255, 255, 255)
        elif self.state_down:
            self.label.color = (255, 255, 255)
        elif self.state_hover:
            self.label.color = (255, 255, 255)
        else:
            self.label.color = (255, 255, 255)