import pyglet
import util, event


class Button:
    def __init__(self):
        self.position = util.Vector()
        self.size = util.Vector()
        self.corner_radius = 10
        self.outline_width = 7

        self.base_colour = util.RGBA(default = True)
        self.base_colour_hover = util.RGBA(default = True)
        self.base_colour_down = util.RGBA(default = True)
        self.base_colour_inactive = util.RGBA(default = True)

        self.outline_colour = util.RGBA(default = True)
        self.outline_colour_hover = util.RGBA(default = True)
        self.outline_colour_down = util.RGBA(default = True)
        self.outline_colour_inactive = util.RGBA(default = True)

        self.base = pyglet.shapes.RoundedRectangle(0, 0, 0, 0, color = self.base_colour.decode(),
                                                   radius = self.corner_radius, segments = 8)
        self.outline = pyglet.shapes.RoundedRectangle(0, 0, 0, 0, color = self.outline_colour.decode(),
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

    def set_base_colour(self, default = None, hover = None, down = None, inactive = None):
        self.base_colour = default or self.base_colour
        self.base_colour_hover = hover or self.base_colour_hover or self.base_colour
        self.base_colour_down = down or self.base_colour_down or self.base_colour
        self.base_colour_inactive = inactive or self.base_colour_inactive or self.base_colour
        return self

    def set_outline_colour(self, default = None, hover = None, down = None, inactive = None):
        self.outline_colour = default or self.outline_colour
        self.outline_colour_hover = hover or self.outline_colour_hover or self.outline_colour
        self.outline_colour_down = down or self.outline_colour_down or self.outline_colour
        self.outline_colour_inactive = inactive or self.outline_colour_inactive or self.outline_colour
        return self

    def collide_point(self, position):
        return self.position.x < position.x < self.position.x + self.size.x and self.position.y < position.y < self.position.y + self.size.y

    def refresh_colour(self):
        if not self.state_active:
            self.base.color = self.base_colour_inactive.decode()
            self.outline.color = self.outline_colour_inactive.decode()
        elif self.state_down:
            self.base.color = self.base_colour_down.decode()
            self.outline.color = self.outline_colour_down.decode()
        elif self.state_hover:
            self.base.color = self.base_colour_hover.decode()
            self.outline.color = self.outline_colour_hover.decode()
        else:
            self.base.color = self.base_colour.decode()
            self.outline.color = self.outline_colour.decode()

    def update(self, game):
        self.pressed = False
        self.released = False

        if not self.visible:
            return

        if not self.state_active:
            self.state_hover = False
            self.state_down = False
            self.refresh_colour()
            return

        if self.collide_point(game.event.mouse_position):
            self.state_hover = True
            if game.event.mouse_down(event.Mouse.Left):
                self.state_down = True
                self.pressed = True
        else:
            self.state_hover = False

        if self.state_down and game.event.mouse_released(event.Mouse.Left):
            self.state_down = False
            if self.collide_point(game.event.mouse_position):
                self.released = True

        self.refresh_colour()


class TextButton(Button):
    def __init__(self):
        super().__init__()
        self.text = "Button"
        self.font_name = "Noto Sans Math"
        self.font_size = 25
        self.font_buffer = util.Vector(20, 10)

        self.text_colour = util.RGBA(default = True)
        self.text_colour_hover = util.RGBA(default = True)
        self.text_colour_down = util.RGBA(default = True)
        self.text_colour_inactive = util.RGBA(default = True)

        self.label = pyglet.text.Label(self.text, 0, 0, anchor_x = "center", anchor_y = "center",
                                       color = self.text_colour.decode(),
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

    def set_text_colour(self, default = None, hover = None, down = None, inactive = None):
        self.text_colour = default or self.text_colour
        self.text_colour_hover = hover or self.text_colour_hover or self.text_colour
        self.text_colour_down = down or self.text_colour_down or self.text_colour
        self.text_colour_inactive = inactive or self.text_colour_inactive or self.text_colour
        return self

    def refresh_colour(self):
        super().refresh_colour()
        if not self.state_active:
            self.label.color = self.text_colour_inactive.decode()
        elif self.state_down:
            self.label.color = self.text_colour_down.decode()
        elif self.state_hover:
            self.label.color = self.text_colour_hover.decode()
        else:
            self.label.color = self.text_colour.decode()


class Label:
    def __init__(self):
        self.text = "Label"
        self.font_name = "Noto Sans Math"
        self.font_size = 25
        self.colour = util.RGBA(default = True)

        self.position = util.Vector()
        self.visible = True

        self.label = pyglet.text.Label(self.text, 0, 0, color = self.colour.decode(),
                                       font_name = self.font_name, font_size = self.font_size)

    def set_visible(self, visible):
        self.visible = visible
        self.label.visible = self.visible
        return self

    def set_position(self, position):
        self.position = position
        self.label.x, self.label.y = self.position.decode()
        return self

    def set_center(self, position):
        size = util.Vector(self.label.content_width, self.label.content_height)
        self.set_position(position - size // 2)
        return self

    def set_text(self, text, center = False):
        current_center = None
        if center:
            current_center = self.position + util.Vector(self.label.content_width, self.label.content_height) // 2
        self.text = text
        self.label.text = self.text
        if center:
            self.set_center(current_center)
        return self

    def set_font(self, font_name = None, font_size = None):
        self.font_name = font_name or self.font_name
        self.font_size = font_size or self.font_size
        self.label.font_name = self.font_name
        self.label.font_size = self.font_size
        return self

    def set_colour(self, colour):
        self.colour = colour
        self.label.color = self.colour.decode()
        return self


class Entry(Label):
    def __init__(self):
        super().__init__()
        self.active = True
        self.submitted = False
        self.text = ""

    def update(self, game):
        self.submitted = False

        if not (self.active and self.visible):
            return

        for k, v in event.typeable.items():
            if game.event.key_pressed(k):
                self.text += v.upper() if game.event.modifier(event.Modifier.Shift) else v
        if game.event.key_pressed(event.Key.Backspace):
            self.text = self.text[:-1]
        if game.event.key_pressed(event.Key.Return):
            self.submitted = True

        self.label.text = self.text