import json, sys, os, math
from typing import Any


class Vector:
    def __init__(self, x: Any = 0, y: Any = 0):
        self.x, self.y = x, y

    def __str__(self):
        return f"<{self.x}, {self.y}>"

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return Vector(self.x + other, self.y + other)

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        return Vector(self.x - other, self.y - other)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x * other.x, self.y * other.y)
        return Vector(self.x * other, self.y * other)

    def __truediv__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x / other.x, self.y / other.y)
        return Vector(self.x / other, self.y / other)

    def __floordiv__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x // other.x, self.y // other.y)
        return Vector(self.x // other, self.y // other)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y

    def __iter__(self):
        yield self.x
        yield self.y

    def set(self, x, y):
        self.x, self.y = x, y

    def decode(self):
        return self.x, self.y

    def invert(self):
        return Vector(self.y, -self.x)

    def magnitude(self):
        return math.sqrt(self.magnitude_squared())

    def round(self):
        return Vector(round(self.x), round(self.y))

    def magnitude_squared(self):
        return self.x ** 2 + self.y ** 2

    def normalize(self):
        if self.magnitude() == 0:
            return Vector(0, 0)
        return self / self.magnitude()

    def rotate(self, angle):
        rad = math.radians(angle)
        cos, sin = math.cos(rad), math.sin(rad)
        return Vector(self.x * cos - self.y * sin, self.x * sin + self.y * cos)

    def copy(self):
        return Vector(self.x, self.y)

def dot(a, b):
    return a.x * b.x + a.y * b.y

def cross(a, b):
    return a.x * b.y - a.y * b.x


def path(name):
    try:
        directory = sys._MEIPASS
    except AttributeError:
        return name
    return os.path.join(directory, name)