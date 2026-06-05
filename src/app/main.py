import pyglet, cv2, math
from PIL import Image
import numpy as np
import nn
from util import Vector, path
from gui import TextButton
import random, time


PREDICTION_RATE = 20
SAMPLE_RATE = 5
CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
WEIGHT = np.array([-1, -0.8, -0.1, 1, -0.5, -1, 0.2])


class Meter:
    def __init__(self, position):
        self.position = position
        self.colours = [(245, 1, 1), (248, 98, 5), (245, 198, 49), (149, 197, 49), (0, 200, 52)]
        self.score = 0
        self.score_visual = 0.5

        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.shapes.Sector(self.position.x, self.position.y, 216, 32, 180, 0,
                                               (255, 255, 255), batch = self.batch)
        self.sectors = []
        for i in range(5):
            angle = i * 36 + 18
            pos = Vector(self.position.x + math.cos(math.radians(angle)) * 10, self.position.y + math.sin(math.radians(angle)) * 10)
            self.sectors.append(pyglet.shapes.Sector(pos.x, pos.y - 2, 200, 32, 36, i * 36,
                                             self.colours[i], batch = self.batch))

        self.mask = pyglet.shapes.Sector(self.position.x, self.position.y, 100, 32, 180, 0,
                                               (255, 255, 255), batch = self.batch)

        self.pointer = pyglet.shapes.Triangle(0, 0, 0, 0, 0, 0, (50, 50, 50), batch = self.batch)
        self.center1 = pyglet.shapes.Circle(self.position.x, self.position.y, 17, 32, (50, 50, 50), batch = self.batch)
        self.center2 = pyglet.shapes.Circle(self.position.x, self.position.y, 12, 32, (255, 255, 255), batch = self.batch)
        self.center3 = pyglet.shapes.Circle(self.position.x, self.position.y, 8, 32, (50, 50, 50), batch = self.batch)

        self.score_label = pyglet.text.Label("cortisol score: N/A", self.position.x, self.position.y + 260,
                                             font_name = "arial", font_size = 30, anchor_x = "center", anchor_y = "center",
                                             weight = "bold", batch = self.batch)

    def update_score(self, prediction):
        self.score = np.sum(WEIGHT * prediction)
        return self.score

    def update(self):
        angle = math.pi * (self.score_visual / 2 + 0.5)
        pointer_head = Vector(math.cos(angle) * 150, math.sin(angle) * 150) + self.position
        pointer_right = Vector(math.cos(angle + 1) * 17, math.sin(angle + 1) * 17) + self.position
        pointer_left = Vector(math.cos(angle - 1) * 17, math.sin(angle - 1) * 17) + self.position
        self.pointer.x, self.pointer.y, self.pointer.x2, self.pointer.y2, self.pointer.x3, self.pointer.y3 = *pointer_head, *pointer_left, *pointer_right

        self.score_visual = self.score_visual + (self.score - self.score_visual) * 0.2
        self.score_label.text = f"cortisol score: {round(self.score, 2)}"
        self.score_label.color = (round(i * 0.8) for i in self.colours[round(self.score * 2 + 2)])

        self.batch.draw()

class App:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.camera_size = Vector(int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.window = pyglet.window.Window(self.camera_size.x, self.camera_size.y, "cortisol detector")
        self.size = Vector(*self.window.get_framebuffer_size())
        self.screen_size = self.camera_size * self.camera_size / self.size
        self.window.set_size(self.screen_size.x, self.screen_size.y)
        self.screen_size = self.camera_size

        self.cascade = cv2.CascadeClassifier(path("models/haarcascade_frontalface_default.xml"))
        self.model = nn.load(path("models/v2-5"))
        self.prediction = np.zeros(len(CLASSES))

        self.meter = Meter(Vector(self.screen_size.x - 250, 50))

        self.sample = None
        self.emotion_sample_tick = 0
        self.face_sample_tick = 0

        self.batch = pyglet.graphics.Batch()
        self.stats_batch = pyglet.graphics.Batch()
        self.record_batch = pyglet.graphics.Batch()
        self.err_batch = pyglet.graphics.Batch()

        self.bg = pyglet.shapes.Rectangle(0, 0, self.screen_size.x, self.screen_size.y, (0, 0, 0), batch = self.err_batch)
        self.err_label = pyglet.text.Label("cant find ur camera dawg", self.screen_size.x // 2, self.screen_size.y // 2,
                                           color = (255, 255, 255), anchor_x = "center", anchor_y = "center",
                                           font_name = "arial", font_size = 40, batch = self.err_batch)
        self.lines = [pyglet.shapes.Line(0, 0, 0, 0, 5, (255, 0, 0), batch = self.batch) for _ in range(4)]
        self.pred_labels = [pyglet.text.Label(CLASSES[i], 20, self.camera_size.y - 50 - 50 * i,
                                              font_name = "arial", font_size = 25, batch = self.stats_batch) for i in range(len(CLASSES))]
        self.pred_data = [pyglet.shapes.Rectangle(200, self.camera_size.y - 55 - 50 * i, 0, 40,
                                                  color = (150, 0, 0), batch = self.stats_batch) for i in range(len(CLASSES))]
        self.face_rgb = pyglet.image.ImageData(100, 100, "RGB", np.zeros((100, 100, 3)).tobytes(), pitch = -300)
        self.face_mono = pyglet.image.ImageData(100, 100, "RGB", np.zeros((100, 100, 3)).tobytes(), pitch = -300)

        self.stats = False

        self.mouse_position = Vector(0, 0)
        self.btn_pressed = False
        self.btn_released = False
        self.stats_btn = TextButton().set_text("stats for nerds 🤓").fit_text().to_batch(self.batch).set_center(Vector(self.screen_size.x - 170, self.screen_size.y - 50))

        self.recording = False
        self.record_tick = 0
        self.record_emoji = None
        self.record_btn = TextButton().set_text("record cortisol").fit_text().to_batch(self.batch).set_center(Vector(self.screen_size.x - 148, self.screen_size.y - 140))
        self.shuffle_emoji()

        self.record_counter = pyglet.text.Label("0", self.screen_size.x // 2, self.screen_size.y // 2,
                                                color = (255, 255, 255), anchor_x = "center", anchor_y = "center",
                                                font_name = "arial", font_size = 80, batch = self.record_batch)
        self.record_score = pyglet.text.Label("lowest cortisol: N/A", self.screen_size.x // 2, 55,
                                                color = (255, 255, 255), anchor_x = "center", anchor_y = "center",
                                                font_name = "arial", font_size = 40, weight = "bold", batch = self.record_batch)
        self.record_lowest = -float("inf")

        self.prev_time = time.perf_counter()

    def shuffle_emoji(self):
        self.record_emoji = random.choice(['🤡', '🤫', '🐒', '🗿', '💀', '🗣️'])

    def run(self):
        @self.window.event
        def on_draw():
            self.window.clear()
            success, frame = self.camera.read()
            if success:
                self.update(frame)
                self.batch.draw()
            else:
                self.err_batch.draw()

            if self.stats:
                self.stats_batch.draw()
            self.meter.update()
            if self.recording:
                self.record_batch.draw()

            self.btn_pressed = False
            self.btn_released = False

        @self.window.event
        def on_mouse_press(x, y, button, modifiers):
            self.btn_pressed = True

        @self.window.event
        def on_mouse_release(x, y, button, modifiers):
            self.btn_released = True

        @self.window.event
        def on_mouse_motion(x, y, dx, dy):
            self.mouse_position.set(x, y)

        @self.window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            self.mouse_position.set(x, y)

        pyglet.app.run(1 / 60)
        self.camera.release()

    def toggle_stats(self):
        self.stats = not self.stats
        if self.stats:
            self.stats_btn.set_text("close stats for nerds 😴").fit_text().set_center(Vector(self.screen_size.x - 215, self.screen_size.y - 50))
        else:
            self.stats_btn.set_text("stats for nerds 🤓").fit_text().set_center(Vector(self.screen_size.x - 170, self.screen_size.y - 50))

    def update(self, frame):
        array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = pyglet.image.ImageData(self.camera_size.x, self.camera_size.y, "RGB", array.tobytes(), pitch = -self.camera_size.x * 3)
        img.blit(0, 0)

        self.face_sample_tick += 1
        self.emotion_sample_tick += 1

        if self.face_sample_tick >= SAMPLE_RATE:
            self.face_sample_tick = 0
            face = self.cascade.detectMultiScale(array)
            if len(face) > 0:
                face = face[0]
                x1, y1, x2, y2 = face[0], self.camera_size.y - face[1], face[0] + face[2], self.camera_size.y - (face[1] + face[3])
                self.lines[0].x, self.lines[0].y, self.lines[0].x2, self.lines[0].y2 = x1, y1, x1, y2
                self.lines[1].x, self.lines[1].y, self.lines[1].x2, self.lines[1].y2 = x1, y2, x2, y2
                self.lines[2].x, self.lines[2].y, self.lines[2].x2, self.lines[2].y2 = x2, y2, x2, y1
                self.lines[3].x, self.lines[3].y, self.lines[3].x2, self.lines[3].y2 = x2, y1, x1, y1

                cropped = array[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
                self.process_face(cropped)

            if self.emotion_sample_tick >= PREDICTION_RATE and self.sample is not None:
                self.emotion_sample_tick = 0
                self.prediction = self.model.forward(self.sample, 1).reshape(7)
                score = self.meter.update_score(self.prediction)
                if self.recording and score > self.record_lowest:
                    self.record_lowest = score
                    self.record_score.text = f"lowest score: {round(score, 2)}"
                    self.record_score.color = (round(i * 0.8) for i in self.meter.colours[round(score * 2 + 2)])

                if self.stats:
                    for i in range(7):
                        self.pred_data[i].width = 200 * self.prediction[i]

        if self.stats:
            self.face_rgb.blit(10, 10)
            self.face_mono.blit(10, 120)

        self.stats_btn.update(self.mouse_position, self.btn_pressed, self.btn_released)
        if self.stats_btn.released:
            self.toggle_stats()
        self.record_btn.update(self.mouse_position, self.btn_pressed, self.btn_released)
        if self.record_btn.released:
            self.recording = True
            self.record_lowest = -float("inf")
            self.record_tick = 0
            self.shuffle_emoji()

        now = time.perf_counter()

        self.record_btn.set_visible(not self.recording)
        if self.recording:
            self.record_tick += now - self.prev_time
            if self.record_tick < 3:
                self.record_counter.y = self.screen_size.y // 2
                self.record_counter.text = str(int(4 - self.record_tick))
                self.record_counter.font_size = 80
                self.record_score.visible = False
            elif self.record_tick < 4:
                self.record_counter.y = self.screen_size.y // 2
                self.record_counter.text = self.record_emoji
                self.record_counter.font_size = 80
                self.record_score.visible = False
            elif self.record_tick < 14:
                self.record_counter.y = 120
                self.record_score.visible = True
                self.record_counter.font_size = 30
                self.record_counter.text = str(format(14 - self.record_tick, ".2f"))
            else:
                self.recording = False

        self.prev_time = now

    def process_face(self, cropped):
        image = Image.fromarray(cropped)
        if self.stats:
            display_img = np.asarray(image.resize((100, 100)))
            self.face_rgb.set_bytes("RGB", -300, display_img.tobytes())

        mono = image.convert('L')
        if self.stats:
            display_img = np.stack([np.asarray(mono.resize((100, 100)))] * 3, axis = -1)
            self.face_mono.set_bytes("RGB", -300, display_img.tobytes())

        self.sample = (np.asarray(mono.resize((48, 48))) / 255).reshape(1, 1, 48, 48)


app = App()
app.run()