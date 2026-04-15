import pyglet, cv2
from PIL import Image
import numpy as np
import nn


camera = cv2.VideoCapture(0)
camera_width, camera_height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
window = pyglet.window.Window(camera_width, camera_height, "cortisol detector")

width, height = window.get_framebuffer_size()
width = camera_width ** 2 // width
height = camera_height ** 2 // height
window.set_size(width, height)

cascade = cv2.CascadeClassifier("src/app/models/haarcascade_frontalface_default.xml")
sample = None
sample_rate = 10
sample_tick = 0

model = nn.load("src/models/v1-5")
prediction = np.zeros(7)
classes = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


def process_face(cropped):
    global sample

    image = Image.fromarray(cropped)
    display_img = np.asarray(image.resize((100, 100)))
    img = pyglet.image.ImageData(100, 100, "RGB", display_img.tobytes(), pitch = -300)
    img.blit(10, 10)

    mono = image.convert('L')
    display_img = np.stack([np.asarray(mono.resize((100, 100)))] * 3, axis = -1)
    img = pyglet.image.ImageData(100, 100, "RGB", display_img.tobytes(), pitch = -300)
    img.blit(10, 120)

    sample = (np.asarray(mono.resize((48, 48))) / 255).reshape(1, 1, 48, 48)

def display_frame(frame):
    global sample_tick, prediction

    array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = pyglet.image.ImageData(camera_width, camera_height, "RGB", array.tobytes(), pitch = -camera_width * 3)
    img.blit(0, 0)

    face = cascade.detectMultiScale(array)
    if len(face) > 0:
        face = face[0]
        x1, y1, x2, y2 = face[0], camera_height - face[1], face[0] + face[2], camera_height - (face[1] + face[3])
        pyglet.shapes.Line(x1, y1, x1, y2, 5, (255, 0, 0)).draw()
        pyglet.shapes.Line(x1, y2, x2, y2, 5, (255, 0, 0)).draw()
        pyglet.shapes.Line(x2, y2, x2, y1, 5, (255, 0, 0)).draw()
        pyglet.shapes.Line(x2, y1, x1, y1, 5, (255, 0, 0)).draw()

        cropped = array[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
        if cropped.shape[0] >= 48:
            process_face(cropped)

    sample_tick += 1
    if sample_tick >= sample_rate and sample is not None:
        sample_tick = 0
        prediction = model.forward(sample, 1).reshape(7)

    for i in range(7):
        l = pyglet.text.Label(classes[i], 20, camera_height - 50 - 50 * i, font_name = "arial", font_size = 25)
        l.draw()
        b = pyglet.shapes.Rectangle(200, camera_height - 55 - 50 * i, 200 * prediction[i], 40, color = (150, 0, 0))
        b.draw()


@window.event
def on_draw():
    window.clear()
    success, frame = camera.read()
    if success:
        display_frame(frame)


pyglet.app.run(1 / 60)
camera.release()