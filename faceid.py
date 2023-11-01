# Import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


class CamApp(App):
    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, 0.1))
        self.verification_label = Label(
            text="Verification Uninitiated", size_hint=(1, 0.1)
        )

        # Add items to layout
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load TensorFlow/Keras model
        self.model = tf.keras.models.load_model(
            "siamesemodel.h5", custom_objects={"L1Dist": L1Dist}
        )

        # Setup video capture device
        self.capture = cv2.VideoCapture(4)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    def update(self, *args):
        """
        Continuously get webcam feed
        """
        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120 : 120 + 250, 200 : 200 + 250, :]

        # Flip horizontally and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
        )
        img_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.web_cam.texture = img_texture

    def preprocess(self, file_path):
        """
        Load image from file and convert to 100x100px
        """
        # Read in image from file path
        byte_img = tf.io

        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        img = tf.image.decode_image(byte_img, channels=3)

        # Resize the image to 100x100 pixels
        img = tf.image.resize(img, (100, 100))

        # Convert to a NumPy array
        img = img.numpy()

        return img

    def verify(self, instance):
        """
        Verify the captured image using the loaded model
        """
        # Define the path to the reference image
        reference_image_path = "reference_image.jpg"

        # Load and preprocess the reference image
        reference_img = self.preprocess(reference_image_path)

        # Capture the current image from the camera and preprocess it
        current_img = self.capture_current_image()

        if reference_img is not None and current_img is not None:
            # Use the model to compute the L1 distance between the images
            distance = self.model.predict(
                [
                    np.expand_dims(reference_img, axis=0),
                    np.expand_dims(current_img, axis=0),
                ]
            )[0][0]

            # You can set a threshold for the distance and decide if it's a match
            threshold = 0.5  # Adjust this threshold as needed
            if distance < threshold:
                self.verification_label.text = "Verification Successful"
            else:
                self.verification_label.text = "Verification Failed"
        else:
            self.verification_label.text = "Image Capture Failed"

    def capture_current_image(self):
        """
        Capture and preprocess the current image from the camera
        """
        ret, frame = self.capture.read()
        frame = frame[120 : 120 + 250, 200 : 200 + 250, :]

        if ret:
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (100, 100))
            frame = frame / 255.0  # Normalize pixel values

            return frame
        else:
            return None

    def on_stop(self):
        """
        Release the video capture when the app is closed
        """
        self.capture.release()


if __name__ == "__main__":
    CamApp().run()
