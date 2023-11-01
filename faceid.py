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
        reference_image_path = 'reference_image.jpg'
        
        # Load and preprocess the reference image
        reference_img = self.preprocess(reference_image_path)
        
        # Capture the current image from the camera and preprocess it
        current_img = self.capture_current_image()
        
        if reference_img is not None and current_img is not None:
            # Use the model to compute the L1 distance between the images
            distance = self.model.predict([np.expand_dims(reference_img, axis=0), np.expand_dims(current_img, axis=0)])[0][0]
            
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
        frame = frame[120:120+250, 200:200+250, :]

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

if __name__ == '__main__':
    CamApp().run()
