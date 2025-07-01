from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy_garden.xcamera import XCamera
from kivy.graphics.texture import Texture
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model and class names
model = YOLO('yolov8n.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

class CameraDetectionApp(App):
    def build(self):
        self.layout = BoxLayout()
        self.camera = XCamera(play=True)
        self.layout.add_widget(self.camera)

        # Schedule detection
        Clock.schedule_interval(self.detect_objects, 1.0 / 10.0)  # 10 fps
        return self.layout

    def detect_objects(self, dt):
        texture = self.camera.texture
        if texture:
            buffer = texture.pixels
            size = texture.size
            # Convert the texture buffer to numpy array
            img = np.frombuffer(buffer, dtype=np.uint8)
            img = img.reshape(size[1], size[0], 4)  # RGBA
            frame = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            # Run detection
            results = model(frame)

            # Draw detections
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = classNames[cls_id]
                    confidence = box.conf[0]
                    xyxy = box.xyxy[0]  # [xmin, ymin, xmax, ymax]
                    xmin, ymin, xmax, ymax = map(int, xyxy)

                    # Draw bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    # Prepare label text with confidence
                    label_text = f"{label} {confidence:.2f}"

                    # Calculate position for label
                    label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, label_size[1] + 10)

                    # Draw label background
                    cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin), (0,255,0), cv2.FILLED)

                    # Put label text
                    cv2.putText(frame, label_text, (xmin, label_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            # Convert back to texture for display
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(size[0], size[1]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera.texture = texture

if __name__ == '__main__':
    CameraDetectionApp().run()