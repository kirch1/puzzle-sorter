from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import serial

arduino = serial.Serial(port='/dev/cu.usbserial-110', baudrate=115200, timeout=.1)
model = YOLO('puzzle.pt')
results = model.predict(source=0, stream=True)

for result in results:
    if len(result.boxes) > 0:
        firstInstance = result.boxes[0]
        if firstInstance.conf > .75:
            if firstInstance.cls == 0.0:
                print('Center Present')
                arduino.write(bytes('0', 'utf-8'))
            elif firstInstance.cls == 1.0:
                print('Edge Present')
                arduino.write(bytes('1', 'utf-8'))
    else:
        print('Nothing Present')
        arduino.write(bytes('0', 'utf-8'))
