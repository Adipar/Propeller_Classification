import cv2
from roboflow import Roboflow
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


rf = Roboflow(api_key="qygFcArdX4WdqtBZrT3X")
project = rf.workspace().project("propellers-hj3hi")
model = project.version("2").model

def infer_on_image(image_path):
    result = model.predict(image_path, confidence=40).json()
    return result

def capture_and_infer():
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image")
        return

    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)

    results = infer_on_image(temp_image_path)

    detected = False
    for prediction in results['predictions']:
        detected = True
        x0 = prediction['x'] - prediction['width'] / 2
        y0 = prediction['y'] - prediction['height'] / 2
        x1 = prediction['x'] + prediction['width'] / 2
        y1 = prediction['y'] + prediction['height'] / 2

        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        cv2.putText(frame, prediction['class'], (int(x0), int(y0) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    if detected:
        messagebox.showinfo("Inference Result", "Object detected!")
    else:
        messagebox.showinfo("Inference Result", "No object detected.")

    cv2.imshow('Captured Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def quit_app():
    cap.release()
    root.destroy()

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        live_feed_label.imgtk = imgtk
        live_feed_label.configure(image=imgtk)
    root.after(10, update_frame)


cap = cv2.VideoCapture(0)


root = tk.Tk()
root.title("Object Detection")


live_feed_label = tk.Label(root)
live_feed_label.pack()


button_frame = tk.Frame(root)
button_frame.pack()

capture_button = tk.Button(button_frame, text="Capture", command=capture_and_infer)
capture_button.pack(side=tk.LEFT, padx=20, pady=20)

quit_button = tk.Button(button_frame, text="Quit", command=quit_app)
quit_button.pack(side=tk.RIGHT, padx=20, pady=20)


update_frame()


root.mainloop()


cap.release()
cv2.destroyAllWindows()
