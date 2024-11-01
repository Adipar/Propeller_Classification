from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("C:/Users/aadip/OneDrive/Documents/autonomous drones/pt file/Propellers final/best.pt")


def capture_image():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Press Space to Capture, Esc to Exit', frame)
        key = cv2.waitKey(1)
        if key % 256 == 27:
            print("Escape hit, closing...")
            break
        elif key % 256 == 32:
            img_path = "captured_image.jpg"
            cv2.imwrite(img_path, frame)
            print(f"Image saved at {img_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

    return img_path


def run_inference(img_path):
    img = cv2.imread(img_path)
    results = model(img)

    for result in results:
        annotated_img = result.plot()

        cv2.imshow('YOLOv8 Detection', annotated_img)
        cv2.waitKey(0)


    results.save("C:/Users/aadip/OneDrive/Documents/jupyterfiles/propellers-2/results/")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    captured_img_path = capture_image()
    run_inference(captured_img_path)
