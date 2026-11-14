import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference")
    parser.add_argument("--model", default="weights/best.pt")
    parser.add_argument("--source", default="media/test.png")
    args = parser.parse_args()

    if args.source == "usb":
        args.source = 0  # First connected camera

    model = YOLO(args.model)

    for result in model.predict(
        source=args.source,
        stream=True,
        show=True,
        save=True,
        project="output"
    ):
        pass

if __name__ == "__main__":
    main()