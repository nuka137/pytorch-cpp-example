import argparse
import torch
import torchvision
import numpy as np
import cv2

from model import ResNet50

SAVED_MODEL_NAME = "resnet.pth"


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", dest="image_file_to_predict", type=str,
        help="Image file to predict", required=True)
    parser.add_argument(
        "-m", dest="saved_model_path", type=str,
        help="Path to saved model", required=True)
    args = parser.parse_args()

    # Load image.
    img = cv2.imread(args.image_file_to_predict, cv2.IMREAD_GRAYSCALE)

    # Pre-process.
    img = cv2.resize(img, (28, 28))
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ])
    input_tensor = transform(img).unsqueeze(0)

    # Create device.
    if torch.cuda.is_available():
        device_type = "cuda"
        print("Predict on GPU.")
    else:
        device_type = "cpu"
        print("Predict on CPU.")
    device = torch.device(device_type)

    # Load model.
    model_path = "{}/{}".format(args.saved_model_path, SAVED_MODEL_NAME)
    model = ResNet50()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Predict.
    model.eval()
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    pred = output.argmax(1, keepdim=True)
    print("Predict: {}".format(pred.cpu()[0][0]))
 

if __name__ == "__main__":
    main()
