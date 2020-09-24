import argparse
import os
import random
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torchsummary import summary

from model import ResNet50

DATA_ROOT = "./mnist"
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LOG_INTERVAL = 100
NUMBER_OF_EPOCHS = 10
SAVED_MODEL_NAME = "resnet.pth"


def fix_randomness(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", dest="saved_model_path", type=str,
        help="Path to saved model", required=True)
    args = parser.parse_args()

    fix_randomness(1)

    # Create device.
    if torch.cuda.is_available():
        device_type = "cuda"
        print("Train on GPU.")
    else:
        device_type = "cpu"
        print("Train on CPU.")
    device = torch.device(device_type)

    # Build model.
    model = ResNet50()
    model.to(device)
    summary(model, (1, 28, 28))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Load dataset.
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                   ])
        ),
        batch_size=TRAIN_BATCH_SIZE,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                   ])
        ),
        batch_size=TEST_BATCH_SIZE,
    )

    # Train loop.
    for epoch in range(NUMBER_OF_EPOCHS):
        print("Epoch {}:".format(epoch))

        # Train.
        print("Start train.")
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            prob = F.log_softmax(output, dim=1)
            loss = F.nll_loss(prob, target)
            loss.backward()
            optimizer.step()

            if batch_idx % LOG_INTERVAL == 0:
                print("Batch: {}, Loss: {}".format(batch_idx, loss.item()))

        # Evaluate.
        print("Start eval.")
        model.eval()
        test_loss = 0
        correct = 0.0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)

                prob = F.log_softmax(output, dim=1)
                test_loss += F.nll_loss(prob, target, reduction="sum").item()
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += TEST_BATCH_SIZE

        print("Average loss: {}, Accuracy: {}"
              .format(test_loss / loss, correct / total))

    # Save trained model.
    os.makedirs(args.saved_model_path, exist_ok=True)
    model_path = "{}/{}".format(args.saved_model_path, SAVED_MODEL_NAME)
    torch.save(model.state_dict(), model_path)
    print("Saved model to '{}'".format(model_path))


if __name__ == "__main__":
    main()
