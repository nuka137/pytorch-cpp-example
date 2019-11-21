import torch
import torchvision
import torch.nn.functional as F

DATA_ROOT = "./mnist-data"
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
LOG_INTERVAL = 10
NUMBER_OF_EPOCHS = 10

def main():
    torch.manual_seed(1)

    # Create device.
    if torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"
    device = torch.device(device_type)

    # Build model.
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 10, (5, 5)),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),

        torch.nn.Conv2d(10, 20, (5, 5)),
        torch.nn.Dropout2d(),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),

        torch.nn.Flatten(),

        torch.nn.Linear(320, 50),
        torch.nn.ReLU(),

        torch.nn.Dropout(0.5),

        torch.nn.Linear(50, 10),

        torch.nn.LogSoftmax(1)
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))

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

    for epoch in range(NUMBER_OF_EPOCHS):
        # train
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % LOG_INTERVAL == 0:
                print("Train Epoch: {}, Loss: {}".format(batch_idx, loss.item()))

        # test
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)

                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print("Test set: Average loss: {} | Accuracy: {}".format(test_loss, correct / len(test_loader.dataset)))


if __name__ == "__main__":
    main()