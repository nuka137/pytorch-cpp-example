import torch
import torch.nn.functional as F
from torchsummary import summary

NUMBER_OF_EPOCHS = 10
LOG_INTERVAL = 100

class YoloLoss(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, predict, target):
        


class YoloV1(torch.nn.Module):
    def __init__(self):
        super(YoloV1, self).__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),

            torch.nn.Conv2d(64, 192, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),

            torch.nn.Conv2d(192, 128, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),

            torch.nn.Conv2d(512, 256, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 256, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 256, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 256, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),

            torch.nn.Conv2d(1024, 512, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(1024, 512, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),

            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True)
        )

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(7*7*1024, 4096),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(4096, 1024)  # TODO
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = self.fc_layers(out)

        return out


def set_random_seed(seed):
    torch.manual_seed(seed)


def get_learning_rate(epoch, iteration, iteration_max):
    if epoch == 0:
        lr = 0.001 + (0.01 - 0.001) * iteration / iteration_max
    elif 1 <= epoch < 76:
        lr = 0.01
    elif 76 <= epoch < 96:
        lr = 0.001
    else:
        lr = 0.0001
    
    return lr


def main():
    set_random_seed(1)

    # Create device.
    if torch.cuda.is_available():
        device_type = "cuda"
        print("Train on GPU.")
    else:
        device_type = "cpu"
        print("Train on CPU.")
    device = torch.device(device_type)

    # Build model.
    model = YoloV1()
    model.to(device)
    summary(model, (3, 448, 448))
    optimizer = torch.optim.SGD(model.parameters(), lr=get_learning_rate(0, 0, 0),
                                momentum=0.9, weight_decay=0.0005)

    # Load dataset.
    iteration_max = len(train_loader)

    
    # Train loop.
    for epoch in range(NUMBER_OF_EPOCHS):
        print("Epoch {}: ".format(epoch))

        # Train.
        print("Start train.")
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Update learning rate.
            for group in optimizer.param_groups:
                group["lr"] = get_learning_rate(epoch, batch_idx, iteration_max)

            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss = ...
            loss.backward()
            optimizer.step()

            if batch_idx % LOG_INTERVAL == 0:
                print("Batch: {}, Loss: {}".format(batch_idx, loss.item()))
        
        # Evaluate.
        # print("Start eval.")
        # model.eval()
        # with torch.no_grad():
        #     for data, target in test_loader:
        #         data = data.to(device)
        #         target = target.to(device)

        #         output = model(data)

        #         loss = ...
                


if __name__ == "__main__":
    main()
