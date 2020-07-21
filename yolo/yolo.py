import torch
import torch.nn.functional as F
from torchsummary import summary


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



if __name__ == "__main__":
    main()
