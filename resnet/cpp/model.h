#pragma once

#include <torch/torch.h>

using namespace torch::nn;


struct ResidualBlockImpl : Module {
    ResidualBlockImpl(int in_channels, int out_channels, int stride=1);

    torch::Tensor forward(torch::Tensor input);

private:

    Conv2d conv1 = nullptr;
    BatchNorm2d bn1 = nullptr;
    ReLU relu1 = nullptr;
    Conv2d conv2 = nullptr;
    BatchNorm2d bn2 = nullptr;
    ReLU relu2 = nullptr;
    Conv2d conv3 = nullptr;
    BatchNorm2d bn3 = nullptr;

    Sequential shortcut = nullptr;
    ReLU relu3 = nullptr;

    int in_channels;
    int out_channels;
};
TORCH_MODULE(ResidualBlock);


struct ResNet50Impl : Module {
    ResNet50Impl();

    torch::Tensor forward(torch::Tensor input);

private:

    Conv2d conv1 = nullptr;
    BatchNorm2d bn1 = nullptr;
    ReLU relu = nullptr;
    MaxPool2d maxpool = nullptr;
    Sequential layer1 = nullptr;
    Sequential layer2 = nullptr;
    Sequential layer3 = nullptr;
    Sequential layer4 = nullptr;
    AdaptiveAvgPool2d avgpool = nullptr;
    Flatten flatten = nullptr;
    Linear fc = nullptr;
};
TORCH_MODULE(ResNet50);


