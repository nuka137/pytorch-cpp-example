#include <torch/torch.h>

#include "model.h"

using namespace torch::nn;


ResidualBlockImpl::ResidualBlockImpl(int in_channels, int out_channels,
                                     int stride) {
    int width = out_channels / 4;

    conv1 = Conv2d(Conv2dOptions(in_channels, width, {1, 1})
                   .stride(1).bias(false));
    bn1 = BatchNorm2d(BatchNormOptions(width));
    relu1 = ReLU(ReLUOptions().inplace(true));

    conv2 = Conv2d(Conv2dOptions(width, width, {3, 3})
                   .stride(stride).padding(1).groups(1)
                   .bias(false).dilation(1));
    bn2 = BatchNorm2d(BatchNormOptions(width));
    relu2 = ReLU(ReLUOptions().inplace(true));

    conv3 = Conv2d(Conv2dOptions(width, out_channels, {1, 1})
                   .stride(1).padding(0).bias(false));
    bn3 = BatchNorm2d(BatchNormOptions(out_channels));

    Sequential shortcut(
        Conv2d(Conv2dOptions(in_channels, out_channels, {1, 1})
               .stride(stride).padding(0).bias(false)),
        BatchNorm2d(BatchNormOptions(out_channels))
    );
    this->shortcut = shortcut;
    relu3 = ReLU(ReLUOptions().inplace(true));

    this->in_channels = in_channels;
    this->out_channels = out_channels;

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("relu1", relu1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("relu2", relu2);
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    register_module("shortcut", shortcut);
    register_module("relu3", relu3);
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor input) {
    torch::Tensor out;
    torch::Tensor tmp;

    out = conv1->forward(input);
    out = bn1->forward(out);
    out = relu1->forward(out);

    out = conv2->forward(out);
    out = bn2->forward(out);
    out = relu2->forward(out);

    out = conv3->forward(out);
    out = bn3->forward(out);

    if (in_channels != out_channels) {
        tmp = shortcut->forward(input);
    } else {
        tmp = input;
    }
    out = relu3->forward(out + tmp);

    return out;
}

ResNet50Impl::ResNet50Impl() {
    conv1 = Conv2d(Conv2dOptions(1, 64, {7, 7})
                   .stride(2).padding(3).bias(false));
    bn1 = BatchNorm2d(BatchNormOptions(64));
    relu = ReLU(ReLUOptions().inplace(true));
    maxpool = MaxPool2d(MaxPoolOptions<2>({3, 3}).stride(2).padding(1));

    Sequential layer1(
        ResidualBlock(64, 256),
        ResidualBlock(256, 256),
        ResidualBlock(256, 256)
    );
    this->layer1 = layer1;

    Sequential layer2(
        ResidualBlock(256, 512, 2),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512)
    );
    this->layer2 = layer2;

    Sequential layer3(
        ResidualBlock(512, 1024, 2),
        ResidualBlock(1024, 1024),
        ResidualBlock(1024, 1024),
        ResidualBlock(1024, 1024),
        ResidualBlock(1024, 1024),
        ResidualBlock(1024, 1024)
    );
    this->layer3 = layer3;

    Sequential layer4(
        ResidualBlock(1024, 2048, 2),
        ResidualBlock(2048, 2048),
        ResidualBlock(2048, 2048)
    );
    this->layer4 = layer4;

    avgpool = AdaptiveAvgPool2d(AdaptiveAvgPool2dOptions({1, 1}));
    flatten = Flatten(FlattenOptions().start_dim(1));
    fc = Linear(2048, 10);

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("relu", relu);
    register_module("maxpool", maxpool);
    register_module("layer1", this->layer1);
    register_module("layer2", this->layer2);
    register_module("layer3", this->layer3);
    register_module("layer4", this->layer4);
    register_module("avgpool", avgpool);
    register_module("flatten", flatten);
    register_module("fc", fc);
}

torch::Tensor ResNet50Impl::forward(torch::Tensor input) {
    torch::Tensor out;

    out = conv1->forward(input);
    out = bn1->forward(out);
    out = relu->forward(out);
    out = maxpool->forward(out);

    out = layer1->forward(out);
    out = layer2->forward(out);
    out = layer3->forward(out);
    out = layer4->forward(out);

    out = avgpool->forward(out);
    out = flatten->forward(out);
    out = fc->forward(out);

    return out;
}

