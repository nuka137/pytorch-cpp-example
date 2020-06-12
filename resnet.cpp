#include <iostream>
#include <torch/torch.h>

const char* kDataRoot = "./data";
const int64_t kTrainBatchSize = 64;
const int64_t kTestBatchSize = 1000;
const int64_t kLogInterval = 10;
const int64_t kNumberOfEpochs = 10;

struct ResidualBlock : torch::nn::Module {
  ResidualBlock(int in_channels, int out_channels, int stride=1)// :
    //width(out_channels / 4),
    //conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, width, {1, 1}).stride(1).bias(false))),
    //bn1(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(width))),
    //relu1(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))),
    //conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(width, width, {3, 3})
    //                        .stride(stride).padding(1).groups(1).bias(false).dilation(1))),
    //bn2(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(width))),
    //relu2(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))),
    //conv3(torch::nn::Conv2d(torch::nn::Conv2dOptions(width, out_channels, {1, 1})
    //                        .stride(1).padding(0).bias(false))),
    //bn3(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels))),
    //shortcut(
    //  torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, {1, 1})
    //                    .stride(stride).padding(0).bias(false))
    //),
    //relu3(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)))
  {
    int width = out_channels / 4;

    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, width, {1, 1})
                              .stride(1).bias(false));
    register_module("conv1", conv1);
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(width));
    register_module("bn1", bn1);
    relu1 = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true));
    register_module("relu1", relu1);

    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(width, width, {3, 3})
                              .stride(stride).padding(1).groups(1).bias(false).dilation(1));
    register_module("conv2", conv2);
    bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(width));
    register_module("bn2", bn2);
    relu2 = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true));
    register_module("relu2", relu2);

    conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(width, out_channels, {1, 1})
                              .stride(1).padding(0).bias(false));
    register_module("conv3", conv3);
    bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels));
    register_module("bn3", bn3);

    shortcut = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, {1, 1})
                                 .stride(stride).padding(0).bias(false));
    register_module("shortcut", shortcut);
    relu3 = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true));
    register_module("relu3", relu3);

    this->in_channels = in_channels;
    this->out_channels = out_channels;
  }

  torch::Tensor forward(torch::Tensor input) {
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

  torch::nn::Conv2d conv1 = nullptr;
  torch::nn::BatchNorm2d bn1 = nullptr;
  torch::nn::ReLU relu1 = nullptr;
  torch::nn::Conv2d conv2 = nullptr;
  torch::nn::BatchNorm2d bn2 = nullptr;
  torch::nn::ReLU relu2 = nullptr;
  torch::nn::Conv2d conv3 = nullptr;
  torch::nn::BatchNorm2d bn3 = nullptr;

  torch::nn::Conv2d shortcut = nullptr;
  torch::nn::ReLU relu3 = nullptr;

  int in_channels;
  int out_channels;
  //int width;
};

/*
torch::nn::Sequential make_residual_block(int in_channels, int out_channels, int stride=1) {
    int width = out_channels / 4;

    torch::nn::Sequential block;

    block->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, width, {1, 1})
                                       .stride(1).bias(false)));
    block->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(width)));
    block->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

    block->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(width, width, {3, 3})
                                       .stride(stride).padding(1).groups(1).bias(false).dilation(1)));
    block->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(width)));
    block->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

    block->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(width, out_channels, {1, 1})
                                       .stride(1).padding(0).bias(false)));
    block->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels)));

    shortcut = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, {1, 1})
                                 .stride(stride).padding(0).bias(false));
    relu3 = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true));
}
*/

struct ResNet50 : torch::nn::Module {
  ResNet50() {
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, {7, 7})
                              .stride(2).padding(3).bias(false));
    register_module("conv1", conv1);
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(64));
    register_module("bn1", bn1);
    relu = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true));
    register_module("relu", relu);
    maxpool = torch::nn::MaxPool2d(torch::nn::MaxPoolOptions<2>({3, 3})
                                   .stride(2).padding(1));
    register_module("maxpool", maxpool);

    torch::nn::Sequential layer1(
      ResidualBlock(64, 256),
      ResidualBlock(256, 256),
      ResidualBlock(256, 256)
    );
    this->layer1 = layer1;
    register_module("layer1", this->layer1);

    torch::nn::Sequential layer2(
      ResidualBlock(256, 512, 2),
      ResidualBlock(512, 512),
      ResidualBlock(512, 512),
      ResidualBlock(512, 512)
    );
    this->layer2 = layer2;
    register_module("layer2", this->layer2);

    torch::nn::Sequential layer3(
      ResidualBlock(512, 1024, 2),
      ResidualBlock(1024, 1024),
      ResidualBlock(1024, 1024),
      ResidualBlock(1024, 1024),
      ResidualBlock(1024, 1024),
      ResidualBlock(1024, 1024)
    );
    this->layer3 = layer3;
    register_module("layer3", this->layer3);

    torch::nn::Sequential layer4(
      ResidualBlock(1024, 2048, 2),
      ResidualBlock(2048, 2048),
      ResidualBlock(2048, 2048)
    );
    this->layer4 = layer4;
    register_module("layer4", this->layer4);

    avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
    register_module("avgpool", avgpool);
    flatten = torch::nn::Flatten(torch::nn::FlattenOptions());
    register_module("flatten", flatten);
    fc = torch::nn::Linear(2048, 10);
    register_module("fc", fc);
  }

  torch::Tensor forward(torch::Tensor input) {
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

  torch::nn::Conv2d conv1 = nullptr;
  torch::nn::BatchNorm2d bn1 = nullptr;
  torch::nn::ReLU relu = nullptr;
  torch::nn::MaxPool2d maxpool = nullptr;
  torch::nn::Sequential layer1 = nullptr;
  torch::nn::Sequential layer2 = nullptr;
  torch::nn::Sequential layer3 = nullptr;
  torch::nn::Sequential layer4 = nullptr;
  torch::nn::AdaptiveAvgPool2d avgpool = nullptr;
  torch::nn::Flatten flatten = nullptr;
  torch::nn::Linear fc = nullptr;
};

auto main() -> int
{
  torch::manual_seed(1);

  // Determine device on which performs training.
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "Train on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Train on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  ResNet50 model;
  model.to(device);

  // Load dataset.
  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  torch::optim::Adam optimizer(
      model.parameters(), torch::optim::AdamOptions(0.1));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    // train
    size_t batch_idx = 0;
    for (auto& batch : *train_loader) {
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);
      auto output = model.forward(data);

      auto prob = torch::log_softmax(output, 1);

      auto loss = torch::nll_loss(prob, target);
      AT_ASSERT(!std::isnan(loss.template item<float>()));
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      batch_idx++;

      if ((batch_idx % kLogInterval) == 0) {
        std::cout << "Train Epoch: " << batch_idx << ", Loss: " << loss.template item<float>() << std::endl;
      }
    }

    // test
    torch::NoGradGuard no_grad;
    model.eval();
    double test_loss = 0;
    int32_t correct = 0;
    for (auto& batch : *test_loader) {
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);
      auto output = model.forward(data);

      test_loss += torch::nll_loss(output, target, {}, at::Reduction::Sum).template item<int64_t>();
      auto pred = output.argmax(1);
      correct += pred.eq(target).sum().template item<int64_t>();
    }

    test_loss /= test_dataset_size;
    std::cout << "Test set: Average loss: " << test_loss
              << " | Accuracy: " << static_cast<double>(correct) / test_dataset_size;
  }

  return 0;
}
