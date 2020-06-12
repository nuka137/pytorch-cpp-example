#include <iostream>
#include <torch/torch.h>

const char* kDataRoot = "./data";
const int64_t kTrainBatchSize = 64;
const int64_t kTestBatchSize = 1000;
const int64_t kLogInterval = 10;
const int64_t kNumberOfEpochs = 10;

struct ResidualBlock : torch::nn::Module {
  ResidualBlock(int in_channels, int out_channels, int stride=1) {
    int width = out_channels / 4;

    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, width, {1, 1})
                              .stride(1).bias(false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(width));
    relu1 = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true));

    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(width, width, {3, 3})
                              .stride(stride).padding(1).groups(1).bias(False).dilation(1));
    bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(width));
    relu2 = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true));

    conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(width, out_channels, {1, 1})
                              .stride(1).padding(0).bias(False));
    bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNromOptions(out_channels));

    shortcut = torch::nn::Sequential(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, {1, 1})
                        .stride(stride).padding(0).bias(False));
    );
    relu3 = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true));

    this.in_channels = in_channels;
    this.out_channels = out_channels;
  }

  torch::tensor forward(torch::tensor input) {
    torch::tensor out;

    out = conv1(input);
    out = bn1(out);
    out = relu1(out);

    out = conv2(out);
    out = bn2(out);
    out = relu2(out);

    out = conv3(out);
    out = bn3(out);

    if (in_channels != out_channels) {
      shortcut = shortcut(input);
    } else {
      shortcut = input;
    }
    out = relu3(out + shortcut);

    return out;
  }

  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm2d bn1;
  torch::nn::ReLU relu1;
  torch::nn::Conv2d conv2;
  torch::nn::BatchNorm2d bn2;
  torch::nn::ReLU relu2;
  torch::nn::Conv2d conv3;
  torch::nn::BatchNorm2d bn3;

  int in_channels;
  int out_channels;
}

struct ResNet50 : torch::nn::Module {
  ResNet50() {
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, {7, 7})
                              .stride(2).padding(3).bias(false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(64));
    relu = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true));
    maxpool = torch::nn::MaxPool2d(torch::nn::MaxPoolOptions<2>({3, 3})
                                   .stride(2).padding(1));

    layer1 = torch::nn::Sequential(
      ResidualBlock(64, 256),
      ResidualBlock(256, 256),
      ResidualBlock(256, 256)
    );

    layer2 = torch::nn::Sequential(
      ResidualBlock(256, 512, stride=2),
      ResidualBlock(512, 512),
      ResidualBlock(512, 512),
      ResidualBlock(512, 512)
    );

    layer3 = torch::nn::Sequential(
      ResidualBlock(512, 1024, stride=2),
      ResidualBlock(1024, 1024),
      ResidualBlock(1024, 1024),
      ResidualBlock(1024, 1024),
      ResidualBlock(1024, 1024),
      ResidualBlock(1024, 1024)
    );

    layer4 = torch::nn::Sequential(
      ResidualBlock(1024, 2048, stride=2),
      ResidualBlock(2048, 2048),
      ResidualBlock(2048, 2048)
    );

    avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
    flatten = torch::nn::Flatten(torch::nn::FlattenOptions());
    fc = torch::nn::Linear(2048, 10);
  }

  torch::Tensor forward(torch::Tensor input) {
    torch::Tensor out;

    out = conv1(input);
    out = bn1(out);
    out = relu(out);
    out = maxpool(out);

    out = layer1(out);
    out = layer2(out);
    out = layer3(out);
    out = layer4(out);

    out = avgpool(out);
    out = flatten(out);
    out = fc(out);
  }

  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm2d bn1;
  torch::nn::ReLU relu;
  torch::nn::Sequential layer1;
  torch::nn::Sequential layer2;
  torch::nn::Sequential layer3;
  torch::nn::Sequential layer4;
  torch::nn::AdaptiveAvgPool2d avgpool;
  torch::nn::Linear fc;
}

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

  /*
  torch::nn::Sequential model(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, {5, 5})),
      torch::nn::MaxPool2d(2),
      torch::nn::ReLU(),
      
      torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, {5, 5})),
      torch::nn::Dropout2d(),
      torch::nn::MaxPool2d(2),
      torch::nn::ReLU(),

      torch::nn::Flatten(torch::nn::FlattenOptions()),

      torch::nn::Linear(320, 50),
      torch::nn::ReLU(),

      torch::nn::Dropout(0.5),

      torch::nn::Linear(50, 10),

      torch::nn::LogSoftmax(1)
      );
      */
  ResNet50 model;
  model->to(device);

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
      model->parameters(), torch::optim::AdamOptions(0.1).betas(std::make_tuple(0.9, 0.999)));

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    // train
    size_t batch_idx = 0;
    for (auto& batch : *train_loader) {
      optimizer.zero_grad();
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);
      auto output = model->forward(data);

      auto prob = torch::log_softmax(output, torch::nn::SoftmaxOptions().dim(1));

      auto loss = torch::nll_loss(prob, target);
      AT_ASSERT(!std::isnan(loss.template item<float>()));
      loss.backward();
      optimizer.step();

      batch_idx++;

      if ((batch_idx % kLogInterval) == 0) {
        std::cout << "Train Epoch: " << batch_idx << ", Loss: " << loss.template item<float>() << std::endl;
      }
    }

    // test
    torch::NoGradGuard no_grad;
    model->eval();
    double test_loss = 0;
    int32_t correct = 0;
    for (auto& batch : *test_loader) {
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);
      auto output = model->forward(data);

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
