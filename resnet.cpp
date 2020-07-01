#include <iostream>
#include <torch/torch.h>

const char* kDataRoot = "./mnist";
const int64_t kTrainBatchSize = 64;
const int64_t kTestBatchSize = 1000;
const int64_t kLogInterval = 100;
const int64_t kNumberOfEpochs = 10;

using namespace torch::nn;
namespace F = torch::nn::functional;


struct ResidualBlock : Module {
  ResidualBlock(int in_channels, int out_channels, int stride=1) {
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


struct ResNet50 : Module {
  ResNet50() {
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

auto main() -> int
{
  torch::manual_seed(1);

  // Create device.
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "Train on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Train on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  // Build model.
  ResNet50 model;
  model.to(device);
  torch::optim::Adam optimizer(
      model.parameters(), torch::optim::AdamOptions(0.01));

  // Load dataset.
  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  // Train loop.
  for (size_t epoch = 0; epoch < kNumberOfEpochs; ++epoch) {
    std::cout << "Epoch " << epoch << ":" << std::endl;

    // Train.
    std::cout << "Start train." << std::endl;
    size_t batch_idx = 0;
    model.train();
    for (auto& batch : *train_loader) {
      optimizer.zero_grad();
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);

      auto output = model.forward(data);

      auto prob = F::log_softmax(output, 1);
      auto loss = F::nll_loss(prob, target);
      AT_ASSERT(!std::isnan(loss.template item<float>()));
      loss.backward();
      optimizer.step();

      if ((batch_idx % kLogInterval) == 0) {
        std::cout << "Batch: " << batch_idx << ", Loss: "
                  << loss.template item<float>() << std::endl;
      }
      batch_idx++;
    }

    // Evaluate.
    std::cout << "Start eval." << std::endl;
    torch::NoGradGuard no_grad;
    model.eval();
    double test_loss = 0.0;
    size_t correct = 0;
    size_t total = 0;
    for (auto& batch : *test_loader) {
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);
      auto output = model.forward(data);

      auto prob = F::log_softmax(output, 1);
      test_loss += F::nll_loss(prob, target, F::NLLLossFuncOptions().reduction(torch::kSum)).template item<double>();
      auto pred = output.argmax(1, true);
      correct += pred.eq(target.view_as(pred)).sum().template item<int64_t>();
      total += kTestBatchSize;
    }

    std::cout << "Average loss: " << test_loss / total
	      << ", Accuracy: " << static_cast<double>(correct) / total << std::endl;
  }

  return 0;
}
