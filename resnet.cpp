#include <iostream>
#include <torch/torch.h>

const char* kDataRoot = "./data";
const int64_t kTrainBatchSize = 64;
const int64_t kLogInterval = 10;

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
  model->to(device);

  // Load dataset.
  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
  const size_t train_dataset_size = train_dataset.size().value();
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_dataset), kTrainBatchSize);

  torch::optim::Adam optimizer(
      model->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));

  size_t batch_idx = 0;
  for (auto& batch : *train_loader) {
    optimizer.zero_grad();
    auto data = batch.data.to(device);
    auto target = batch.target.to(device);
    auto output = model->forward(data);

    auto loss = torch::nll_loss(output, target);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    batch_idx++;

    if ((batch_idx % kLogInterval) == 0) {
      std::cout << "Train Epoch: " << batch_idx << ", Loss: " << loss.template item<float>() << std::endl;
    }
  }

  return 0;
}
