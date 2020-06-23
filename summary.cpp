#include <iostream>
#include <torch/torch.h>
#include <torch/hooks.h>

using namespace torch::nn;

const char* kDataRoot = "./data";
const int64_t kTrainBatchSize = 64;
const int64_t kTestBatchSize = 1000;
const int64_t kLogInterval = 10;
const int64_t kNumberOfEpochs = 10;


void register_hook(Module* module) {
}


void summary_string(Module* model, const std::vector<int64_t>& input_size) {

//  model->apply(
}

void summary(Module* model, const std::vector<int64_t>& input_size) {
  summary_string(model, input_size);
}


struct SimpleNet : Module {
  SimpleNet() {
    flatten = Flatten(FlattenOptions().start_dim(1));
    fc1 = Linear(28*28, 256);
    fc2 = Linear(256, 10);

    register_module("flatten", flatten);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor input) {
    torch::Tensor out;

    out = flatten->forward(input);
    out = fc1->forward(out);
    out = fc2->forward(out);

    return out;
  }

  Flatten flatten = nullptr;
  Linear fc1 = nullptr;
  Linear fc2 = nullptr;
};

std::map<int64_t, torch::utils::hooks::RemovableHandle*> handles;

auto main() -> int {
  torch::manual_seed(1);

  torch::utils::hooks::RemovableHandle handle(&handles);

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

  SimpleNet model;
  model.to(device);
  torch::optim::Adam optimizer(
      model.parameters(), torch::optim::AdamOptions(0.1));

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

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    // Train.
    std::cout << "Start train." << std::endl;
    size_t batch_idx = 0;
    model.train();
    for (auto& batch : *train_loader) {
      optimizer.zero_grad();
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);

      auto output = model.forward(data);

      auto prob = torch::log_softmax(output, 1);
      auto loss = torch::nll_loss(prob, target);
      AT_ASSERT(!std::isnan(loss.template item<float>()));
      loss.backward();
      optimizer.step();

      batch_idx++;
      if ((batch_idx % kLogInterval) == 0) {
        std::cout << "Train Epoch: " << batch_idx << ", Loss: "
                  << loss.template item<float>() << std::endl;
      }
    }

    // Evaluate.
    std::cout << "Start eval." << std::endl;
    torch::NoGradGuard no_grad;
    model.eval();
    double test_loss = 0;
    int32_t correct = 0;
    for (auto& batch : *test_loader) {
      auto data = batch.data.to(device);
      auto target = batch.target.to(device);
      auto output = model.forward(data);

      auto prob = torch::log_softmax(output, 1);

      test_loss += torch::nll_loss(prob, target, {}, at::Reduction::Sum).template item<double>();
      //auto loss = torch::nll_loss(prob, target, {}, at::Reduction::Sum);
      auto pred = output.argmax(1, true);
      correct += pred.eq(target).sum().template item<int64_t>();
    }

    test_loss /= test_dataset_size;
    std::cout << "Test set: Average loss: " << test_loss
              << " | Accuracy: "
              << static_cast<double>(correct) / test_dataset_size;
  }



  return 0;
}
