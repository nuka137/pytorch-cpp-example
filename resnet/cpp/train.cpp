#include <sys/stat.h>
#include <iostream>
#include <torch/torch.h>

#include "model.h"

const std::string kDataRoot = "./mnist";
const int64_t kTrainBatchSize = 64;
const int64_t kTestBatchSize = 1000;
const int64_t kLogInterval = 100;
const int64_t kNumberOfEpochs = 10;
const std::string kSavedModelNamePrefix = "resnet";

using namespace torch::nn;
namespace F = torch::nn::functional;


void fix_randomness(int seed) {
    torch::manual_seed(seed);
}

std::vector<std::string> convert_argv_to_strings(int argc, char** argv)
{
    std::vector<std::string> s;
    for (int i = 0; i < argc; ++i) {
        s.push_back(argv[i]);
    }

    return s;
}

void print_usage_and_exit(int exit_code) {
    std::cout << "Usage: train" << std::endl;
    std::cout << "    -m <saved_model_path>" << std::endl;
    std::cout << "    -h" << std::endl;

    exit(exit_code);
}

struct CommandArguments {
    std::string saved_model_path;
};

auto main(int argc, char* argv[]) -> int
{
    // Parse arguments.
    if (argc < 2) {
        print_usage_and_exit(1);
    }

    std::vector<std::string> argv_strings = convert_argv_to_strings(argc, argv);
    CommandArguments args;
    if (argv_strings[1] == "-m") {
        if (argc != 3) {
            print_usage_and_exit(1);
        }
        args.saved_model_path = argv_strings[2];
    } else if (argv_strings[1] == "-h") {
        print_usage_and_exit(0);
    } else {
        print_usage_and_exit(1);
    }

    fix_randomness(1);

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
    model->to(device);
    torch::optim::Adam optimizer(
        model->parameters(), torch::optim::AdamOptions(0.01));

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
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), kTestBatchSize);

    // Train loop.
    for (size_t epoch = 0; epoch < kNumberOfEpochs; ++epoch) {
        std::cout << "Epoch " << epoch << ":" << std::endl;

        // Train.
        std::cout << "Start train." << std::endl;
        size_t batch_idx = 0;
        model->train();
        for (auto& batch : *train_loader) {
            optimizer.zero_grad();
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            auto output = model->forward(data);

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
        model->eval();
        double test_loss = 0.0;
        size_t correct = 0;
        size_t total = 0;
        for (auto& batch : *test_loader) {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            auto output = model->forward(data);

            auto prob = F::log_softmax(output, 1);
            test_loss += F::nll_loss(
                prob, target,
                F::NLLLossFuncOptions().reduction(torch::kSum)).template item<double>();
            auto pred = output.argmax(1, true);
            correct += pred.eq(target.view_as(pred)).sum().template item<int64_t>();
            total += kTestBatchSize;
        }

        std::cout << "Average loss: " << test_loss / total
                  << ", Accuracy: " << static_cast<double>(correct) / total
                  << std::endl;
    }

    // Save trained model.
    std::string model_path = args.saved_model_path + "/"
        + kSavedModelNamePrefix + "_model.pth";
    std::string optimizer_path = args.saved_model_path + "/"
        + kSavedModelNamePrefix + "_optimizer.pth";
    struct stat buf;
    if (stat(args.saved_model_path.c_str(), &buf)) {
        int rc;
        rc = mkdir(args.saved_model_path.c_str(), 0755);
        if (rc < 0) {
            std::cout << "Error: Failed to create diretory '"
                      << args.saved_model_path  << "' ("
                      << errno << ": " << strerror(errno) << ")" << std::endl;
            return 1;
        }
    }
    torch::save(model, model_path);
    torch::save(optimizer, optimizer_path);
    std::cout << "Saved model." << std::endl;
    std::cout << "  Model: " << model_path << std::endl;
    std::cout << "  Optimizer: " << model_path << std::endl;

    return 0;
}
