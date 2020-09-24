#include <sys/stat.h>
#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "model.h"

const std::string kSavedModelNamePrefix = "resnet";

using namespace torch::nn;
namespace F = torch::nn::functional;


std::vector<std::string> convert_argv_to_strings(int argc, char** argv)
{
    std::vector<std::string> s;
    for (int i = 0; i < argc; ++i) {
        s.push_back(argv[i]);
    }

    return s;
}

void print_usage_and_exit(int exit_code) {
    std::cout << "Usage: predict" << std::endl;
    std::cout << "    -i <image_to_predict> -m <saved_model_path>" << std::endl;
    std::cout << "    -h" << std::endl;

    exit(exit_code);
}

struct CommandArguments {
    std::string image_to_predict;
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
    if (argv_strings[1] == "-i") {
        if (argc != 5) {
            print_usage_and_exit(1);
        }
        args.image_to_predict = argv_strings[2];
        if (argv_strings[3] != "-m") {
            print_usage_and_exit(1);
        }
        args.saved_model_path = argv_strings[4];
    } else if (argv_strings[1] == "-h") {
        print_usage_and_exit(0);
    } else {
        print_usage_and_exit(1);
    }

    // Load image.
    cv::Mat mat = cv::imread(args.image_to_predict, cv::IMREAD_GRAYSCALE);
    cv::resize(mat, mat, cv::Size(28, 28));

    // Pre-process.
    cv::Mat mat_float;
    mat.convertTo(mat_float, CV_32F);
    mat_float /= 255.0;

    torch::TensorOptions options(torch::kFloat32);
    torch::Tensor input_tensor = torch::from_blob(
        mat_float.data, {1, 1, 28, 28}, options);

    torch::data::transforms::Normalize<> transforms(0.1307, 0.3081);
    input_tensor = transforms(input_tensor);

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

    // Load model.
    std::string model_path = args.saved_model_path + "/"
        + kSavedModelNamePrefix + "_model.pth";
    std::string optimizer_path = args.saved_model_path + "/"
        + kSavedModelNamePrefix + "_optimizer.pth";

    ResNet50 model;
    torch::optim::Adam optimizer(
        model->parameters(), torch::optim::AdamOptions(0.01));

    torch::load(model, model_path);
    torch::load(optimizer, optimizer_path);
    model->to(device);

    // Predict.
    model->eval();
    input_tensor = input_tensor.to(device);
    auto output = model(input_tensor);
    auto pred = output.argmax(1, true);
    std::cout << "Predict: " << pred.cpu()[0][0].template item<int>()
              << std::endl;

  return 0;
}
