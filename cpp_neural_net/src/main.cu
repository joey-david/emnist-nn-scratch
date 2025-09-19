#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

inline void checkCuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error: " << cudaGetErrorString(result) << " at " << file << ":" << line
            << " in " << func;
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(val) checkCuda((val), #val, __FILE__, __LINE__)

// ----- Device kernels ----- //
__global__ void linear_relu_forward(const float* weights, const float* bias, const float* input,
                                    float* output, int out_dim, int in_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) {
        return;
    }
    float sum = bias[row];
    for (int col = 0; col < in_dim; ++col) {
        sum += weights[row * in_dim + col] * input[col];
    }
    output[row] = sum > 0.0f ? sum : 0.0f;
}

__global__ void linear_forward(const float* weights, const float* bias, const float* input,
                               float* output, int out_dim, int in_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) {
        return;
    }
    float sum = bias[row];
    for (int col = 0; col < in_dim; ++col) {
        sum += weights[row * in_dim + col] * input[col];
    }
    output[row] = sum;
}

__global__ void softmax_kernel(const float* logits, float* probs, int length) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float max_val = logits[0];
        for (int i = 1; i < length; ++i) {
            max_val = fmaxf(max_val, logits[i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < length; ++i) {
            float ex = expf(logits[i] - max_val);
            probs[i] = ex;
            sum += ex;
        }
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < length; ++i) {
            probs[i] *= inv_sum;
        }
    }
}

__global__ void compute_output_gradient(const float* probs, int target_idx, float* grad, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }
    float value = probs[idx];
    if (idx == target_idx) {
        value -= 1.0f;
    }
    grad[idx] = value;
}

__global__ void matvec_transposed(const float* weights, const float* vec, float* result,
                                  int out_dim, int in_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_dim) {
        return;
    }
    float sum = 0.0f;
    for (int row = 0; row < out_dim; ++row) {
        sum += weights[row * in_dim + idx] * vec[row];
    }
    result[idx] = sum;
}

__global__ void relu_backward_inplace(const float* activations, float* grad, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }
    grad[idx] = activations[idx] > 0.0f ? grad[idx] : 0.0f;
}

__global__ void update_weights(float* weights, const float* grad, const float* prev_activations,
                               float learning_rate, int out_dim, int in_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_dim * in_dim;
    if (idx >= total) {
        return;
    }
    int row = idx / in_dim;
    int col = idx % in_dim;
    weights[idx] -= learning_rate * grad[row] * prev_activations[col];
}

__global__ void update_biases(float* bias, const float* grad, float learning_rate, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }
    bias[idx] -= learning_rate * grad[idx];
}

// ----- Host-side helpers ----- //
struct Dataset {
    int num_samples = 0;
    int feature_dim = 0;
    std::vector<float> features;  // Row-major layout (sample-major)
    std::vector<int> labels;
    std::vector<std::string> label_names;
    std::vector<float> mean;
    std::vector<float> stddev;
};

struct TrainingHistory {
    std::vector<float> losses;
    std::vector<float> accuracies;
};

std::string trim(const std::string& input) {
    const std::string whitespace = " \t\n\r";
    const auto start = input.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        return "";
    }
    const auto end = input.find_last_not_of(whitespace);
    return input.substr(start, end - start + 1);
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::string current;
    bool in_quotes = false;
    for (char ch : line) {
        if (ch == '"') {
            in_quotes = !in_quotes;
        } else if (ch == ',' && !in_quotes) {
            tokens.push_back(trim(current));
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    tokens.push_back(trim(current));
    return tokens;
}

Dataset load_music_dataset(const std::filesystem::path& csv_path) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open dataset: " + csv_path.string());
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        throw std::runtime_error("Dataset is empty: " + csv_path.string());
    }

    std::vector<std::string> columns = split_csv_line(header_line);
    const std::unordered_set<std::string> ignored = {"filename", "length", "label"};

    std::vector<int> feature_indices;
    int label_index = -1;
    for (int idx = 0; idx < static_cast<int>(columns.size()); ++idx) {
        const std::string& column = columns[idx];
        if (column == "label") {
            label_index = idx;
        }
        if (!ignored.count(column)) {
            feature_indices.push_back(idx);
        }
    }

    if (label_index < 0) {
        throw std::runtime_error("CSV missing 'label' column");
    }
    if (feature_indices.empty()) {
        throw std::runtime_error("No feature columns detected in dataset");
    }

    std::vector<float> features;
    std::vector<std::string> raw_labels;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> tokens = split_csv_line(line);
        if (tokens.size() != columns.size()) {
            std::cerr << "Skipping malformed row: " << line << '\n';
            continue;
        }
        const std::string& label = tokens[label_index];
        if (label.empty()) {
            continue;  // Skip unlabeled rows
        }
        std::vector<float> row_features;
        row_features.reserve(feature_indices.size());
        bool row_ok = true;
        for (int feature_idx : feature_indices) {
            try {
                row_features.push_back(std::stof(tokens[feature_idx]));
            } catch (const std::exception&) {
                std::cerr << "Skipping row with invalid numeric value: " << line << '\n';
                row_ok = false;
                break;
            }
        }
        if (!row_ok) {
            continue;
        }
        features.insert(features.end(), row_features.begin(), row_features.end());
        raw_labels.push_back(label);
    }

    if (raw_labels.empty()) {
        throw std::runtime_error("Dataset contains no labeled samples");
    }

    const int feature_dim = static_cast<int>(feature_indices.size());
    const int num_samples = static_cast<int>(raw_labels.size());
    if (static_cast<int>(features.size()) != num_samples * feature_dim) {
        throw std::runtime_error("Feature count does not match expected dimensions");
    }

    // Mirror the Python pipeline: labels are assigned after alphabetical sort
    std::vector<std::string> unique_labels = raw_labels;
    std::sort(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(std::unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());

    std::unordered_map<std::string, int> label_to_id;
    for (int idx = 0; idx < static_cast<int>(unique_labels.size()); ++idx) {
        label_to_id.emplace(unique_labels[idx], idx);
    }

    std::vector<int> label_ids(raw_labels.size());
    for (std::size_t i = 0; i < raw_labels.size(); ++i) {
        label_ids[i] = label_to_id.at(raw_labels[i]);
    }

    Dataset dataset;
    dataset.num_samples = num_samples;
    dataset.feature_dim = feature_dim;
    dataset.features = std::move(features);
    dataset.labels = std::move(label_ids);
    dataset.label_names = std::move(unique_labels);
    dataset.mean.assign(feature_dim, 0.0f);
    dataset.stddev.assign(feature_dim, 1.0f);
    return dataset;
}

void normalize_dataset(Dataset& dataset) {
    const int samples = dataset.num_samples;
    const int dims = dataset.feature_dim;
    if (samples == 0 || dims == 0) {
        return;
    }
    std::vector<double> mean_acc(dims, 0.0);
    std::vector<double> var_acc(dims, 0.0);
    for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < dims; ++j) {
            mean_acc[j] += dataset.features[i * dims + j];
        }
    }
    for (int j = 0; j < dims; ++j) {
        mean_acc[j] /= static_cast<double>(samples);
        dataset.mean[j] = static_cast<float>(mean_acc[j]);
    }
    for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < dims; ++j) {
            const double diff = dataset.features[i * dims + j] - mean_acc[j];
            var_acc[j] += diff * diff;
        }
    }
    constexpr double epsilon = 1e-8;
    for (int j = 0; j < dims; ++j) {
        var_acc[j] /= static_cast<double>(samples);
        dataset.stddev[j] = static_cast<float>(std::sqrt(var_acc[j]) + epsilon);
    }
    for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < dims; ++j) {
            float& value = dataset.features[i * dims + j];
            value = (value - dataset.mean[j]) / dataset.stddev[j];
        }
    }
}

class NeuralNetCuda {
public:
    NeuralNetCuda(int input_size, int hidden1, int hidden2, int output_size, float learning_rate, int seed)
        : input_size_(input_size), hidden1_(hidden1), hidden2_(hidden2), output_size_(output_size),
          learning_rate_(learning_rate), rng_(seed) {
        if (input_size_ <= 0 || hidden1_ <= 0 || hidden2_ <= 0 || output_size_ <= 0) {
            throw std::invalid_argument("All layer sizes must be positive");
        }
        allocate_device_memory();
        initialize_parameters();
    }

    ~NeuralNetCuda() {
        free_device_memory();
    }

    TrainingHistory train(const float* d_inputs, const std::vector<int>& labels, int num_samples, int epochs) {
        if (num_samples <= 0) {
            throw std::invalid_argument("Training requested with zero samples");
        }
        if (epochs <= 0) {
            throw std::invalid_argument("Epoch count must be positive");
        }
        TrainingHistory history;
        history.losses.reserve(epochs);
        history.accuracies.reserve(epochs);

        std::vector<float> host_probs(output_size_);
        std::vector<int> indices(num_samples);
        std::iota(indices.begin(), indices.end(), 0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng_);
            double epoch_loss = 0.0;
            int correct = 0;

            for (int idx : indices) {
                const float* sample_ptr = d_inputs + static_cast<long long>(idx) * input_size_;
                forward(sample_ptr);

                CUDA_CHECK(cudaMemcpy(host_probs.data(), d_probs_, output_size_ * sizeof(float), cudaMemcpyDeviceToHost));

                const int target = labels[idx];
                const float epsilon = 1e-12f;
                const float prob_target = std::max(host_probs[target], epsilon);
                epoch_loss += -std::log(prob_target);

                const int prediction = static_cast<int>(std::distance(host_probs.begin(),
                    std::max_element(host_probs.begin(), host_probs.end())));
                if (prediction == target) {
                    ++correct;
                }

                backward(sample_ptr, target);
            }

            const float avg_loss = static_cast<float>(epoch_loss / static_cast<double>(num_samples));
            const float accuracy = static_cast<float>(correct) / static_cast<float>(num_samples);
            history.losses.push_back(avg_loss);
            history.accuracies.push_back(accuracy);
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " - loss: " << avg_loss
                      << ", accuracy: " << accuracy * 100.0f << "%" << std::endl;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        return history;
    }

    void predict_device_sample(const float* d_input, std::vector<float>& host_probs) {
        if (static_cast<int>(host_probs.size()) != output_size_) {
            host_probs.resize(output_size_);
        }
        forward(d_input);
        CUDA_CHECK(cudaMemcpy(host_probs.data(), d_probs_, output_size_ * sizeof(float), cudaMemcpyDeviceToHost));
    }

    std::vector<float> predict_host_vector(const std::vector<float>& normalized_input) {
        if (static_cast<int>(normalized_input.size()) != input_size_) {
            throw std::invalid_argument("Input vector size does not match network input");
        }
        float* d_input = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, input_size_ * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, normalized_input.data(), input_size_ * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float> probs(output_size_);
        predict_device_sample(d_input, probs);
        CUDA_CHECK(cudaFree(d_input));
        return probs;
    }

private:
    int input_size_;
    int hidden1_;
    int hidden2_;
    int output_size_;
    float learning_rate_;
    std::mt19937 rng_;

    float* d_W0_ = nullptr;
    float* d_W1_ = nullptr;
    float* d_W2_ = nullptr;
    float* d_b0_ = nullptr;
    float* d_b1_ = nullptr;
    float* d_b2_ = nullptr;
    float* d_hidden1_ = nullptr;
    float* d_hidden2_ = nullptr;
    float* d_logits_ = nullptr;
    float* d_probs_ = nullptr;
    float* d_grad_output_ = nullptr;
    float* d_grad_hidden1_ = nullptr;
    float* d_grad_hidden2_ = nullptr;

    void allocate_device_memory() {
        CUDA_CHECK(cudaMalloc(&d_W0_, hidden1_ * input_size_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W1_, hidden2_ * hidden1_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W2_, output_size_ * hidden2_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b0_, hidden1_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b1_, hidden2_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b2_, output_size_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden1_, hidden1_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden2_, hidden2_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_logits_, output_size_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_probs_, output_size_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_output_, output_size_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_hidden1_, hidden1_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_hidden2_, hidden2_ * sizeof(float)));
    }

    void free_device_memory() {
        cudaFree(d_W0_);
        cudaFree(d_W1_);
        cudaFree(d_W2_);
        cudaFree(d_b0_);
        cudaFree(d_b1_);
        cudaFree(d_b2_);
        cudaFree(d_hidden1_);
        cudaFree(d_hidden2_);
        cudaFree(d_logits_);
        cudaFree(d_probs_);
        cudaFree(d_grad_output_);
        cudaFree(d_grad_hidden1_);
        cudaFree(d_grad_hidden2_);
    }

    void initialize_parameters() {
        std::normal_distribution<float> dist(0.0f, 1.0f);

        auto init_matrix = [&](float* device_ptr, int rows, int cols) {
            std::vector<float> host(rows * cols);
            const float scale = std::sqrt(2.0f / static_cast<float>(cols));
            for (float& value : host) {
                value = dist(rng_) * scale;
            }
            CUDA_CHECK(cudaMemcpy(device_ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
        };

        init_matrix(d_W0_, hidden1_, input_size_);
        init_matrix(d_W1_, hidden2_, hidden1_);
        init_matrix(d_W2_, output_size_, hidden2_);

        CUDA_CHECK(cudaMemset(d_b0_, 0, hidden1_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_b1_, 0, hidden2_ * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_b2_, 0, output_size_ * sizeof(float)));
    }

    void forward(const float* d_input) {
        const int threads = 256;
        int blocks = (hidden1_ + threads - 1) / threads;
        linear_relu_forward<<<blocks, threads>>>(d_W0_, d_b0_, d_input, d_hidden1_, hidden1_, input_size_);
        CUDA_CHECK(cudaGetLastError());

        blocks = (hidden2_ + threads - 1) / threads;
        linear_relu_forward<<<blocks, threads>>>(d_W1_, d_b1_, d_hidden1_, d_hidden2_, hidden2_, hidden1_);
        CUDA_CHECK(cudaGetLastError());

        blocks = (output_size_ + threads - 1) / threads;
        linear_forward<<<blocks, threads>>>(d_W2_, d_b2_, d_hidden2_, d_logits_, output_size_, hidden2_);
        CUDA_CHECK(cudaGetLastError());

        softmax_kernel<<<1, 1>>>(d_logits_, d_probs_, output_size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void backward(const float* d_input, int target) {
        const int threads = 256;
        int blocks = (output_size_ + threads - 1) / threads;
        compute_output_gradient<<<blocks, threads>>>(d_probs_, target, d_grad_output_, output_size_);
        CUDA_CHECK(cudaGetLastError());

        blocks = (hidden2_ + threads - 1) / threads;
        matvec_transposed<<<blocks, threads>>>(d_W2_, d_grad_output_, d_grad_hidden2_, output_size_, hidden2_);
        CUDA_CHECK(cudaGetLastError());
        relu_backward_inplace<<<blocks, threads>>>(d_hidden2_, d_grad_hidden2_, hidden2_);
        CUDA_CHECK(cudaGetLastError());

        blocks = (hidden1_ + threads - 1) / threads;
        matvec_transposed<<<blocks, threads>>>(d_W1_, d_grad_hidden2_, d_grad_hidden1_, hidden2_, hidden1_);
        CUDA_CHECK(cudaGetLastError());
        relu_backward_inplace<<<blocks, threads>>>(d_hidden1_, d_grad_hidden1_, hidden1_);
        CUDA_CHECK(cudaGetLastError());

        blocks = (output_size_ * hidden2_ + threads - 1) / threads;
        update_weights<<<blocks, threads>>>(d_W2_, d_grad_output_, d_hidden2_, learning_rate_, output_size_, hidden2_);
        CUDA_CHECK(cudaGetLastError());
        blocks = (output_size_ + threads - 1) / threads;
        update_biases<<<blocks, threads>>>(d_b2_, d_grad_output_, learning_rate_, output_size_);
        CUDA_CHECK(cudaGetLastError());

        blocks = (hidden2_ * hidden1_ + threads - 1) / threads;
        update_weights<<<blocks, threads>>>(d_W1_, d_grad_hidden2_, d_hidden1_, learning_rate_, hidden2_, hidden1_);
        CUDA_CHECK(cudaGetLastError());
        blocks = (hidden2_ + threads - 1) / threads;
        update_biases<<<blocks, threads>>>(d_b1_, d_grad_hidden2_, learning_rate_, hidden2_);
        CUDA_CHECK(cudaGetLastError());

        blocks = (hidden1_ * input_size_ + threads - 1) / threads;
        update_weights<<<blocks, threads>>>(d_W0_, d_grad_hidden1_, d_input, learning_rate_, hidden1_, input_size_);
        CUDA_CHECK(cudaGetLastError());
        blocks = (hidden1_ + threads - 1) / threads;
        update_biases<<<blocks, threads>>>(d_b0_, d_grad_hidden1_, learning_rate_, hidden1_);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

struct CmdOptions {
    std::filesystem::path dataset_path;
    int epochs = 50;
    float learning_rate = 1e-3f;
    int hidden1 = 32;
    int hidden2 = 16;
    int seed = std::random_device{}();
    bool show_help = false;
};

void print_usage(const std::filesystem::path& exe_path) {
    std::cout << "Usage: " << exe_path.filename().string() << " [options]\n"
              << "Options:\n"
              << "  --dataset <path>     Path to CSV features file (default: ../music/dataset/features_30_sec.csv)\n"
              << "  --epochs <int>       Number of training epochs (default: 50)\n"
              << "  --lr <float>         Learning rate (default: 0.001)\n"
              << "  --hidden1 <int>      Hidden layer 1 size (default: 32)\n"
              << "  --hidden2 <int>      Hidden layer 2 size (default: 16)\n"
              << "  --seed <int>         RNG seed (default: random)\n"
              << "  -h, --help           Show this message\n";
}

CmdOptions parse_arguments(int argc, char** argv, const std::filesystem::path& exe_dir) {
    CmdOptions options;
    options.dataset_path = (exe_dir / ".." / "music" / "dataset" / "features_30_sec.csv").lexically_normal();

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            options.dataset_path = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            options.epochs = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            options.learning_rate = std::stof(argv[++i]);
        } else if (arg == "--hidden1" && i + 1 < argc) {
            options.hidden1 = std::stoi(argv[++i]);
        } else if (arg == "--hidden2" && i + 1 < argc) {
            options.hidden2 = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            options.seed = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            options.show_help = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path exe_path = std::filesystem::weakly_canonical(argv[0]);
        const std::filesystem::path exe_dir = exe_path.parent_path();
        CmdOptions options = parse_arguments(argc, argv, exe_dir);
        if (options.show_help) {
            print_usage(exe_path);
            return 0;
        }

        std::cout << "Loading dataset from: " << options.dataset_path << std::endl;
        Dataset dataset = load_music_dataset(options.dataset_path);
        std::cout << "Loaded " << dataset.num_samples << " samples with " << dataset.feature_dim
                  << " features and " << dataset.label_names.size() << " labels." << std::endl;

        normalize_dataset(dataset);

        float* d_inputs = nullptr;
        CUDA_CHECK(cudaMalloc(&d_inputs, dataset.features.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_inputs, dataset.features.data(), dataset.features.size() * sizeof(float), cudaMemcpyHostToDevice));

        NeuralNetCuda network(dataset.feature_dim, options.hidden1, options.hidden2,
                              static_cast<int>(dataset.label_names.size()), options.learning_rate, options.seed);

        TrainingHistory history = network.train(d_inputs, dataset.labels, dataset.num_samples, options.epochs);

        std::cout << "Training completed. Final accuracy: "
                  << history.accuracies.back() * 100.0f << "%" << std::endl;

        // Demonstrate inference on the first sample
        std::vector<float> probs(dataset.label_names.size());
        network.predict_device_sample(d_inputs, probs);
        const int predicted = static_cast<int>(std::distance(probs.begin(),
            std::max_element(probs.begin(), probs.end())));
        std::cout << "Sample prediction: " << dataset.label_names[predicted]
                  << " (confidence: " << probs[predicted] * 100.0f << "%)"
                  << ", actual label: " << dataset.label_names[dataset.labels.front()] << std::endl;

        CUDA_CHECK(cudaFree(d_inputs));
        CUDA_CHECK(cudaDeviceReset());
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
