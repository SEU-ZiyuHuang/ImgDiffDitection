// 已登记 ONNX 模型的本地契约与参考推理检查。
//
// 该工具不读取项目图片；每个模型都使用确定性合成张量运行一次。它的职责是
// 在模型进入比较管线之前，把文件替换、节点/张量契约变化和运行时不兼容变成
// 明确的非零退出，而不是让旧原型在运行时才以模糊错误失败。
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace {

class Sha256 {
public:
    Sha256() { reset(); }

    void update(const unsigned char* data, size_t length) {
        bit_count_ += static_cast<std::uint64_t>(length) * 8U;
        for (size_t index = 0; index < length; ++index) {
            buffer_[buffer_length_++] = data[index];
            if (buffer_length_ == buffer_.size()) {
                transform(buffer_.data());
                buffer_length_ = 0;
            }
        }
    }

    std::array<unsigned char, 32> finish() {
        const std::uint64_t source_bits = bit_count_;
        const unsigned char one = 0x80U;
        update(&one, 1);
        const unsigned char zero = 0;
        while (buffer_length_ != 56) update(&zero, 1);
        std::array<unsigned char, 8> bit_length = {};
        for (size_t index = 0; index < bit_length.size(); ++index) {
            bit_length[bit_length.size() - 1 - index] =
                static_cast<unsigned char>((source_bits >> (index * 8U)) & 0xffU);
        }
        update(bit_length.data(), bit_length.size());

        std::array<unsigned char, 32> digest = {};
        for (size_t index = 0; index < state_.size(); ++index) {
            digest[index * 4] = static_cast<unsigned char>((state_[index] >> 24U) & 0xffU);
            digest[index * 4 + 1] = static_cast<unsigned char>((state_[index] >> 16U) & 0xffU);
            digest[index * 4 + 2] = static_cast<unsigned char>((state_[index] >> 8U) & 0xffU);
            digest[index * 4 + 3] = static_cast<unsigned char>(state_[index] & 0xffU);
        }
        return digest;
    }

private:
    static std::uint32_t rotate_right(std::uint32_t value, unsigned int count) {
        return (value >> count) | (value << (32U - count));
    }

    void reset() {
        state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                  0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
        buffer_length_ = 0;
        bit_count_ = 0;
    }

    void transform(const unsigned char* block) {
        static constexpr std::array<std::uint32_t, 64> k = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
            0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
            0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
            0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
            0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
            0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
        std::array<std::uint32_t, 64> words = {};
        for (size_t index = 0; index < 16; ++index) {
            words[index] = (static_cast<std::uint32_t>(block[index * 4]) << 24U) |
                           (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16U) |
                           (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8U) |
                           static_cast<std::uint32_t>(block[index * 4 + 3]);
        }
        for (size_t index = 16; index < words.size(); ++index) {
            const std::uint32_t sigma0 = rotate_right(words[index - 15], 7) ^
                                         rotate_right(words[index - 15], 18) ^
                                         (words[index - 15] >> 3U);
            const std::uint32_t sigma1 = rotate_right(words[index - 2], 17) ^
                                         rotate_right(words[index - 2], 19) ^
                                         (words[index - 2] >> 10U);
            words[index] = words[index - 16] + sigma0 + words[index - 7] + sigma1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (size_t index = 0; index < words.size(); ++index) {
            const std::uint32_t sigma1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const std::uint32_t choice = (e & f) ^ ((~e) & g);
            const std::uint32_t temporary1 = h + sigma1 + choice + k[index] + words[index];
            const std::uint32_t sigma0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temporary2 = sigma0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = {};
    std::array<unsigned char, 64> buffer_ = {};
    size_t buffer_length_ = 0;
    std::uint64_t bit_count_ = 0;
};

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open model file: " + path.u8string());

    Sha256 sha256;
    std::array<unsigned char, 8192> buffer = {};
    while (stream.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size())) ||
           stream.gcount() > 0) {
        sha256.update(buffer.data(), static_cast<size_t>(stream.gcount()));
    }
    if (!stream.eof()) throw std::runtime_error("cannot read model file: " + path.u8string());

    std::ostringstream text;
    for (unsigned char byte : sha256.finish()) {
        text << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(byte);
    }
    return text.str();
}

Ort::Session create_session(Ort::Env& environment, const std::filesystem::path& model_path) {
    Ort::SessionOptions options;
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
#ifdef _WIN32
    return Ort::Session(environment, model_path.c_str(), options);
#else
    return Ort::Session(environment, model_path.c_str(), options);
#endif
}

std::string tensor_type_name(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        default: return "ONNX type " + std::to_string(static_cast<int>(type));
    }
}

std::string shape_text(const std::vector<int64_t>& shape) {
    std::ostringstream text;
    text << '[';
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) text << ',';
        text << shape[index];
    }
    text << ']';
    return text.str();
}

size_t element_count(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (const int64_t dimension : shape) {
        if (dimension < 0) throw std::runtime_error("reference inference returned an unresolved tensor dimension");
        if (dimension == 0) return 0;
        if (count > std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension)) {
            throw std::runtime_error("tensor is too large to validate");
        }
        count *= static_cast<size_t>(dimension);
    }
    return count;
}

std::vector<std::string> session_names(Ort::Session& session, bool inputs) {
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
    std::vector<std::string> names;
    names.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        Ort::AllocatedStringPtr name = inputs ? session.GetInputNameAllocated(index, allocator)
                                              : session.GetOutputNameAllocated(index, allocator);
        if (!name) throw std::runtime_error("cannot read ONNX node name");
        names.emplace_back(name.get());
    }
    return names;
}

std::vector<const char*> c_string_views(const std::vector<std::string>& names) {
    std::vector<const char*> result;
    result.reserve(names.size());
    for (const std::string& name : names) result.push_back(name.c_str());
    return result;
}

struct ModelSpec {
    const char* name;
    const char* filename;
    const char* sha256;
    const char* input_name;
    std::vector<int64_t> input_contract_shape;
    std::vector<int64_t> reference_input_shape;
    double reference_sum;
    double reference_abs_max;
};

const std::array<ModelSpec, 3> kModels = {{
    {"SuperPoint+LightGlue", "superpoint_lightglue_pipeline.onnx",
     "228994cea8c010146fa2aef933baa3ffaa4bcdc522bc8aa560087fcff8134526", "images",
     {-1, 1, -1, -1}, {2, 1, 512, 512}, 1076416.000000, 503.000000},
    {"LDC", "LDC_640x360.onnx",
     "1895fa66262c9caac1dfe0e4ff7b180f99ea2c1b5993906b6443bede4da4ac62", "input_image",
     {1, 3, 360, 640}, {1, 3, 360, 640}, -3233133.436660, 10.780195},
    {"ResNet18 feature extractor", "resnet18.onnx",
     "c812837bb5132a5757b42c63d07041e6f84a563f18b5d780d4ae8e5dfed37c2b", "input",
     {1, 3, 224, 224}, {1, 3, 224, 224}, 5651.709883, 1.832119},
}};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string output_layout(const std::vector<std::string>& names, const std::vector<Ort::Value>& outputs) {
    std::ostringstream text;
    for (size_t index = 0; index < outputs.size(); ++index) {
        if (index != 0) text << "; ";
        const auto info = outputs[index].GetTensorTypeAndShapeInfo();
        text << (index < names.size() ? names[index] : "<unnamed>") << ' '
             << tensor_type_name(info.GetElementType()) << ' ' << shape_text(info.GetShape());
    }
    return text.str();
}

void validate_output_contract(const ModelSpec& spec, const std::vector<std::string>& output_names,
                              const std::vector<Ort::Value>& outputs) {
    require(output_names.size() == outputs.size(), "runtime output count is inconsistent");
    if (std::strcmp(spec.filename, "superpoint_lightglue_pipeline.onnx") == 0) {
        require(output_names == std::vector<std::string>({"keypoints", "matches", "mscores"}),
                "SuperPoint+LightGlue output names changed");
        require(outputs.size() == 3, "SuperPoint+LightGlue must return exactly three tensors");
        const auto keypoints = outputs[0].GetTensorTypeAndShapeInfo();
        const auto matches = outputs[1].GetTensorTypeAndShapeInfo();
        const auto scores = outputs[2].GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> keypoint_shape = keypoints.GetShape();
        const std::vector<int64_t> match_shape = matches.GetShape();
        require(keypoints.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
                    keypoint_shape.size() == 3 && keypoint_shape[0] == 2 && keypoint_shape[1] >= 0 &&
                    keypoint_shape[2] == 2,
                "SuperPoint+LightGlue keypoints must be int64 [2,N,2]");
        require(matches.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
                    !match_shape.empty() && match_shape.back() == 3,
                "SuperPoint+LightGlue matches must be int64 triplets");
        require(scores.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "SuperPoint+LightGlue match scores must be non-empty float32");
        return;
    }
    if (std::strcmp(spec.filename, "LDC_640x360.onnx") == 0) {
        require(outputs.size() == 5, "LDC must return five edge-map tensors");
        for (const Ort::Value& output : outputs) {
            const auto info = output.GetTensorTypeAndShapeInfo();
            const std::vector<int64_t> shape = info.GetShape();
            require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && shape.size() == 4 &&
                        shape[0] == 1 && shape[1] == 1 && shape[2] > 0 && shape[3] > 0,
                    "each LDC output must be float32 [1,1,H,W]");
        }
        return;
    }
    require(output_names == std::vector<std::string>({"output"}), "ResNet18 output name changed");
    require(outputs.size() == 1, "ResNet18 must return exactly one feature tensor");
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                info.GetShape() == std::vector<int64_t>({1, 256, 14, 14}),
            "ResNet18 output must be float32 [1,256,14,14]");
}

std::pair<double, double> summarize_outputs(const std::vector<Ort::Value>& outputs) {
    double sum = 0.0;
    double maximum = 0.0;
    for (const Ort::Value& output : outputs) {
        const auto info = output.GetTensorTypeAndShapeInfo();
        const size_t count = element_count(info.GetShape());
        if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float* values = output.GetTensorData<float>();
            for (size_t index = 0; index < count; ++index) {
                require(std::isfinite(values[index]), "reference inference returned a non-finite float");
                sum += values[index];
                maximum = std::max(maximum, std::abs(static_cast<double>(values[index])));
            }
        } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            const int64_t* values = output.GetTensorData<int64_t>();
            for (size_t index = 0; index < count; ++index) {
                sum += static_cast<double>(values[index]);
                maximum = std::max(maximum, std::abs(static_cast<double>(values[index])));
            }
        } else {
            throw std::runtime_error("reference inference returned an unsupported " +
                                     tensor_type_name(info.GetElementType()) + " tensor");
        }
    }
    return {sum, maximum};
}

bool approximately_equal(double actual, double expected) {
    const double tolerance = 0.0001 + std::abs(expected) * 0.000001;
    return std::abs(actual - expected) <= tolerance;
}

void verify_model(Ort::Env& environment, const std::filesystem::path& model_directory, const ModelSpec& spec) {
    std::cerr << "[CHECK] " << spec.name << '\n';
    const std::filesystem::path model_path = model_directory / spec.filename;
    const std::string actual_sha256 = sha256_file(model_path);
    require(actual_sha256 == spec.sha256,
            "SHA-256 differs from the governed artifact (expected " + std::string(spec.sha256) + ", got " + actual_sha256 + ')');

    Ort::Session session = create_session(environment, model_path);
    const std::vector<std::string> input_names = session_names(session, true);
    const std::vector<std::string> output_names = session_names(session, false);
    require(input_names == std::vector<std::string>({spec.input_name}), "input node name changed");
    const auto input_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    require(input_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            "input tensor type changed (expected float32, got " + tensor_type_name(input_info.GetElementType()) + ')');
    require(input_info.GetShape() == spec.input_contract_shape,
            "input shape changed (expected " + shape_text(spec.input_contract_shape) + ", got " +
                shape_text(input_info.GetShape()) + ')');

    const size_t input_count = element_count(spec.reference_input_shape);
    std::vector<float> reference_input(input_count);
    for (size_t index = 0; index < input_count; ++index) {
        reference_input[index] = static_cast<float>((index * 17U + 29U) % 256U);
        if (std::strcmp(spec.filename, "superpoint_lightglue_pipeline.onnx") == 0 ||
            std::strcmp(spec.filename, "resnet18.onnx") == 0) {
            reference_input[index] /= 255.0F;
        }
    }
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(memory, reference_input.data(), reference_input.size(),
                                                         spec.reference_input_shape.data(), spec.reference_input_shape.size());
    const std::vector<const char*> input_name_views = c_string_views(input_names);
    const std::vector<const char*> output_name_views = c_string_views(output_names);
    std::vector<Ort::Value> outputs = session.Run(Ort::RunOptions{nullptr}, input_name_views.data(), &input, 1,
                                                  output_name_views.data(), output_name_views.size());
    std::cerr << "[CHECK] " << spec.name << " reference inference completed\n";
    try {
        validate_output_contract(spec, output_names, outputs);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(error.what()) + " (received " + output_layout(output_names, outputs) + ')');
    }
    const auto summary = summarize_outputs(outputs);
    std::cerr << "[CHECK] " << spec.name << " reference output summarized\n";
    require(approximately_equal(summary.first, spec.reference_sum) &&
                approximately_equal(summary.second, spec.reference_abs_max),
            "deterministic reference inference changed (expected sum=" + std::to_string(spec.reference_sum) +
                ", abs_max=" + std::to_string(spec.reference_abs_max) + "; got sum=" +
                std::to_string(summary.first) + ", abs_max=" + std::to_string(summary.second) + ')');

    std::cout << "[PASS] " << spec.name << '\n'
              << "  sha256: " << actual_sha256 << '\n'
              << "  input: " << input_names[0] << ' ' << shape_text(spec.input_contract_shape) << '\n'
              << "  reference input: " << shape_text(spec.reference_input_shape) << '\n';
    for (size_t index = 0; index < outputs.size(); ++index) {
        const auto info = outputs[index].GetTensorTypeAndShapeInfo();
        std::cout << "  output[" << index << "]: " << output_names[index] << ' '
                  << tensor_type_name(info.GetElementType()) << ' ' << shape_text(info.GetShape()) << '\n';
    }
    std::cout << std::fixed << std::setprecision(6)
              << "  reference_sum: " << summary.first << '\n'
              << "  reference_abs_max: " << summary.second << '\n';
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [--model-dir <directory>]\n"
              << "Validates governed ONNX artifacts without reading project images.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::filesystem::path model_directory = ".";
#ifdef _WIN32
    int wide_argc = 0;
    LPWSTR* wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc);
    if (!wide_argv) {
        std::cerr << "Cannot read the Windows command line.\n";
        return 2;
    }
    std::vector<std::wstring> arguments(wide_argv, wide_argv + wide_argc);
    LocalFree(wide_argv);
    for (int index = 1; index < wide_argc; ++index) {
        const std::wstring& argument = arguments[static_cast<size_t>(index)];
        if (argument == L"--model-dir" && index + 1 < wide_argc) {
            model_directory = std::filesystem::path(arguments[static_cast<size_t>(++index)]);
        } else if (argument == L"--help" || argument == L"-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::wcerr << L"Unknown or incomplete argument: " << argument << '\n';
            print_usage(argv[0]);
            return 2;
        }
    }
#else
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--model-dir" && index + 1 < argc) {
            model_directory = std::filesystem::u8path(argv[++index]);
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << argument << '\n';
            print_usage(argv[0]);
            return 2;
        }
    }
#endif

    try {
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "imagecmp_model_contract_check");
        std::cout << "ONNX Runtime " << OrtGetApiBase()->GetVersionString() << '\n';
        for (const ModelSpec& spec : kModels) verify_model(environment, model_directory, spec);
        std::cout << "All governed ONNX model contracts passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
