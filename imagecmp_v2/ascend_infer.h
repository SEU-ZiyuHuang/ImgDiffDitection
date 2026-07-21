// 华为 Ascend ACL 的轻量 C++ 封装。
//
// AscendRuntime 管理进程/设备级 context 和 stream；AclModel 管理单个 .om 模型的
// 元数据、主机/设备缓冲区与一次同步推理。业务模型类组合使用这两个基础设施。
#pragma once

#include <acl/acl.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// 将 ACL 错误统一转换为带 API 名称和最近运行时消息的 C++ 异常。
static inline void CheckAcl(aclError ret, const char* api) {
    if (ret != ACL_SUCCESS) {
        const char* msg = aclGetRecentErrMsg();
        if (msg && msg[0] != '\0') {
            throw std::runtime_error(std::string(api) + " failed : " + std::to_string(static_cast<int>(ret)) + " , msg: " + msg);
        } else {
            throw std::runtime_error(std::string(api) + " failed : " + std::to_string(static_cast<int>(ret)));
        }
    }
}

static inline bool FileExists(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    return f.good();
}

class AscendRuntime {
public:
    // 昇腾运行时资源管理器（线程安全，按 device_id 共享）
    //
    // 设计目标：
    // 1) 进程级只做一次 aclInit/aclFinalize（避免重复初始化/多次 finalize 导致崩溃）
    // 2) 按 device_id 管理一套 context/stream（同一设备可被多个模型/对象复用）
    // 3) 引用计数：最后一个使用者析构时才销毁该设备资源并 reset device
    //
    // 使用方式：
    // - 每个推理类持有一个 AscendRuntime（或共享同一个 device_id 的实例）
    // - 推理/拷贝时使用 stream() 获取该设备的 aclrtStream
    explicit AscendRuntime(uint32_t device_id);

    // 析构时释放当前 device_id 的引用；若引用归零，会销毁 stream/context 并 reset device。
    ~AscendRuntime();

    AscendRuntime(const AscendRuntime&) = delete;
    AscendRuntime& operator=(const AscendRuntime&) = delete;

    aclrtStream stream() const;
    aclrtContext context() const;
    uint32_t device_id() const;

private:
    // 每个 device_id 对应一份共享状态（被多个 AscendRuntime 实例共享）
    struct DeviceState {
        uint32_t device_id;
        aclrtContext context;
        aclrtStream stream;
        int ref;
    };

    static DeviceState* acquire(uint32_t device_id);
    static void release(uint32_t device_id);
    // 在持有全局锁(mu)的情况下执行的“全局一次性初始化”：
    // - 第一次调用时执行 aclInit，并打印一次 soc/sdk 信息
    // - 后续调用仅增加全局引用计数，不会重复 aclInit
    static void init_global_once_locked();

    uint32_t device_id_;
    DeviceState* state_;

    static std::mutex mu_;
    static std::map<uint32_t, DeviceState> states_;
    static std::atomic<int> global_ref_;
};

class AclModel {
public:
    AclModel(AscendRuntime& rt, const std::string& om_path)
        : rt_(rt),
          model_path_(om_path),
          model_id_(0),
          model_loaded_(false),
          desc_(nullptr),
          input_count_(0),
          output_count_(0) {
        try {
            load(model_path_);
            create_io();
            print_model_io();
        } catch (...) {
            release();
            throw;
        }
    }

    ~AclModel() {
        release();
    }

    // 禁用拷贝构造和赋值
    AclModel(const AclModel&) = delete;
    AclModel& operator=(const AclModel&) = delete;

    size_t num_inputs() const { return input_sizes_.size(); }
    size_t num_outputs() const { return output_sizes_.size(); }

    size_t input_size(size_t idx) const { return input_sizes_.at(idx); }
    size_t output_size(size_t idx) const { return output_sizes_.at(idx); }

    const aclmdlIODims& input_dims(size_t idx) const { return input_dims_.at(idx); }
    const aclmdlIODims& output_dims(size_t idx) const { return output_dims_.at(idx); }

    void* input_host(size_t idx) { return input_host_.at(idx); }
    void* output_host(size_t idx) { return output_host_.at(idx); }

    // 一次同步推理：复制主机输入到 NPU，构造 ACL dataset，执行模型，再复制输出回主机。
    // 调用方负责在 input_host() 填入与模型输入大小、数据类型一致的数据。
    void Execute() {
        CheckAcl(aclrtSetDevice(rt_.device_id()), "aclrtSetDevice");
        CheckAcl(aclrtSetCurrentContext(rt_.context()), "aclrtSetCurrentContext");

        aclmdlDataset* input_ds = nullptr;
        aclmdlDataset* output_ds = nullptr;
        std::vector<aclDataBuffer*> input_dbs;
        std::vector<aclDataBuffer*> output_dbs;

        auto cleanup = [&]() {
            for (aclDataBuffer* db : input_dbs) {
                if (db) (void)aclDestroyDataBuffer(db);
            }
            for (aclDataBuffer* db : output_dbs) {
                if (db) (void)aclDestroyDataBuffer(db);
            }
            input_dbs.clear();
            output_dbs.clear();
            if (input_ds) {
                (void)aclmdlDestroyDataset(input_ds);
                input_ds = nullptr;
            }
            if (output_ds) {
                (void)aclmdlDestroyDataset(output_ds);
                output_ds = nullptr;
            }
        };

        try {
            for (size_t i = 0; i < input_count_; ++i) {
                CheckAcl(aclrtMemcpy(
                             input_dev_[i],
                             input_sizes_[i],
                             input_host_[i],
                             input_sizes_[i],
                             ACL_MEMCPY_HOST_TO_DEVICE),
                         "aclrtMemcpy(H2D)");
            }

            for (size_t i = 0; i < output_count_; ++i) {
                CheckAcl(aclrtMemset(output_dev_[i], output_sizes_[i], 0, output_sizes_[i]), "aclrtMemset(output)");
            }

            input_ds = aclmdlCreateDataset();
            output_ds = aclmdlCreateDataset();
            if (!input_ds || !output_ds) {
                throw std::runtime_error("aclmdlCreateDataset failed");
            }

            input_dbs.reserve(input_count_);
            output_dbs.reserve(output_count_);

            for (size_t i = 0; i < input_count_; ++i) {
                aclDataBuffer* db = aclCreateDataBuffer(input_dev_[i], input_sizes_[i]);
                if (!db) {
                    throw std::runtime_error("aclCreateDataBuffer(input) failed");
                }
                input_dbs.push_back(db);
                CheckAcl(aclmdlAddDatasetBuffer(input_ds, db), "aclmdlAddDatasetBuffer(input)");
            }

            for (size_t i = 0; i < output_count_; ++i) {
                aclDataBuffer* db = aclCreateDataBuffer(output_dev_[i], output_sizes_[i]);
                if (!db) {
                    throw std::runtime_error("aclCreateDataBuffer(output) failed");
                }
                output_dbs.push_back(db);
                CheckAcl(aclmdlAddDatasetBuffer(output_ds, db), "aclmdlAddDatasetBuffer(output)");
            }

            CheckAcl(aclmdlExecute(model_id_, input_ds, output_ds), "aclmdlExecute");
            CheckAcl(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice");

            for (size_t i = 0; i < output_count_; ++i) {
                CheckAcl(aclrtMemcpy(
                             output_host_[i],
                             output_sizes_[i],
                             output_dev_[i],
                             output_sizes_[i],
                             ACL_MEMCPY_DEVICE_TO_HOST),
                         "aclrtMemcpy(D2H)");
            }

            cleanup();
        } catch (...) {
            cleanup();
            throw;
        }
    }

private:
    AscendRuntime& rt_;
    std::string model_path_;
    uint32_t model_id_;
    bool model_loaded_;
    aclmdlDesc* desc_;
    size_t input_count_;
    size_t output_count_;

    std::vector<size_t> input_sizes_;
    std::vector<size_t> output_sizes_;
    std::vector<aclmdlIODims> input_dims_;
    std::vector<aclmdlIODims> output_dims_;

    std::vector<void*> input_host_;
    std::vector<void*> output_host_;
    std::vector<void*> input_dev_;
    std::vector<void*> output_dev_;

    static void print_dims(const aclmdlIODims& dims) {
        std::cout << "[";
        for (int i = 0; i < dims.dimCount; ++i) {
            std::cout << dims.dims[i];
            if (i + 1 < dims.dimCount) std::cout << ", ";
        }
        std::cout << "]";
    }

    static const char* dtype_name(aclDataType t) {
        switch (t) {
            case ACL_FLOAT: return "float32";
            case ACL_FLOAT16: return "float16";
            case ACL_INT8: return "int8";
            case ACL_INT16: return "int16";
            case ACL_INT32: return "int32";
            case ACL_INT64: return "int64";
            case ACL_UINT8: return "uint8";
            case ACL_UINT16: return "uint16";
            case ACL_UINT32: return "uint32";
            case ACL_UINT64: return "uint64";
            case ACL_BOOL: return "bool";
            default: return "unknown";
        }
    }

    static const char* format_name(aclFormat f) {
        switch (f) {
            case ACL_FORMAT_NCHW: return "NCHW";
            case ACL_FORMAT_NHWC: return "NHWC";
            case ACL_FORMAT_ND: return "ND";
            case ACL_FORMAT_NC1HWC0: return "NC1HWC0";
            case ACL_FORMAT_FRACTAL_Z: return "FRACTAL_Z";
            case ACL_FORMAT_FRACTAL_NZ: return "FRACTAL_NZ";
            case ACL_FORMAT_HWCN: return "HWCN";
            case ACL_FORMAT_NDHWC: return "NDHWC";
            case ACL_FORMAT_NCDHW: return "NCDHW";
            default: return "UNKNOWN";
        }
    }

    void print_model_io() const {
        std::cout << "===== AclModel IO =====" << std::endl;
        std::cout << "model: " << model_path_ << std::endl;
        std::cout << "inputs: " << input_sizes_.size() << ", outputs: " << output_sizes_.size() << std::endl;

        for (size_t i = 0; i < input_sizes_.size(); ++i) {
            aclDataType dt = aclmdlGetInputDataType(desc_, i);
            aclFormat fmt = aclmdlGetInputFormat(desc_, i);
            std::cout << "  input[" << i << "] bytes=" << input_sizes_[i] << " dtype=" << dtype_name(dt) << " format=" << format_name(fmt)
                      << " dims=";
            print_dims(input_dims_[i]);
            std::cout << std::endl;
        }
        for (size_t i = 0; i < output_sizes_.size(); ++i) {
            aclDataType dt = aclmdlGetOutputDataType(desc_, i);
            aclFormat fmt = aclmdlGetOutputFormat(desc_, i);
            std::cout << "  output[" << i << "] bytes=" << output_sizes_[i] << " dtype=" << dtype_name(dt) << " format=" << format_name(fmt)
                      << " dims=";
            print_dims(output_dims_[i]);
            std::cout << std::endl;
        }
        std::cout << "=======================" << std::endl;
    }

    // 加载 .om 并读取输入输出数量、形状、格式和数据类型。
    // 注意：当前实现遇到模型缺失会 std::exit(-1)，这会终止宿主进程；生产化时宜改为抛异常。
    void load(const std::string& om_path) {
        if (!FileExists(om_path)) {
            std::cerr << "ERROR: 模型文件不存在 -> " << om_path << std::endl;
            std::exit(-1);
        }
        CheckAcl(aclmdlLoadFromFile(om_path.c_str(), &model_id_), "aclmdlLoadFromFile");
        model_loaded_ = true;

        desc_ = aclmdlCreateDesc();
        if (!desc_) {
            std::cerr << "aclmdlCreateDesc failed" << std::endl;
            std::exit(-1);
        }
        CheckAcl(aclmdlGetDesc(desc_, model_id_), "aclmdlGetDesc");
    }

    // 为每个模型输入/输出分配一份 pinned host 内存和一份 NPU device 内存。
    void create_io() {
        input_count_ = aclmdlGetNumInputs(desc_);
        output_count_ = aclmdlGetNumOutputs(desc_);

        input_sizes_.resize(input_count_);
        input_dims_.resize(input_count_);
        input_host_.resize(input_count_);
        input_dev_.resize(input_count_);

        for (size_t i = 0; i < input_count_; ++i) {
            size_t sz = aclmdlGetInputSizeByIndex(desc_, i);
            input_sizes_[i] = sz;

            aclmdlIODims dims;
            CheckAcl(aclmdlGetInputDims(desc_, i, &dims), "aclmdlGetInputDims");
            input_dims_[i] = dims;

            void* host = nullptr;
            void* dev = nullptr;
            CheckAcl(aclrtMallocHost(&host, sz), "aclrtMallocHost(input)");
            CheckAcl(aclrtMalloc(&dev, sz, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(input)");
            input_host_[i] = host;
            input_dev_[i] = dev;
        }

        output_sizes_.resize(output_count_);
        output_dims_.resize(output_count_);
        output_host_.resize(output_count_);
        output_dev_.resize(output_count_);

        for (size_t i = 0; i < output_count_; ++i) {
            size_t sz = aclmdlGetOutputSizeByIndex(desc_, i);
            output_sizes_[i] = sz;

            aclmdlIODims dims;
            CheckAcl(aclmdlGetOutputDims(desc_, i, &dims), "aclmdlGetOutputDims");
            output_dims_[i] = dims;

            void* host = nullptr;
            void* dev = nullptr;
            CheckAcl(aclrtMallocHost(&host, sz), "aclrtMallocHost(output)");
            CheckAcl(aclrtMalloc(&dev, sz, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(output)");
            output_host_[i] = host;
            output_dev_[i] = dev;
        }
    }

    // 按与创建相反的顺序释放设备内存、主机内存、模型描述和模型句柄。
    void release() {
        for (void* p : input_dev_) {
            if (p) {
                (void)aclrtFree(p);
            }
        }
        for (void* p : output_dev_) {
            if (p) {
                (void)aclrtFree(p);
            }
        }
        for (void* p : input_host_) {
            if (p) {
                (void)aclrtFreeHost(p);
            }
        }
        for (void* p : output_host_) {
            if (p) {
                (void)aclrtFreeHost(p);
            }
        }
        input_dev_.clear();
        output_dev_.clear();
        input_host_.clear();
        output_host_.clear();
        input_sizes_.clear();
        output_sizes_.clear();
        input_dims_.clear();
        output_dims_.clear();
        input_count_ = 0;
        output_count_ = 0;

        if (desc_) {
            (void)aclmdlDestroyDesc(desc_);
            desc_ = nullptr;
        }
        if (model_loaded_) {
            (void)aclmdlUnload(model_id_);
            model_loaded_ = false;
        }
    }
};
