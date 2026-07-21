// AscendRuntime 的进程级/设备级资源生命周期实现。
// 同一个 device_id 共用 ACL context 和 stream，并以引用计数管理创建与销毁。
#include "ascend_infer.h"
#include <atomic>
#include <mutex>
#include <utility>

std::mutex AscendRuntime::mu_;
std::map<uint32_t, AscendRuntime::DeviceState> AscendRuntime::states_;
std::atomic<int> AscendRuntime::global_ref_{0};

AscendRuntime::AscendRuntime(uint32_t device_id) : device_id_(device_id), state_(acquire(device_id_)) {}

AscendRuntime::~AscendRuntime() {
    release(device_id_);
}

aclrtStream AscendRuntime::stream() const {
    return state_->stream;
}

aclrtContext AscendRuntime::context() const {
    return state_->context;
}

uint32_t AscendRuntime::device_id() const {
    return device_id_;
}

// 进程首次使用时初始化 ACL；aclFinalize 被保留为注释，避免多库/多线程场景过早结束运行时。
void AscendRuntime::init_global_once_locked() {
    if (global_ref_.fetch_add(1, std::memory_order_acq_rel) != 0) {
        return;
    }
    CheckAcl(aclInit(nullptr), "aclInit");
    const char* socName = aclrtGetSocName();
    int32_t majorVersion = 0;
    int32_t minorVersion = 0;
    int32_t patchVersion = 0;
    (void)aclrtGetVersion(&majorVersion, &minorVersion, &patchVersion);
    std::cout << "Ascend socName:" << (socName ? socName : "unknown") << ", sdk version:" << majorVersion << "."
              << minorVersion << "." << patchVersion << std::endl;
}

// 取得指定设备的共享 context/stream；首次请求该设备时创建资源。
AscendRuntime::DeviceState* AscendRuntime::acquire(uint32_t device_id) {
    std::lock_guard<std::mutex> lock(mu_);
    init_global_once_locked();

    auto it = states_.find(device_id);
    if (it != states_.end()) {
        it->second.ref += 1;
        CheckAcl(aclrtSetDevice(device_id), "aclrtSetDevice");
        CheckAcl(aclrtSetCurrentContext(it->second.context), "aclrtSetCurrentContext");
        return &it->second;
    }

    DeviceState st;
    st.device_id = device_id;
    st.context = nullptr;
    st.stream = nullptr;
    st.ref = 1;

    CheckAcl(aclrtSetDevice(device_id), "aclrtSetDevice");
    CheckAcl(aclrtCreateContext(&st.context, device_id), "aclrtCreateContext");
    CheckAcl(aclrtSetCurrentContext(st.context), "aclrtSetCurrentContext");
    CheckAcl(aclrtCreateStream(&st.stream), "aclrtCreateStream");

    auto res = states_.emplace(device_id, st);
    return &res.first->second;
}

// 最后一个引用离开时释放该设备的 stream/context。
void AscendRuntime::release(uint32_t device_id) {
    std::lock_guard<std::mutex> lock(mu_);

    auto it = states_.find(device_id);
    if (it != states_.end()) {
        it->second.ref -= 1;
        if (it->second.ref == 0) {
            // CheckAcl(aclrtSetDevice(device_id), "aclrtSetDevice");
            if (it->second.stream) {
                (void)aclrtDestroyStream(it->second.stream);
                it->second.stream = nullptr;
            }
            if (it->second.context) {
                (void)aclrtDestroyContext(it->second.context);
                it->second.context = nullptr;
            }
            // (void)aclrtResetDevice(device_id);
            states_.erase(it);
        }
    }

    if (global_ref_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        // (void)aclFinalize();
    }
}
