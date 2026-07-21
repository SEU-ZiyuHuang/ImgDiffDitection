// JPEG 等二进制结果的 Base64 编解码工具。
// 主算法仅在调用方要求时，将样本和实时部件裁剪图封装为 data:image/jpeg;base64,...。
#ifndef HV_BASE64_H_
#define HV_BASE64_H_



#define BASE64_ENCODE_OUT_SIZE(s)   (((s) + 2) / 3 * 4)
#define BASE64_DECODE_OUT_SIZE(s)   (((s) + 3) / 4 * 3)


#ifdef __cplusplus
extern "C" {
#endif

    // @return encoded size
    int hv_base64_encode(const unsigned char* in, unsigned int inlen, char* out);

    // @return decoded size
    int hv_base64_decode(const char* in, unsigned int inlen, unsigned char* out);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <string.h>
#include <string>


inline std::string Base64Encode(const unsigned char* data, unsigned int len) {
    int encoded_size = BASE64_ENCODE_OUT_SIZE(len);
    std::string encoded_str(encoded_size + 1, 0);
    encoded_size = hv_base64_encode(data, len, (char*)encoded_str.data());
    encoded_str.resize(encoded_size);
    return encoded_str;
}

inline std::string Base64Decode(const char* str, unsigned int len = 0) {
    if (len == 0) len = strlen(str);
    int decoded_size = BASE64_DECODE_OUT_SIZE(len);
    std::string decoded_buf(decoded_size + 1, 0);
    decoded_size = hv_base64_decode(str, len, (unsigned char*)decoded_buf.data());
    if (decoded_size > 0) {
        decoded_buf.resize(decoded_size);
    } else {
        decoded_buf.clear();
    }
    return decoded_buf;
}

#endif

#endif // HV_BASE64_H_
