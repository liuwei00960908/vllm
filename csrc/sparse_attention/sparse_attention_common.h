#pragma once
#include <cstdint>

enum StorageDType : int32_t {
    DTYPE_FP32 = 0,
    DTYPE_FP16 = 1,
    DTYPE_BF16 = 2,
};

enum ErrorCode : int32_t {
    ERR_OK = 0,
    ERR_BAD_PARAM = 1,
    ERR_HQ_HKV_MISMATCH = 2,
    ERR_UNSUPPORTED_DTYPE = 3,
    ERR_LAUNCH = 4,
    ERR_NO_FREE_BLOCK = 13,
    ERR_NB_OVERFLOW = 15,
    ERR_BT_LEN_OVERFLOW = 16,
};

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#define FULL_MASK 0xFFFFFFFFu