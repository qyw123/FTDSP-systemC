#ifndef CONST_H
#define CONST_H

#include <systemc>
#include "tlm.h"
#include "tlm_utils/simple_target_socket.h"
#include "tlm_utils/simple_initiator_socket.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <array>

using namespace sc_core;
using namespace sc_dt;
using namespace std;
using namespace tlm;

// Word size
const uint64_t WORD_SIZE = 4;  // Word size in bytes (32 bits)

// DDR configurations
const uint64_t DDR_BASE_ADDR = 0x080000000;  // DDR base address
const uint64_t DDR_SIZE = 4L * 1024 * 1024 * 1024;  // DDR size (4GB)
const uint64_t DDR_DATA_WIDTH = 64;  //每拍传输的字节数

// GSM configurations
const uint64_t GSM_BASE_ADDR = 0x070000000;  // GSM base address
const uint64_t GSM_SIZE = 8L * 1024 * 1024;  // GSM size (8MB)

// VCore configurations
const uint64_t VCORE_BASE_ADDR = 0x010000000;  // VCore base address
const uint64_t VCORE_SIZE = 4L * 1024 * 1024;  // VCore size (4MB)
// SM configurations
const uint64_t SM_BASE_ADDR = 0x010000000;  // SM base address
const uint64_t SM_SIZE = 128L * 1024 ;  // SM size (128KB)
// AM configurations
const uint64_t AM_BASE_ADDR = 0x010030000;  // AM base address
const uint64_t AM_SIZE = 768L * 1024 ;  // AM size (768KB)

//访存延迟
const sc_time DDR_LATENCY=sc_time(20, SC_NS);
const sc_time GSM_LATENCY=sc_time(10, SC_NS);
const sc_time VCore_LATENCY=sc_time(10, SC_NS);
// 每个 MAC 的延迟
const sc_time MAC_LATENCY = sc_time(50, SC_NS); 
const int MACS_PER_VPE = 4;                    // 每个 VPE 包含的 MAC 单元数
const int VPES_PER_VPU = 16;                   // 每个 VPU 包含的 VPE 数量
const uint64_t mac_address_base = 0x100;        // MAC起始地址为 0x100
const int macs_per_vpu = MACS_PER_VPE*VPES_PER_VPU; 
//分块参数
const int cu_max = 64;
const int k_gsm_max = 384;
const int m_gsm_max = 384;
const int sm_max = 12;

#endif
