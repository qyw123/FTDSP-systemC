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

const uint64_t N_TARGETS = 3;
const uint64_t DDR_id = 0;
const uint64_t GSM_id = 1;
const uint64_t VCore_id = 2;
// Word size
const uint64_t WORD_SIZE = 4;  // Word size in bytes (32 bits)

// DDR configurations
const uint64_t DDR_BASE_ADDR = 0x080000000;  // DDR base address
const uint64_t DDR_SIZE = 4L * 1024 * 1024 * 1024;  // DDR size (4GB)
const uint64_t DDR_DATA_WIDTH = 64;  //每拍传输的字节数

// GSM configurations
const uint64_t GSM_BASE_ADDR = 0x070000000;  // GSM base address
const uint64_t GSM_SIZE = 8L * 1024 * 1024;  // GSM size (8MB)
const uint64_t GSM_DATA_WIDTH = 64;  //每拍传输的字节数
// VCore configurations
const uint64_t VCORE_BASE_ADDR = 0x010000000;  // VCore base address
const uint64_t VCORE_SIZE = 4L * 1024 * 1024;  // VCore size (4MB)
// SM configurations
const uint64_t SM_BASE_ADDR = 0x010000000;  // SM base address
const uint64_t SM_SIZE = 128L * 1024 ;  // SM size (128KB)
// AM configurations
const uint64_t AM_BASE_ADDR = 0x010030000;  // AM base address
const uint64_t AM_SIZE = 768L * 1024 ;  // AM size (768KB)
// MAC configurations,在VCore中，且不影响AM和SM的空间
const uint64_t VPU_BASE_ADDR = 0x010100000;  // MAC base address
const uint64_t VPU_REGISTER_SIZE = 8L * 64 ;  // 64个64位寄存器
const uint64_t MAC_PER_VPU = 64;
//访存延迟
const sc_time DDR_LATENCY=sc_time(20, SC_NS);
const sc_time GSM_LATENCY=sc_time(10, SC_NS);
const sc_time SM_LATENCY=sc_time(1, SC_NS);
const sc_time AM_LATENCY=sc_time(1, SC_NS);
const sc_time MAC_LATENCY = sc_time(50, SC_NS); 

//分块参数
const int cu_max = 64;
const int k_gsm_max = 384;
const int m_gsm_max = 384;
const int sm_max = 12;

#endif
