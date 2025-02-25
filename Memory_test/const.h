#ifndef CONST_H
#define CONST_H

#include <systemc>
#include "tlm.h"
#include "tlm_utils/simple_target_socket.h"
#include "tlm_utils/simple_initiator_socket.h"

using namespace sc_core;
using namespace sc_dt;
using namespace std;

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

const sc_time DDR_LATENCY=sc_time(20, SC_NS);
const sc_time GSM_LATENCY=sc_time(10, SC_NS);
const sc_time VCore_LATENCY=sc_time(10, SC_NS);

#endif
