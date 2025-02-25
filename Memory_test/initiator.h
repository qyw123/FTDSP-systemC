#ifndef INITIATOR_H
#define INITIATOR_H

#include "systemc"
#include "DDR.h"
#include <vector>

using namespace sc_core;
using namespace sc_dt;

template <typename T>
struct Initiator : sc_module {
    tlm_utils::simple_initiator_socket<Initiator> socket;
    DDR<T>* ddr_instance;  // 存储 DDR 实例的指针

    SC_CTOR(Initiator) : socket("socket"), ddr_instance(nullptr) {
        socket.register_invalidate_direct_mem_ptr(this, &Initiator::invalidate_direct_mem_ptr);
        SC_THREAD(test_process);
    }

    void set_target(DDR<T>* ddr) {
        ddr_instance = ddr;
    }

    void test_process() {
        if (!ddr_instance) {
            SC_REPORT_ERROR("Initiator", "DDR instance not set.");
            return;
        }

        // 数据大小和缓冲区
        const uint64_t data_count = 6 * 384;  // 6 * 384 个 T 类型数据
        std::vector<T> write_buffer(data_count);
        std::vector<T> read_buffer(data_count);

        // 初始化写入缓冲区数据，从 1.1 开始，依次累加 1.1
        for (uint64_t i = 0; i < data_count; ++i) {
            write_buffer[i] = static_cast<T>(1.1 * (i + 1));
        }

        // 写入 DDR
        uint64_t write_start_address = DDR_BASE_ADDR;
        uint64_t write_end_address = write_start_address + (data_count * sizeof(T)) - 1;

        std::cout << "=== Writing to DDR ===" << std::endl;

        ddr_instance->transfer_data(write_start_address, write_end_address, write_buffer.data(), true);

        // 打印前十个写入数据和地址
        std::cout << "=== First 10 Written Data ===" << std::endl;
        for (uint64_t i = 0; i < 10; ++i) {
            uint64_t addr = write_start_address + (i * sizeof(T));
            std::cout << "Address: 0x" << std::hex << addr
                      << " Data: " << std::dec << write_buffer[i] << std::endl;
        }

        // Step 2: Read data from DDR
        std::cout << "=== Step 2: Read from DDR ===" << std::endl;
        //std::fill(buffer.begin(), buffer.end(), T());

        ddr_instance->transfer_data(write_start_address, write_end_address, read_buffer.data(), false);
        // 打印后十个读取数据和地址
        std::cout << "=== Last 10 Read Data ===" << std::endl;
        for (uint64_t i = data_count - 10; i < data_count; ++i) {
            uint64_t addr = write_start_address + (i * sizeof(T));
            std::cout << "Address: 0x" << std::hex << addr
                      << " Data: " << std::dec << read_buffer[i] << std::endl;
        }
        sc_stop();
    }

    virtual void invalidate_direct_mem_ptr(sc_dt::uint64 start_range, sc_dt::uint64 end_range) {
        std::cout << "DMI invalidated. Range: " << std::hex << start_range << " - " << end_range << std::endl;
    }
};

#endif
