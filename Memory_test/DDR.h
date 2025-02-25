#ifndef DDR_H
#define DDR_H

#include "const.h"  // 引入公用常量和头文件

template<typename T>
struct DDR : sc_module {
    tlm_utils::simple_target_socket<DDR> socket;

    DDR(sc_module_name name) : sc_module(name), socket("socket") {
        socket.register_get_direct_mem_ptr(this, &DDR::get_direct_mem_ptr);

        mem = new T[DDR_SIZE / sizeof(T)];
        for (uint64_t i = 0; i < DDR_SIZE / sizeof(T); i++) {
            mem[i] = T();  // 初始化内存
        }
    }

    virtual bool get_direct_mem_ptr(tlm::tlm_generic_payload& trans, tlm::tlm_dmi& dmi_data) {
        dmi_data.set_start_address(DDR_BASE_ADDR);
        dmi_data.set_end_address(DDR_BASE_ADDR + DDR_SIZE - 1);
        dmi_data.set_dmi_ptr(reinterpret_cast<unsigned char*>(mem));
        dmi_data.set_read_latency(DDR_LATENCY);
        dmi_data.set_write_latency(DDR_LATENCY);
        dmi_data.allow_read_write();
        return true;
    }

    void transfer_data(uint64_t start_address, uint64_t end_address, T* buffer, bool is_write) {
        if (start_address < DDR_BASE_ADDR || end_address >= DDR_BASE_ADDR + DDR_SIZE || start_address > end_address) {
            SC_REPORT_ERROR("DDR", "Invalid address range for transfer");
            return;
        }

        uint64_t addr = start_address;
        T* buf_ptr = buffer;

        while (addr <= end_address) {
            // 根据模板常量 DDR_DATA_WIDTH 计算对应的数据单元个数
            uint64_t elements_per_transfer = DDR_DATA_WIDTH / sizeof(T);

            if (is_write) {
                // 写入数据
                memcpy(mem + (addr - DDR_BASE_ADDR) / sizeof(T), buf_ptr, DDR_DATA_WIDTH);
            } else {
                // 读取数据
                memcpy(buf_ptr, mem + (addr - DDR_BASE_ADDR) / sizeof(T), DDR_DATA_WIDTH);
            }

            // 模拟传输延时
            wait(DDR_LATENCY);

            // 输出调试信息
            /*std::cout << sc_time_stamp() << ": "
                      << (is_write ? "WRITE" : "READ") << " Addr = 0x" << std::hex << addr
                      << " - 0x" << std::hex << (addr + DDR_DATA_WIDTH - 1) << std::endl;*/

            // 更新地址和缓冲区指针
            addr += DDR_DATA_WIDTH;  // 根据传输大小更新地址
            buf_ptr += elements_per_transfer;  // 更新缓冲区指针
        }
    }

    ~DDR() {
        delete[] mem;
    }

private:
    T* mem;
};

#endif
