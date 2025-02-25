#ifndef DDR_H
#define DDR_H

#include "const.h"  // 引入公用常量和头文件

template<typename T>
struct DDR : sc_module
{
    tlm_utils::simple_target_socket<DDR> socket;

    DDR(sc_module_name name) : sc_module(name), socket("socket") {
        socket.register_get_direct_mem_ptr(this, &DDR::get_direct_mem_ptr);

        mem = new T[DDR_SIZE / sizeof(T)];
        for (uint64_t i = 0; i < DDR_SIZE / sizeof(T); i++) {
            mem[i] = T();  // Initialize memory
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

    ~DDR() {
        delete[] mem;
    }

private:
    T* mem;
};

#endif
