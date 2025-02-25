#ifndef GSM_H
#define GSM_H

#include "const.h"  // 引入公用常量和头文件

template<typename T>
struct GSM : sc_module
{
    tlm_utils::simple_target_socket<GSM> socket;

    GSM(sc_module_name name) : sc_module(name), socket("socket") {
        socket.register_get_direct_mem_ptr(this, &GSM::get_direct_mem_ptr);

        mem = new T[GSM_SIZE / sizeof(T)];
        for (uint64_t i = 0; i < GSM_SIZE / sizeof(T); i++) {
            mem[i] = T();  // Initialize memory
        }
    }

    virtual bool get_direct_mem_ptr(tlm::tlm_generic_payload& trans, tlm::tlm_dmi& dmi_data) {
        dmi_data.set_start_address(GSM_BASE_ADDR);
        dmi_data.set_end_address(GSM_BASE_ADDR + GSM_SIZE - 1);
        dmi_data.set_dmi_ptr(reinterpret_cast<unsigned char*>(mem));
        dmi_data.set_read_latency(GSM_LATENCY);
        dmi_data.set_write_latency(GSM_LATENCY);
        dmi_data.allow_read_write();
        return true;
    }

    ~GSM() {
        delete[] mem;
    }

private:
    T* mem;
};

#endif
