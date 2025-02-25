#ifndef VCORE_H
#define VCORE_H

#include "const.h"

template<typename T>
struct VCore : sc_module {
    tlm_utils::simple_target_socket<VCore> target_socket;
    tlm_utils::simple_initiator_socket<VCore> initiator_socket;

    VCore(sc_module_name name) 
        : sc_module(name), 
          target_socket("target_socket"), 
          initiator_socket("initiator_socket")
    {
        // 注册 b_transport 和 get_direct_mem_ptr 回调
        target_socket.register_b_transport(this, &VCore::b_transport);
        target_socket.register_get_direct_mem_ptr(this, &VCore::get_direct_mem_ptr);

        mem = new T[VCORE_SIZE / sizeof(T)];
        for (uint64_t i = 0; i < VCORE_SIZE / sizeof(T); i++) {
            mem[i] = T();  // Initialize memory
        }
    }

    virtual void b_transport(tlm::tlm_generic_payload& trans, sc_time& delay) {
        // if (initiator_socket.get_base_port()==0) {
        //     SC_REPORT_ERROR("VCore", "Initiator socket not bound");
        //     return;
        // }
        initiator_socket->b_transport(trans, delay);
    }

    virtual bool get_direct_mem_ptr(tlm::tlm_generic_payload& trans, tlm::tlm_dmi& dmi_data) {
        dmi_data.set_start_address(VCORE_BASE_ADDR);
        dmi_data.set_end_address(VCORE_BASE_ADDR + VCORE_SIZE - 1);
        dmi_data.set_dmi_ptr(reinterpret_cast<unsigned char*>(mem)); // 使用 vector
        dmi_data.set_read_latency(VCore_LATENCY);
        dmi_data.set_write_latency(VCore_LATENCY);
        dmi_data.allow_read_write();
        return true; // 返回 true 表示支持 DMI
    }

    ~VCore() {
        delete[] mem;
    }

    private:
        T* mem;
};

#endif
