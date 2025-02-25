#ifndef SOC_H
#define SOC_H

#include "CAC.h"
#include "DDR.h"
#include "GSM.h"
#include "VCore.h"
#include "vpu_tlm.h"
#include "vpe_tlm.h"
#include "mac_tlm.h"
#include "../util/const.h"

template <typename T>
SC_MODULE(Soc) {
    CAC<3>* cac;// 3个部分：DDR、GSM、VCore
    DDR<T>* ddr;
    GSM<T>* gsm;
    VCore<T>* vcore;
    tlm_utils::simple_target_socket<Soc> target_socket;
    tlm_utils::simple_initiator_socket<Soc> initiator_socket;

    bool get_direct_mem_ptr(tlm::tlm_generic_payload& trans, tlm::tlm_dmi& dmi_data) {
        uint64_t addr = trans.get_address();
        
        // 根据地址范围转发到对应的模块
        if (addr >= DDR_BASE_ADDR && addr < DDR_BASE_ADDR + DDR_SIZE) {
            // 转发到 DDR
            return (*(cac->mem_ctr_socket[0]))->get_direct_mem_ptr(trans, dmi_data);
        }
        else if (addr >= GSM_BASE_ADDR && addr < GSM_BASE_ADDR + GSM_SIZE) {
            // 转发到 GSM
            return (*(cac->mem_ctr_socket[1]))->get_direct_mem_ptr(trans, dmi_data);
        }
        else if (addr >= VCORE_BASE_ADDR && addr < VCORE_BASE_ADDR + VCORE_SIZE) {
            // 转发到 VCore
            return (*(cac->mem_ctr_socket[2]))->get_direct_mem_ptr(trans, dmi_data);
        }
        
        return false;  // 如果地址不在任何有效范围内
    }

    void b_transport(tlm::tlm_generic_payload& trans, sc_time& delay) {
        initiator_socket->b_transport(trans, delay);
    }

    SC_CTOR(Soc) {
        // 注册 DMI 处理函数
        target_socket.register_get_direct_mem_ptr(this, &Soc::get_direct_mem_ptr);
        // 注册 b_transport 处理函数
        target_socket.register_b_transport(this, &Soc::b_transport);
        cac = new CAC<3>("CAC");
        ddr = new DDR<T>("DDR");
        gsm = new GSM<T>("GSM");
        vcore = new VCore<T>("VCore");

        initiator_socket.bind(cac->target_socket);
        cac->mem_ctr_socket[0]->bind(ddr->socket);
        cac->mem_ctr_socket[1]->bind(gsm->socket);
        cac->mem_ctr_socket[2]->bind(vcore->target_socket);

        VPU_TLM<16>* vpu; // Instantiate a VPU with 16 VPEs
        vpu = new VPU_TLM<16> ("VPU_TLM");
        vcore->initiator_socket.bind(vpu->target_socket);

        // Instantiate VPEs
        VPE_TLM<4>* vpes[16];
        for (int i = 0; i < 16; ++i) {
            char name[20];
            sprintf(name, "VPE_%d", i);
            vpes[i] = new VPE_TLM<4>(name);
        }
        // Instantiate MACs
        MAC_TLM<T>* macs[64];
        for (int i = 0; i < 64; ++i) {
            char name[20];
            sprintf(name, "MAC_%d", i);
            macs[i] = new MAC_TLM<double>(name);
        }
        // Bind VPU to VPEs
        for (int i = 0; i < 16; ++i) {
            vpu->initiator_sockets[i]->bind(vpes[i]->target_socket);
        }

        // Bind VPEs to MACs
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 4; ++j) {
                vpes[i]->initiator_sockets[j]->bind(macs[i * 4 + j]->target_socket);
            }
        }

    }
};

#endif
