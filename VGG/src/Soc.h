#ifndef SOC_H
#define SOC_H

#include "DMA.h"
#include "DDR.h"
#include "GSM.h"
#include "VCore.h"
#include "../util/const.h"
#include "../util/tools.h"
template <typename T>
SC_MODULE(Soc) {
    DMA<N_TARGETS>* dma;
    DDR<T>* ddr;
    GSM<T>* gsm;
    VCore<T>* vcore;
    tlm_utils::simple_target_socket<Soc> target_socket;
    tlm_utils::simple_initiator_socket<Soc> initiator_socket;

    SC_CTOR(Soc) {
        // 注册 DMI 处理函数
        target_socket.register_get_direct_mem_ptr(this, &Soc::get_direct_mem_ptr);
        // 注册 b_transport 处理函数
        target_socket.register_b_transport(this, &Soc::b_transport);
        dma = new DMA<N_TARGETS>("DMA");
        ddr = new DDR<T>("DDR");
        gsm = new GSM<T>("GSM");
        vcore = new VCore<T>("VCore");

        initiator_socket.bind(dma->target_socket);
        dma->initiator_socket[DDR_id]->bind(ddr->socket);
        dma->initiator_socket[GSM_id]->bind(gsm->socket);
        dma->initiator_socket[VCore_id]->bind(vcore->socket);
    }
    bool get_direct_mem_ptr(tlm::tlm_generic_payload& trans, tlm::tlm_dmi& dmi_data) {
        uint64_t addr = trans.get_address();
        // 根据地址范围转发到对应的模块
        if (addr >= DDR_BASE_ADDR && addr < DDR_BASE_ADDR + DDR_SIZE) {
            // 转发到 DDR
            return (*(dma->initiator_socket[DDR_id]))->get_direct_mem_ptr(trans, dmi_data);
        }
        else if (addr >= GSM_BASE_ADDR && addr < GSM_BASE_ADDR + GSM_SIZE) {
            // 转发到 GSM
            return (*(dma->initiator_socket[GSM_id]))->get_direct_mem_ptr(trans, dmi_data);
        }
        else if (addr >= VCORE_BASE_ADDR && addr < VCORE_BASE_ADDR + VCORE_SIZE) {
            // 转发到 VCore
            return (*(dma->initiator_socket[VCore_id]))->get_direct_mem_ptr(trans, dmi_data);
        }
        
        return false;  // 如果地址不在任何有效范围内
    }

    void b_transport(tlm::tlm_generic_payload& trans, sc_time& delay) {
        initiator_socket->b_transport(trans, delay);
    }
    //析构函数
    ~Soc() {
        delete dma;
        delete ddr;
        delete gsm;
        delete vcore;
    }

};

#endif
