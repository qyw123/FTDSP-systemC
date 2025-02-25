#ifndef TOP_H
#define TOP_H

#include "../GEMM/mem_ctr.h"
#include "CAC.h"
#include "DDR.h"
#include "GSM.h"
#include "VCore.h"
#include "vpu_tlm.h"
#include "vpe_tlm.h"
#include "mac_tlm.h"

using DataType = double;

SC_MODULE(Top) {
    Mem_ctr<DataType>* mem_ctr;
    CAC<3>* cac;
    DDR<DataType>* ddr;
    GSM<DataType>* gsm;
    VCore<DataType>* vcore;
    // Instantiate modules
    //Initiator<DataType>* initiator;
    VPU_TLM<16>* vpu; // Instantiate a VPU with 16 VPEs

    SC_CTOR(Top) {
        mem_ctr = new Mem_ctr<DataType>("Mem_ctr");
        cac = new CAC<3>("CAC");

        ddr = new DDR<DataType>("DDR");
        gsm = new GSM<DataType>("GSM");
        vcore = new VCore<DataType>("VCore");
        //initiator = new Initiator<DataType>("Initiator");
        vpu = new VPU_TLM<16> ("VPU_TLM");

        mem_ctr->socket.bind(cac->target_socket);
        cac->mem_ctr_socket[0]->bind(ddr->socket);
        cac->mem_ctr_socket[1]->bind(gsm->socket);
        cac->mem_ctr_socket[2]->bind(vcore->target_socket);
        vcore->initiator_socket.bind(vpu->target_socket);

        // Instantiate VPEs
        VPE_TLM<4>* vpes[16];
        for (int i = 0; i < 16; ++i) {
            char name[20];
            sprintf(name, "VPE_%d", i);
            vpes[i] = new VPE_TLM<4>(name);
        }

        // Instantiate MACs
        MAC_TLM<double>* macs[64];
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
