#ifndef TOP_H
#define TOP_H

#include "mem_ctr.h"
#include "CAC.h"
#include "DDR.h"
#include "GSM.h"
#include "VCore.h"

using DataType = double;

SC_MODULE(Top) {
    Mem_ctr<DataType>* mem_ctr;
    CAC<3>* cac;
    DDR<DataType>* ddr;
    GSM<DataType>* gsm;
    VCore<DataType>* vcore;

    SC_CTOR(Top) {
        mem_ctr = new Mem_ctr<DataType>("Mem_ctr");
        cac = new CAC<3>("CAC");

        ddr = new DDR<DataType>("DDR");
        gsm = new GSM<DataType>("GSM");
        vcore = new VCore<DataType>("VCore");

        mem_ctr->socket.bind(cac->target_socket);
        cac->mem_ctr_socket[0]->bind(ddr->socket);
        cac->mem_ctr_socket[1]->bind(gsm->socket);
        cac->mem_ctr_socket[2]->bind(vcore->socket);
    }
};

#endif
