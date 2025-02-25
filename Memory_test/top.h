#ifndef TOP_H
#define TOP_H

#include "initiator.h"
#include "CAC.h"
#include "DDR.h"
#include "GSM.h"
#include "VCore.h"

using DataType = double;

SC_MODULE(Top) {
    Initiator<DataType>* initiator;
    CAC<3>* cac;
    DDR<DataType>* ddr;
    GSM<DataType>* gsm;
    VCore<DataType>* vcore;

    SC_CTOR(Top) {
        initiator = new Initiator<DataType>("Initiator");
        cac = new CAC<3>("CAC");

        ddr = new DDR<DataType>("DDR");
        gsm = new GSM<DataType>("GSM");
        vcore = new VCore<DataType>("VCore");

        initiator->socket.bind(cac->target_socket);
        cac->initiator_socket[0]->bind(ddr->socket);
        cac->initiator_socket[1]->bind(gsm->socket);
        cac->initiator_socket[2]->bind(vcore->socket);
    }
};

#endif
