#ifndef VPU_TLM_H
#define VPU_TLM_H

#include "../util/const.h"

template <int N_VPE=16>
SC_MODULE(VPU_TLM) {
    // Target socket for receiving requests
    tlm_utils::simple_target_socket<VPU_TLM> target_socket;

    // Tagged initiator sockets for connecting to multiple VPE units
    tlm_utils::simple_initiator_socket_tagged<VPU_TLM>* initiator_sockets[N_VPE];

    SC_CTOR(VPU_TLM)
        : target_socket("target_socket") {
        target_socket.register_b_transport(this, &VPU_TLM::b_transport);

        // Initialize each initiator socket
        for (int i = 0; i < N_VPE; ++i) {
            char name[20];
            sprintf(name, "initiator_socket_%d", i);
            initiator_sockets[i] = new tlm_utils::simple_initiator_socket_tagged<VPU_TLM>(name);
        }
    }

    ~VPU_TLM() {
        for (int i = 0; i < N_VPE; ++i) {
            delete initiator_sockets[i];
        }
    }

    virtual void b_transport(tlm_generic_payload& trans, sc_time& delay) {
        // Decode target ID from transaction address
        sc_dt::uint64 address = trans.get_address();
        unsigned int target_id = decode_vpe_id(address);

        if (target_id >= N_VPE) {
            SC_REPORT_ERROR("VPU_TLM", "Invalid target ID");
            return;
        }
        // Forward the transaction to the selected VPE using the tagged socket
        (*initiator_sockets[target_id])->b_transport(trans, delay);
    }

    unsigned int decode_vpe_id(uint64_t address) {
        return static_cast<unsigned int>((address >> 2) & 0xF); // Assume bits [2:5] represent the VPE ID
    }
};

#endif
