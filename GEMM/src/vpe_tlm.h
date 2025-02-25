#ifndef VPE_TLM_H
#define VPE_TLM_H

#include "../util/const.h"

template <int N_MAC=4>
SC_MODULE(VPE_TLM) {
    // Target socket for receiving requests
    tlm_utils::simple_target_socket<VPE_TLM> target_socket;

    // Tagged initiator sockets for connecting to multiple MAC units
    tlm_utils::simple_initiator_socket_tagged<VPE_TLM>* initiator_sockets[N_MAC];

    SC_CTOR(VPE_TLM)
        : target_socket("target_socket") {
        target_socket.register_b_transport(this, &VPE_TLM::b_transport);

        // Initialize each initiator socket
        for (int i = 0; i < N_MAC; ++i) {
            char name[20];
            sprintf(name, "initiator_socket_%d", i);
            initiator_sockets[i] = new tlm_utils::simple_initiator_socket_tagged<VPE_TLM>(name);
        }
    }

    ~VPE_TLM() {
        for (int i = 0; i < N_MAC; ++i) {
            delete initiator_sockets[i];
        }
    }

    virtual void b_transport(tlm_generic_payload& trans, sc_time& delay) {
        // Decode target ID from transaction address
        sc_dt::uint64 address = trans.get_address();
        unsigned int target_id = decode_mac_id(address);
        //cout << "MAC id :" << target_id << endl;

        if (target_id >= N_MAC) {
            SC_REPORT_ERROR("VPE_TLM", "Invalid target ID");
            return;
        }
        // Forward the transaction to the selected MAC using the tagged socket
        (*initiator_sockets[target_id])->b_transport(trans, delay);
    }

    unsigned int decode_mac_id(uint64_t address) {
        return static_cast<unsigned int>(address & 0b11); // Assume the lower 2 bits represent the MAC ID
    }
};

#endif
