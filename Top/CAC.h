#ifndef CAC_H
#define CAC_H

#include "const.h"

template<unsigned int N_TARGETS>
struct CAC : sc_module
{
    tlm_utils::simple_target_socket<CAC> target_socket;
    tlm_utils::simple_initiator_socket_tagged<CAC>* mem_ctr_socket[N_TARGETS];

    SC_CTOR(CAC) : 
        target_socket("target_socket") 
    {
        target_socket.register_b_transport(this, &CAC::b_transport);
        target_socket.register_get_direct_mem_ptr(this, &CAC::get_direct_mem_ptr);

        for (unsigned int i = 0; i < N_TARGETS; i++) {
            char txt[20];
            sprintf(txt, "socket_%d", i);
            mem_ctr_socket[i] = new tlm_utils::simple_initiator_socket_tagged<CAC>(txt);
            mem_ctr_socket[i]->register_invalidate_direct_mem_ptr(this, &CAC::invalidate_direct_mem_ptr, i);
        }
    }
    //阻塞传输方法
    virtual void b_transport( tlm::tlm_generic_payload& trans, sc_time& delay )
    {
        sc_dt::uint64 address = trans.get_address();
        sc_dt::uint64 masked_address;
        unsigned int target_nr = decode_address(address, masked_address);

        trans.set_address(masked_address);
        (*mem_ctr_socket[target_nr])->b_transport(trans, delay);
    }
    //DMI请求方法
    virtual bool get_direct_mem_ptr(tlm::tlm_generic_payload& trans, tlm::tlm_dmi& dmi_data) {
        sc_dt::uint64 addr = trans.get_address();
        sc_dt::uint64 masked_addr;
        unsigned int target_id = decode_address(addr, masked_addr);
        trans.set_address(masked_addr);
        if (target_id < N_TARGETS) {
            return (*mem_ctr_socket[target_id])->get_direct_mem_ptr(trans, dmi_data);
        } else {
            SC_REPORT_ERROR("CAC", "Invalid target ID in get_direct_mem_ptr");
            return false;
        }
    }

    virtual void invalidate_direct_mem_ptr(int id, sc_dt::uint64 start_range, sc_dt::uint64 end_range)
    {
        sc_dt::uint64 bw_start_range = compose_address(id, start_range);
        sc_dt::uint64 bw_end_range = compose_address(id, end_range);
        target_socket->invalidate_direct_mem_ptr(bw_start_range, bw_end_range);
    }

    inline unsigned int decode_address(sc_dt::uint64 addr, sc_dt::uint64& masked_addr) {
        if (addr >= DDR_BASE_ADDR && addr < DDR_BASE_ADDR + DDR_SIZE) {
            masked_addr = addr - DDR_BASE_ADDR;
            return 0;  // DDR
        } else if (addr >= GSM_BASE_ADDR && addr < GSM_BASE_ADDR + GSM_SIZE) {
            masked_addr = addr - GSM_BASE_ADDR;
            return 1;  // GSM
        } else if (addr >= VCORE_BASE_ADDR && addr < VCORE_BASE_ADDR + VCORE_SIZE) {
            masked_addr = addr - VCORE_BASE_ADDR;
            return 2;  // VCore
        }
        cout << "EEOR adder : " << addr << endl;
        SC_REPORT_ERROR("CAC", "Address out of range");
        sc_stop(); // 或者 throw std::runtime_error("Address out of range");
        return 0;
    }

    inline sc_dt::uint64 compose_address( unsigned int target_nr, sc_dt::uint64 address)
    {
        return (target_nr << 8) | (address & 0xFF);
    }
};

#endif
