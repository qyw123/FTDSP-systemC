#ifndef DMA_H
#define DMA_H

#include "../util/const.h"
#include "../util/tools.h"

template<unsigned int N_TARGETS>
class DMA : public sc_module
{
public:
    tlm_utils::simple_target_socket<DMA> target_socket;
    tlm_utils::simple_initiator_socket_tagged<DMA>* initiator_socket[N_TARGETS];
    SC_CTOR(DMA) : target_socket("target_socket") 
    {
        target_socket.register_b_transport(this, &DMA::b_transport);
        target_socket.register_get_direct_mem_ptr(this, &DMA::get_direct_mem_ptr);

        for (unsigned int i = 0; i < N_TARGETS; i++) {
            char txt[20];
            sprintf(txt, "socket_%d", i);
            initiator_socket[i] = new tlm_utils::simple_initiator_socket_tagged<DMA>(txt);
            initiator_socket[i]->register_invalidate_direct_mem_ptr(this, &DMA::invalidate_direct_mem_ptr, i);
        }
    }
    //阻塞传输方法
    virtual void b_transport( tlm::tlm_generic_payload& trans, sc_time& delay )
    {
        sc_dt::uint64 address = trans.get_address();
        sc_dt::uint64 masked_address;
        unsigned int target_nr = decode_address(address, masked_address);

        trans.set_address(address);
        (*initiator_socket[target_nr])->b_transport(trans, delay);
    }
    //DMI请求方法
    virtual bool get_direct_mem_ptr(tlm::tlm_generic_payload& trans, tlm::tlm_dmi& dmi_data) {
        sc_dt::uint64 addr = trans.get_address();
        sc_dt::uint64 masked_addr;
        
        // 解码地址，获取目标模块ID
        unsigned int target_nr = decode_address(addr, masked_addr);
        
        // 更新事务中的地址为目标模块的本地地址
        trans.set_address(addr);
        
        // 转发DMI请求到对应的目标模块
        return (*initiator_socket[target_nr])->get_direct_mem_ptr(trans, dmi_data);
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
            return DDR_id;  // DDR
        } else if (addr >= GSM_BASE_ADDR && addr < GSM_BASE_ADDR + GSM_SIZE) {
            masked_addr = addr - GSM_BASE_ADDR;
            return GSM_id;  // GSM
        } else if (addr >= VCORE_BASE_ADDR && addr < VCORE_BASE_ADDR + VCORE_SIZE) {
            masked_addr = addr - VCORE_BASE_ADDR;
            return VCore_id;  // VCore
        }
        cout << "addr:"<<addr<<endl;
        SC_REPORT_ERROR("DMA", "Address out of range");
        
        sc_stop(); // 或者 throw std::runtime_error("Address out of range");
        return 0;
    }

    inline sc_dt::uint64 compose_address(unsigned int target_nr, sc_dt::uint64 address)
    {
        // 假设使用高4位作为目标ID，低60位作为地址
        return ((sc_dt::uint64)target_nr << 60) | (address & 0x0FFFFFFFFFFFFFFFULL);
    }
};

#endif
