#ifndef VCORE_H
#define VCORE_H

#include "../util/const.h"
#include "../util/tools.h"
#include <systemc>
#include <tlm>
#include <tlm_utils/simple_target_socket.h>

using namespace sc_core;
using namespace sc_dt;
using namespace std;

// 首先定义MAC子模块
template<typename T>
class MAC : public sc_module {
public:
    // TLM接口
    tlm_utils::simple_target_socket<MAC> socket;
    
    SC_CTOR(MAC) : socket("socket") {
        socket.register_b_transport(this, &MAC::b_transport);
        result = T();  // 初始化结果
    }
        // MAC的计算函数
    void compute(T* source) {
        // 直接计算 a*b+c
        source[2] = source[0] * source[1] + source[2];//C = A*B + C
        //wait(MAC_LATENCY);  // 模拟计算延迟
    }

    // 处理传入的数据
    void b_transport(tlm::tlm_generic_payload& trans, sc_time& delay) {
        if (trans.get_command() == tlm::TLM_WRITE_COMMAND) {
            T* data = reinterpret_cast<T*>(trans.get_data_ptr());
            // 直接计算并存储结果
            compute(data);  
            trans.set_response_status(tlm::TLM_OK_RESPONSE);
            //delay += MAC_LATENCY;
            
        } else if (trans.get_command() == tlm::TLM_READ_COMMAND) {
            // 如果不需要读取操作，可以返回错误状态
            SC_REPORT_INFO("MAC", "Read command is not supported");
            trans.set_response_status(tlm::TLM_COMMAND_ERROR_RESPONSE);
        }
    }

private:
    T result;
};
//定义VPU子模块，VPU包含所有的MAC单元
template<typename T>
class VPU : public sc_module {
public:
    tlm_utils::simple_target_socket<VPU> socket;
    vector<tlm_utils::simple_initiator_socket<VPU>*> mac_sockets;  // 连接MAC的socket
    vector<MAC<T>*> mac_units;  // MAC单元数组
    SC_CTOR(VPU) : socket("socket") {
        socket.register_b_transport(this, &VPU::b_transport);
        // 创建MAC单元
        for(int i = 0; i < MAC_PER_VPU; i++) {
            string mac_name = "mac_" + to_string(i);
            MAC<T>* mac = new MAC<T>(mac_name.c_str());
            mac_units.push_back(mac);
            
            // 创建并绑定socket
            auto* init_socket = new tlm_utils::simple_initiator_socket<VPU>(
                ("init_socket_" + to_string(i)).c_str());
            mac_sockets.push_back(init_socket);
            init_socket->bind(mac->socket);
        }
    }
    void b_transport(tlm::tlm_generic_payload& trans, sc_time& delay) {
        if (trans.get_command() == tlm::TLM_WRITE_COMMAND) {
            //cout <<"VPU收到计算命令"<<endl;
            T* input_data = reinterpret_cast<T*>(trans.get_data_ptr());
            uint64_t total_elements = trans.get_data_length() / sizeof(T);
            uint64_t num_groups = total_elements / 3;  // 每组3个元素(A,B,C)
            
            // 使用的MAC数量不能超过数据组数和可用MAC数量的较小值
            uint64_t active_macs = min(num_groups, static_cast<uint64_t>(MAC_PER_VPU));
            
            // 并行启动MAC单元
            for (uint64_t mac_id = 0; mac_id < active_macs; mac_id++) {
                // 为每个MAC创建事务
                tlm::tlm_generic_payload mac_trans;
                mac_trans.set_command(tlm::TLM_WRITE_COMMAND);
                // 直接传递原始数据的指针位置
                mac_trans.set_data_ptr(reinterpret_cast<unsigned char*>(input_data + mac_id * 3));
                mac_trans.set_data_length(sizeof(T) * 3);  // 每组3个元素
                
                // 发送到MAC进行计算
                (*mac_sockets[mac_id])->b_transport(mac_trans, delay);
            }
            
            // 等待所有活跃的MAC完成
            wait(MAC_LATENCY);
            delay += MAC_LATENCY;
            //cout <<"VPU等待MAC完成计算,延迟为"<< MAC_LATENCY <<endl;
            
            trans.set_response_status(tlm::TLM_OK_RESPONSE);
        } else if (trans.get_command() == tlm::TLM_READ_COMMAND) {
            // 读取结果（如果需要）
            trans.set_response_status(tlm::TLM_OK_RESPONSE);
        }
    }
    ~VPU() {
        for(auto mac : mac_units) {
            delete mac;
        }
        for(auto socket : mac_sockets) {
            delete socket;
        }
    }
};
// 定义SM子模块（Scalar Memory）
template<typename T>
class SM : public sc_module {
public:
    tlm_utils::simple_target_socket<SM> socket;
    
    SC_CTOR(SM) : socket("socket") {
        socket.register_b_transport(this, &SM::b_transport);
        socket.register_get_direct_mem_ptr(this, &SM::get_direct_mem_ptr);
        memory.resize(SM_SIZE / sizeof(T));
    }

    bool get_direct_mem_ptr(tlm::tlm_generic_payload& trans, tlm::tlm_dmi& dmi_data) {
        dmi_data.set_start_address(SM_BASE_ADDR);
        dmi_data.set_end_address(SM_BASE_ADDR + SM_SIZE - 1);
        dmi_data.set_dmi_ptr(reinterpret_cast<unsigned char*>(memory.data()));
        dmi_data.set_read_latency(SM_LATENCY);
        dmi_data.set_write_latency(SM_LATENCY);
        dmi_data.allow_read_write();
        return true;
    }

    void b_transport(tlm::tlm_generic_payload& trans, sc_time& delay) {
        uint64_t addr = (trans.get_address() - SM_BASE_ADDR) / sizeof(T);
        
        if (trans.get_command() == tlm::TLM_WRITE_COMMAND) {
            T* data = reinterpret_cast<T*>(trans.get_data_ptr());
            memory[addr] = *data;
        } else {
            T* data = reinterpret_cast<T*>(trans.get_data_ptr());
            *data = memory[addr];
        }
        
        trans.set_response_status(tlm::TLM_OK_RESPONSE);
        wait(SM_LATENCY);
    }

private:
    vector<T> memory;
};

// 定义AM子模块（Array Memory）
template<typename T>
class AM : public sc_module {
public:
    tlm_utils::simple_target_socket<AM> socket;
    
    SC_CTOR(AM) : socket("socket") {
        socket.register_b_transport(this, &AM::b_transport);
        socket.register_get_direct_mem_ptr(this, &AM::get_direct_mem_ptr);
        memory.resize(AM_SIZE / sizeof(T));
    }

    bool get_direct_mem_ptr(tlm::tlm_generic_payload& trans, tlm::tlm_dmi& dmi_data) {
        dmi_data.set_start_address(AM_BASE_ADDR);
        dmi_data.set_end_address(AM_BASE_ADDR + AM_SIZE - 1);
        dmi_data.set_dmi_ptr(reinterpret_cast<unsigned char*>(memory.data()));
        dmi_data.set_read_latency(AM_LATENCY);
        dmi_data.set_write_latency(AM_LATENCY);
        dmi_data.allow_read_write();
        return true;
    }

    void b_transport(tlm::tlm_generic_payload& trans, sc_time& delay) {
        uint64_t addr = (trans.get_address() - AM_BASE_ADDR) / sizeof(T);
        
        if (trans.get_command() == tlm::TLM_WRITE_COMMAND) {
            T* data = reinterpret_cast<T*>(trans.get_data_ptr());
            memory[addr] = *data;
        } else {
            T* data = reinterpret_cast<T*>(trans.get_data_ptr());
            *data = memory[addr];
        }
        
        trans.set_response_status(tlm::TLM_OK_RESPONSE);
        wait(AM_LATENCY);
    }

private:
    vector<T> memory;
};

// 修改VCore以包含子模块
template<typename T>
class VCore : public sc_module {
public:
    tlm_utils::simple_target_socket<VCore> socket;
    tlm_utils::simple_initiator_socket<VCore> sm_socket;
    tlm_utils::simple_initiator_socket<VCore> am_socket;
    tlm_utils::simple_initiator_socket<VCore> vpu_socket;

    VPU<T>* vpu;
    SM<T>* sm;
    AM<T>* am;

    VCore(sc_module_name name) : sc_module(name), socket("socket"), 
                                sm_socket("sm_socket"), am_socket("am_socket") {
        socket.register_b_transport(this, &VCore::b_transport);
        socket.register_get_direct_mem_ptr(this, &VCore::get_direct_mem_ptr);
        //创建vpu
        vpu = new VPU<T>("vpu");
        vpu_socket.bind(vpu->socket);
        //创建sm和am
        sm = new SM<T>("sm");
        am = new AM<T>("am");
        sm_socket.bind(sm->socket);
        am_socket.bind(am->socket);
    }

    virtual bool get_direct_mem_ptr(tlm::tlm_generic_payload& trans, tlm::tlm_dmi& dmi_data) {
        uint64_t addr = trans.get_address();
        
        if (addr >= SM_BASE_ADDR && addr < SM_BASE_ADDR + SM_SIZE) {
            // 转发DMI请求到SM
            return sm_socket->get_direct_mem_ptr(trans, dmi_data);
        }
        else if (addr >= AM_BASE_ADDR && addr < AM_BASE_ADDR + AM_SIZE) {
            // 转发DMI请求到AM
            return am_socket->get_direct_mem_ptr(trans, dmi_data);
        }
        else if (addr >= VPU_BASE_ADDR && addr < VPU_BASE_ADDR + VPU_REGISTER_SIZE) {
            // VPU单元不支持DMI
            cout <<"VPU单元不支持DMI"<<endl;
            return false;
        }
        return false;
    }

    void b_transport(tlm::tlm_generic_payload& trans, sc_time& delay) {
        uint64_t addr = trans.get_address();
        
        // 如果不支持DMI，使用普通传输
        if (addr >= SM_BASE_ADDR && addr < SM_BASE_ADDR + SM_SIZE) {
            sm_socket->b_transport(trans, delay);
        }
        else if (addr >= AM_BASE_ADDR && addr < AM_BASE_ADDR + AM_SIZE) {
            am_socket->b_transport(trans, delay);
        }
        else if (addr >= VPU_BASE_ADDR && addr < VPU_BASE_ADDR + VPU_REGISTER_SIZE) {
            vpu_socket->b_transport(trans, delay);
        }
    }

    ~VCore() {
        delete vpu;
        delete sm;
        delete am;
    }
};

#endif
