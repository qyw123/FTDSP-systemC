#ifndef MAC_TLM_H
#define MAC_TLM_H

#include "../util/const.h"

template <typename T>
SC_MODULE(MAC_TLM) {
    tlm_utils::simple_target_socket<MAC_TLM> target_socket;


    SC_CTOR(MAC_TLM) : target_socket("target_socket") {
        target_socket.register_b_transport(this, &MAC_TLM::b_transport);
    }

public:
    T accumulator; // 存储累加结果

    void b_transport(tlm_generic_payload& trans, sc_time& delay) {
        if (trans.get_command() == TLM_WRITE_COMMAND) {
            // 提取输入数据
            T* data_ptr = reinterpret_cast<T*>(trans.get_data_ptr());
            T multiplicand = data_ptr[0];
            T multiplier = data_ptr[1];
            T addend = data_ptr[2];

            // 执行乘加操作
            accumulator = multiplicand * multiplier + addend;

            //std::cout << "accumulator = " << accumulator << std::endl;

            // 设置响应状态为 OK，表示操作完成
            trans.set_response_status(TLM_OK_RESPONSE);
            //delay += sc_time(10, SC_NS); // 模拟计算延迟
        } 
        else if (trans.get_command() == TLM_READ_COMMAND) {
            // 返回累加器的结果
            T* result_ptr = reinterpret_cast<T*>(trans.get_data_ptr());
            *result_ptr = accumulator; // 将计算结果写回传输的数据指针

            //std::cout << "Returning result = " << accumulator << std::endl;

            // 设置响应状态为 OK
            trans.set_response_status(TLM_OK_RESPONSE);
        }
        wait(delay); // 模拟延迟
    }
};
#endif
