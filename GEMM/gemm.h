#ifndef Gemm_H
#define Gemm_H

#include "./util/const.h"
#include "./util/tools.h"
#include "./util/ThreadPool.h"

template <typename T>
class MatrixBlockTransfer {
private:
    string transfer_name;
    sc_time init_time;
    
    void read_data(uint64_t addr, vector<T>& values, const tlm::tlm_dmi& dmi, unsigned int data_num) {
        dmi_utils::read_from_dmi(addr, values, dmi, data_num, transfer_name);
    }
    
    void write_data(uint64_t start_addr, uint64_t& end_addr, const vector<T>& values, 
                   const tlm::tlm_dmi& dmi, unsigned int data_num) {
        dmi_utils::write_to_dmi(start_addr, end_addr, values, dmi, data_num, transfer_name);
    }

public:
    MatrixBlockTransfer(const string& name, sc_time init_t) 
        : transfer_name(name), init_time(init_t) {}

    void transfer(
        uint64_t start_addr, 
        uint64_t& end_addr,
        uint64_t& next_block_start_addr,
        uint64_t matrix_start_addr,
        uint64_t matrix_end_addr,
        uint64_t target_start_addr,
        uint64_t& target_end_addr,
        int m_rows, 
        int m_cols, 
        int block_rows, 
        int block_cols,
        int& real_block_rows, 
        int& real_block_cols,
        const tlm::tlm_dmi& source_dmi, 
        const tlm::tlm_dmi& target_dmi,
        bool traverse_by_row,
        bool& rowloop_complete
    ) {
        // 参数验证
        if (block_rows <= 0 || block_cols <= 0 || m_rows <= 0 || m_cols <= 0) {
            cout << transfer_name << " ERROR: Invalid dimensions detected!\n"
                << "block_rows=" << block_rows << ", block_cols=" << block_cols << "\n"
                << "m_rows=" << m_rows << ", m_cols=" << m_cols << endl;
            sc_stop();
            return;
        }

        // ... [其余验证代码保持不变，只是在错误输出时加上transfer_name] ...
        // 确保起始地址在有效范围内
        if (start_addr > matrix_end_addr || start_addr < matrix_start_addr) {
            cout<<"Error transfer_matrixblock: "<<transfer_name<<endl;
            cout << "ERROR: Invalid start address: 0x" << hex << start_addr << endl;
            cout<<"matrix_start_addr:"<<matrix_start_addr<<",matrix_end_addr:"<<matrix_end_addr<<endl;
            sc_stop();
            return;
        }

        // 计算当前位置
        uint64_t offset = start_addr - matrix_start_addr;
        int start_row = (offset / sizeof(T)) / m_cols;
        int start_col = (offset / sizeof(T)) % m_cols;

        // 计算实际块大小
        real_block_rows = std::min(block_rows, m_rows - start_row);
        real_block_cols = std::min(block_cols, m_cols - start_col);

        // 验证计算结果
        if (real_block_rows <= 0 || real_block_cols <= 0) {
            cout <<(sc_time_stamp() - init_time)<< ":ERROR: Invalid block size calculated: [" << real_block_rows 
                << "," << real_block_cols << "]" << endl;
            cout<<"start_row:"<<start_row<<",start_col:"<<start_col<<endl;
            cout<<hex<<"start_addr:"<<start_addr<<",matrix_start_addr:"<<matrix_start_addr<<",matrix_end_addr:"<<matrix_end_addr<<endl;
            sc_stop();
            return;
        }

        vector<T> block_buffer(real_block_cols);
        for(int i = 0; i < real_block_rows; i++){
            read_data(start_addr + i * m_cols * sizeof(T), block_buffer, source_dmi, real_block_cols);
            //check_all_zero(block_buffer);
            write_data(target_start_addr + i * real_block_cols * sizeof(T), target_end_addr, block_buffer, target_dmi, real_block_cols);
        }
        end_addr = start_addr + (((real_block_rows-1) * m_cols+real_block_cols)) * sizeof(T) - 1;
        // 计算下一个块的起始地址
        if (traverse_by_row) {
            //先行后列循环
            next_block_start_addr = start_addr + (real_block_cols * sizeof(T));
            rowloop_complete = false;
            if ((start_col + real_block_cols) == m_cols) {
                next_block_start_addr = matrix_start_addr + 
                                    ((start_row + real_block_rows) * m_cols * sizeof(T));
                rowloop_complete = true;
            }
        } else {
            //先列后行循环
            next_block_start_addr = start_addr + (real_block_rows * m_cols * sizeof(T));
            rowloop_complete = false;
            if ((start_row + real_block_rows) == m_rows) {
                next_block_start_addr = matrix_start_addr + 
                                    (start_col + real_block_cols) * sizeof(T);
                rowloop_complete = true;
            }
        }

    }
    void transfer_back(
        uint64_t start_addr, 
        uint64_t target_start_addr,
        uint64_t target_end_addr,
        int am_rows,
        int am_cols,
        int ddr_rows,
        int ddr_cols,
        const tlm::tlm_dmi& source_dmi, 
        const tlm::tlm_dmi& target_dmi
    ){
        vector<T> block_buffer(am_cols);
        uint64_t end_addr;
        for(int i = 0; i<am_rows; i++){
            read_data(start_addr + i * am_cols * sizeof(T), block_buffer, source_dmi, am_cols);
            //check_all_zero(block_buffer);
            write_data(target_start_addr + i * ddr_cols * sizeof(T), end_addr, block_buffer, target_dmi, am_cols);
        }
        if(end_addr != target_end_addr){
            cout<<"Error transfer_back: "<<transfer_name<<endl;
            cout<<"start_addr:"<<start_addr<<",target_start_addr:"<<target_start_addr<<endl;
            cout<<"end_addr:"<<end_addr<<",target_end_addr:"<<target_end_addr<<endl;
            cout<<"am_rows:"<<am_rows<<",am_cols:"<<am_cols<<endl;
            cout<<"ddr_rows:"<<ddr_rows<<",ddr_cols:"<<ddr_cols<<endl;  
            sc_stop();
            return;
        }
        
    }
};

template <typename T>
struct Gemm : sc_module {
    tlm_utils::simple_initiator_socket<Gemm> socket;
    SC_CTOR(Gemm) : socket("socket") {
        socket.register_invalidate_direct_mem_ptr(this, &Gemm::invalidate_direct_mem_ptr);
        SC_THREAD(top);
        SC_THREAD(init_process);
        SC_THREAD(computing_process);
        SC_THREAD(writeback_C_process);
        SC_THREAD(output_result);
    }

    void setup_dmi(uint64_t base_addr, uint64_t size, tlm::tlm_dmi& dmi) {
        tlm::tlm_generic_payload trans;
        trans.set_address(base_addr);
        trans.set_command(tlm::TLM_READ_COMMAND);
        trans.set_data_length(sizeof(T));

        if (socket->get_direct_mem_ptr(trans, dmi)) {
            cout << "DMI setup successful for range: 0x" << hex
                 << dmi.get_start_address() << " - 0x" << dmi.get_end_address() << endl;
        } else {
            SC_REPORT_ERROR("Gemm", "DMI setup failed");
        }
    }

    void read_data(uint64_t addr, vector<T>& values, const tlm::tlm_dmi& dmi, unsigned int data_num) {
        dmi_utils::read_from_dmi(addr, values, dmi, data_num, "Gemm");
    }
    
    void write_data(uint64_t start_addr, uint64_t& end_addr, const vector<T>& values, 
                   const tlm::tlm_dmi& dmi, unsigned int data_num) {
        dmi_utils::write_to_dmi(start_addr, end_addr, values, dmi, data_num, "Gemm");
    }
    
    // //没有多线程
    // void kernel_mul(
    //     vector<T>& A_1D,
    //     vector<T>& B_1D,
    //     vector<T>& C_1D,
    //     const int rows_a,
    //     const int cols_a,
    //     const int rows_b,
    //     const int cols_b,
    //     const int rows_c,
    //     const int cols_c
    // ){
    //     // 定义事务和延迟
    //     tlm_generic_payload trans;
    //     sc_time delay = SC_ZERO_TIME;
    //     // //向量外积
    //     // // 遍历 A 的每一行M
    //     // for(int m = 0; i < rows_a; i++){
    //     //     //遍历A的每一列K,B的每一行K，根据规定，B一行的元素不会超过macs_per_vpu
    //     //     for(int k = 0; k < cols_a; k++){
    //     //         //C[m][start:end] += A[m][k]*B[k][start:end]
    //     //         //同时定义cols_b个trans事务，
    //     //     }
    //     // }
    //     //向量内积，现在的访存逻辑不对（B_1D的索引是跳跃式的）
    //     // 遍历 A 的每一行M
    //     for (int i = 0; i < rows_a; ++i) {
    //         // 遍历 A 的每一列K
    //         for (int k = 0; k < cols_a; ++k) {
    //             // 遍历 B 的每列N
    //             for (int j = 0; j < cols_b; ++j) {
    //                 T input[3] = {A_1D[i*cols_a+k], B_1D[k*cols_b+j], C_1D[i*cols_c+j]}; // 当前 A 和 B 元素，以及 C 的当前累加值
    //                 trans.set_command(TLM_WRITE_COMMAND);
    //                 trans.set_address(VCORE_BASE_ADDR + j % macs_per_vpu); // 路由到对应的 MAC 单元
    //                 trans.set_data_ptr(reinterpret_cast<unsigned char*>(input));
    //                 trans.set_data_length(sizeof(input));
    //                 trans.set_response_status(TLM_INCOMPLETE_RESPONSE);
    //                 socket->b_transport(trans, delay);
    //                 //wait(delay);
    //                 // 读取计算结果
    //                 trans.set_command(TLM_READ_COMMAND);
    //                 trans.set_address(VCORE_BASE_ADDR + j % macs_per_vpu); // 路由到对应的 MAC 单元
    //                 trans.set_data_ptr(reinterpret_cast<unsigned char*>(&C_1D[i*cols_c+j]));
    //                 trans.set_data_length(sizeof(T));
    //                 trans.set_response_status(TLM_INCOMPLETE_RESPONSE);
    //                 // 发送读事务
    //                 socket->b_transport(trans, delay);
    //                 //wait(delay);
    //             }
    //             wait(MAC_LATENCY);
    //         }
    //     }
    // }

    // 带线程池的矩阵乘法
    void kernel_mul(
        std::vector<T>& A_1D,
        std::vector<T>& B_1D,
        std::vector<T>& C_1D,
        const int rows_a,
        const int cols_a,
        const int rows_b,
        const int cols_b,
        const int rows_c,
        const int cols_c
    ) {

        // 创建一个线程池，线程数设置为 20
        ThreadPool pool(2);
        int count = 0;

        // 用于存储 future 对象，以便后续等待所有任务完成
        std::vector<std::future<void>> futures;
        cout << "start kernel_mul" << endl;
        // 遍历 A 的每一行 M
        for (int i = 0; i < rows_a; ++i) {
            // 遍历 B 的每一列 N
            for (int j = 0; j < cols_b; ++j) {
                // 提交任务到线程池
                futures.emplace_back(pool.enqueue([&, i, j]() {
                    // 遍历 A 的每一列 K
                    for (int k = 0; k < cols_a; ++k) {
                        T input[3] = {A_1D[i * cols_a + k], B_1D[k * cols_b + j], C_1D[i * cols_c + j]}; // 当前 A 和 B 元素，以及 C 的当前累加值
                        cout << "进入最内侧循环"<<endl;
                        count++;
                        cout << "count: "<<count<<endl; 
                        // 定义事务和延迟
                        tlm::tlm_generic_payload trans;
                        sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
                        trans.set_command(tlm::TLM_WRITE_COMMAND);
                        trans.set_address(VCORE_BASE_ADDR + j % macs_per_vpu); // 路由到对应的 MAC 单元
                        trans.set_data_ptr(reinterpret_cast<unsigned char*>(input));
                        trans.set_data_length(sizeof(input));
                        trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);

                        socket->b_transport(trans, delay);
                        //wait(delay);

                        // 读取计算结果
                        trans.set_command(tlm::TLM_READ_COMMAND);
                        trans.set_address(VCORE_BASE_ADDR + j % macs_per_vpu); // 路由到对应的 MAC 单元
                        trans.set_data_ptr(reinterpret_cast<unsigned char*>(&C_1D[i * cols_c + j]));
                        trans.set_data_length(sizeof(T));
                        trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);

                        // 发送读事务
                        socket->b_transport(trans, delay);
                        //wait(delay);
                        //cout << "结束最内侧循环"<<endl;
                        
                    }
                    wait(MAC_LATENCY);
                }));
            }
        }

        // 等待所有任务完成
        for (auto& future : futures) {
            future.get();
        }
    }
    //GEMM C=A*B+C 外积循环逻辑：
        //已知：A：M*K,B:K*N,C:M*N
        // for m = M/m_gsm_max
        //     for k = K/k_gsm_max
        //         A:m_gsm_max*k_gsm_max(DDR->GSM)
        //         for n = N/cu_max
        //             B:k_gsm_max*cu_max(DDR->AM)
        //             C:k_gsm_max*cu_max(DDR->AM)
        //             for sm = m_gsm_max/sm_max
        //                 A_sm:sm_max*k_gsm_max(GSM->SM)
        //                 C += A*B
        //             C:k_gsm_max*cu_max(AM->DDR)
    void top() {
        wait(init_ready);
        cout << (sc_time_stamp() - init_time) << "=====================初始化完成============================" << endl;
        int m,k,n,sm;
        
        // 初始化矩阵传输对象
        MatrixBlockTransfer<T> gsm_transfer("GSM_Transfer", init_time);
        MatrixBlockTransfer<T> sm_transfer("SM_Transfer", init_time);
        MatrixBlockTransfer<T> amB_transfer("AMB_Transfer", init_time);
        MatrixBlockTransfer<T> amC_transfer("AMC_Transfer", init_time);
        
        // 预计算循环次数
        int M_blocks = (A_rows + m_gsm_max - 1) / m_gsm_max;  // M方向的块数1
        int K_blocks = (A_cols + k_gsm_max - 1) / k_gsm_max;  // K方向的块数2
        int N_blocks = (B_cols + cu_max - 1) / cu_max;        // N方向的块数2
        cout<<"M_blocks: "<<M_blocks<<endl;
        cout<<"K_blocks: "<<K_blocks<<endl;
        cout<<"N_blocks: "<<N_blocks<<endl;


        // M方向循环 (处理MatrixA的行)
        for(m = 0; m < M_blocks && !M_complete; m++) {
            // 计算当前M块的实际大小
            int current_m_size = min(m_gsm_max, A_rows - m * m_gsm_max);
            for(k = 0; k < K_blocks && !K_complete; k++) {
                gsm_transfer.transfer(
                    A_addr[DDR_A][start],
                    A_addr[DDR_A][end],
                    A_next_addr[DDR_A][start],
                    Matrix_addr[A][start],
                    Matrix_addr[A][end],
                    A_addr[GSM][start],
                    A_addr[GSM][end],
                    A_rows, A_cols,
                    m_gsm_max, k_gsm_max,
                    A_GSM_size[row], A_GSM_size[col],
                    ddr_dmi, gsm_dmi,
                    true,
                    K_complete
                );
                // cout << (sc_time_stamp() - init_time)<< "==============输出A矩阵DDR->GSM======================="<<endl;
                // cout << "A_addr[DDR_A][start]: "<<A_addr[DDR_A][start]<<endl;
                // cout << "A_addr[DDR_A][end]: "<<A_addr[DDR_A][end]<<endl;
                // cout << "A_next_addr[DDR_A][start]: "<<A_next_addr[DDR_A][start]<<endl;
                // cout << "Matrix_addr[A][start]: "<<Matrix_addr[A][start]<<endl;
                // cout << "Matrix_addr[A][end]: "<<Matrix_addr[A][end]<<endl;
                // cout << "A_addr[GSM][start]: "<<A_addr[GSM][start]<<endl;
                // cout << "A_addr[GSM][end]: "<<A_addr[GSM][end]<<endl;
                // cout << "A_GSM_size[row]: "<<A_GSM_size[row]<<endl;
                // cout << "A_GSM_size[col]: "<<A_GSM_size[col]<<endl;
                // cout << "K_complete: "<<K_complete<<endl;

                // N方向循环 (处理MatrixB的列)
                for(n = 0; n < N_blocks && !N_complete; n++) {
                    // 计算当前N块的实际大小
                    // int current_n_size = min(cu_max, B_cols - n * cu_max);
                    // cout <<"n:"<<n<<endl;
                    // cout << "current_n_size: " << current_n_size << endl;
                    // 从DDR加载B矩阵块到AM
                    amB_transfer.transfer(
                        B_addr[DDR_BC][start],
                        B_addr[DDR_BC][end],
                        B_next_addr[DDR_BC][start],
                        Matrix_addr[B][start],
                        Matrix_addr[B][end],
                        B_addr[AM][start],
                        B_addr[AM][end],
                        B_rows, B_cols,
                        k_gsm_max, cu_max,
                        B_AM_size[row], B_AM_size[col],
                        ddr_dmi, vcore_dmi,
                        true,
                        N_complete
                    );
                    // cout << (sc_time_stamp() - init_time)<< "==============输出B矩阵DDR->AM======================="<<endl;
                    // cout << "B_addr[DDR_BC][start]: "<<B_addr[DDR_BC][start]<<endl;
                    // cout << "B_addr[DDR_BC][end]: "<<B_addr[DDR_BC][end]<<endl;
                    // cout << "B_next_addr[DDR_BC][start]: "<<B_next_addr[DDR_BC][start]<<endl;
                    // cout << "Matrix_addr[B][start]: "<<Matrix_addr[B][start]<<endl;
                    // cout << "Matrix_addr[B][end]: "<<Matrix_addr[B][end]<<endl;
                    // cout << "B_addr[AM][start]: "<<B_addr[AM][start]<<endl;
                    // cout << "B_addr[AM][end]: "<<B_addr[AM][end]<<endl;
                    // cout << "B_AM_size[row]: "<<B_AM_size[row]<<endl;
                    // cout << "B_AM_size[col]: "<<B_AM_size[col]<<endl;
                    // cout << "N_complete: "<<N_complete<<endl;
                    // 从DDR加载C矩阵块到AM
                    C_addr[AM][start] =B_addr[AM][end] + 1;
                    // cout <<"============检查C矩阵DDR数据是否为0============="<<endl;
                    // DDR_data.clear();
                    // read_from_dmi(C_addr[DDR_BC][start], DDR_data, ddr_dmi, C_rows*C_cols);
                    // check_all_zero(DDR_data);
                    amC_transfer.transfer(
                        C_addr[DDR_BC][start],
                        C_addr[DDR_BC][end],
                        C_next_addr[DDR_BC][start],
                        Matrix_addr[C][start],
                        Matrix_addr[C][end],
                        C_addr[AM][start],
                        C_addr[AM][end],
                        C_rows, C_cols,
                        m_gsm_max, cu_max,
                        C_AM_size[row], C_AM_size[col],
                        ddr_dmi, vcore_dmi,
                        true,
                        N_complete
                    );
                    // cout << (sc_time_stamp() - init_time)<< "==============输出C矩阵DDR->AM======================="<<endl;
                    // cout << "C_addr[DDR_BC][start]: "<<C_addr[DDR_BC][start]<<endl;
                    // cout << "C_addr[DDR_BC][end]: "<<C_addr[DDR_BC][end]<<endl;
                    // cout << "C_next_addr[DDR_BC][start]: "<<C_next_addr[DDR_BC][start]<<endl;
                    // cout << "Matrix_addr[C][start]: "<<Matrix_addr[C][start]<<endl;
                    // cout << "Matrix_addr[C][end]: "<<Matrix_addr[C][end]<<endl;
                    // cout << "C_addr[AM][start]: "<<C_addr[AM][start]<<endl;
                    // cout << "C_addr[AM][end]: "<<C_addr[AM][end]<<endl;
                    // cout << "C_AM_size[row]: "<<C_AM_size[row]<<endl;
                    // cout << "C_AM_size[col]: "<<C_AM_size[col]<<endl;
                    // cout << "N_complete: "<<N_complete<<endl;
                    // SM方向循环 (处理GSM中A矩阵的行分块)
                    int SM_blocks = (current_m_size + sm_max - 1) / sm_max;
                    // cout<<"SM_blocks: "<<SM_blocks<<endl;
                    A_addr[GSMSM][start] = A_addr[GSM][start];
                    for(sm = 0; sm < SM_blocks; sm++) {
                        sm_transfer.transfer(
                            A_addr[GSMSM][start],
                            A_addr[GSMSM][end],
                            A_next_addr[GSMSM][start],
                            A_addr[GSM][start],
                            A_addr[GSM][end],
                            A_addr[SM][start],
                            A_addr[SM][end],
                            A_GSM_size[row], A_GSM_size[col],
                            sm_max, k_gsm_max,
                            A_SM_size[row], A_SM_size[col],
                            gsm_dmi, vcore_dmi,
                            false, //列循环
                            SM_complete
                        );
                        // cout << (sc_time_stamp() - init_time)<< "==============输出A矩阵GSM->SM======================="<<endl;
                        // cout << "A_addr[GSMSM][start]: "<<A_addr[GSMSM][start]<<endl;
                        // cout << "A_addr[GSMSM][end]: "<<A_addr[GSMSM][end]<<endl;
                        // cout << "A_next_addr[GSMSM][start]: "<<A_next_addr[GSMSM][start]<<endl;
                        // cout << "A_addr[GSM][start]: "<<A_addr[GSM][start]<<endl;
                        // cout << "A_addr[GSM][end]: "<<A_addr[GSM][end]<<endl;
                        // cout << "A_addr[SM][start]: "<<A_addr[SM][start]<<endl;
                        // cout << "A_addr[SM][end]: "<<A_addr[SM][end]<<endl;
                        // cout << "A_SM_size[row]: "<<A_SM_size[row]<<endl;
                        // cout << "A_SM_size[col]: "<<A_SM_size[col]<<endl;
                        // cout << "SM_complete: "<<SM_complete<<endl;
                        A_GSM_addr_flag = A_addr[GSMSM][start];
                        A_addr[GSMSM][start] = A_next_addr[GSMSM][start];
                        
                        // 执行计算
                        compute_ready.notify();
                        wait(kernel_com_finished);
                        cout << (sc_time_stamp() - init_time)
                        << " m:"<<m<< " k:"<<k<< " n:"<<n<< " sm:"<<sm<<  " kernel计算完成" << endl;
                        // cout <<"M_complete: "<<M_complete<<endl;
                        // cout <<"K_complete: "<<K_complete<<endl;
                        // cout <<"N_complete: "<<N_complete<<endl;
                        // cout <<"SM_complete: "<<SM_complete<<endl;
                    }
                    // 等待计算结果写回
                    C_write_back_ready.notify();
                    wait(C_writeback_done);

                    B_addr[DDR_BC][start] = B_next_addr[DDR_BC][start];
                    C_addr[DDR_BC][start] = C_next_addr[DDR_BC][start];
                }
                if(!K_complete){
                        //K方向循环未完成，N方向重新开始循环
                        N_complete= false;
                        //C分块起始地址回到本行第一个分块，重新开始N放方向循环
                        //B_addr[DDR_BC][start] = B_addr[DDR_BC][start] - B_AM_size[row]*B_cols*sizeof(T);
                        C_addr[DDR_BC][start] = C_addr[DDR_BC][start] - C_AM_size[row]*C_cols*sizeof(T);
                }
                A_addr[DDR_A][start] = A_next_addr[DDR_A][start];
                //B_addr[DDR_BC][start] = B_next_addr[DDR_BC][start];
                
            }
            if(!M_complete){
                //M方向循环未完成，K方向重新开始循环
                K_complete = false;
                N_complete = false;
                //B分块起始地址回到整个矩阵的第一个分块
                B_addr[DDR_BC][start] = Matrix_addr[B][start];
                //C_addr[DDR_BC][start] = Matrix_addr[C][start];
            }

        }
        
        //MatrixA_DDR_empty = true;
        cout << (sc_time_stamp() - init_time)<< "=====================所有计算完成============================" << endl;
        computation_done.notify();
        wait(output_result_done);
        sc_stop();
    }
    void init_process(){
        //获取输入矩阵大小
        record_matrix_shape<T>(matrixA_file_path, A_rows, A_cols);
        record_matrix_shape<T>(matrixB_file_path, B_rows, B_cols);
        C_rows = A_rows;
        C_cols = B_cols;
        //record_matrix_shape<T>(matrixC_file_path, C_rows, C_cols);
        if(A_cols != B_rows){
            SC_REPORT_ERROR("Gemm", "A_cols != B_rows, 矩阵A的列数与矩阵B的行数不匹配");
            sc_stop();
            return;
        }
        cout << "A_rows:"<<A_rows<< ",A_cols:"<<A_cols<<endl;
        cout << "B_rows:"<<B_rows<< ",B_cols:"<<B_cols<<endl;
        cout << "C_rows:"<<C_rows<< ",C_cols:"<<C_cols<<endl;   

        // Setup DMI for DDR and GSM
        setup_dmi(DDR_BASE_ADDR, DDR_SIZE, ddr_dmi);
        setup_dmi(GSM_BASE_ADDR, GSM_SIZE, gsm_dmi);
        setup_dmi(VCORE_BASE_ADDR, VCORE_SIZE, vcore_dmi);
        // 初始化矩阵在DDR中的起始结束地址
        Matrix_addr[A][start] = DDR_BASE_ADDR;
        Matrix_addr[A][end] = DDR_BASE_ADDR + A_rows * A_cols * sizeof(T)-1;
        Matrix_addr[B][start] = DDR_BASE_ADDR + A_rows * A_cols * sizeof(T);
        Matrix_addr[B][end] = DDR_BASE_ADDR + (A_rows * A_cols + B_rows * B_cols) * sizeof(T)-1;
        Matrix_addr[C][start] = DDR_BASE_ADDR + (A_rows * A_cols + B_rows * B_cols) * sizeof(T);
        Matrix_addr[C][end] = DDR_BASE_ADDR + (A_rows * A_cols + B_rows * B_cols + C_rows * C_cols) * sizeof(T)-1;
        A_addr[DDR_A][start] = Matrix_addr[A][start];
        B_addr[DDR_BC][start] = Matrix_addr[B][start];
        C_addr[DDR_BC][start] = Matrix_addr[C][start];
        A_addr[GSM][start] = GSM_BASE_ADDR;
        A_addr[GSMSM][start] = GSM_BASE_ADDR;
        A_addr[SM][start] = SM_BASE_ADDR;
        B_addr[AM][start] = AM_BASE_ADDR ;
        
        //初始化

        //cout<<"A_addr[DDR_A][start]:"<<A_addr[DDR_A][start]<<endl;
        // Write batch data to DDR
        cout << "=== MatrixA Write to DDR ===" << endl;
        load_from_file<T>(DDR_data, matrixA_file_path);
        int MatrixA_num = A_rows * A_cols;
        write_data(A_addr[DDR_A][start], A_addr[DDR_A][end], DDR_data, ddr_dmi, MatrixA_num);
        cout <<"A_addr[DDR_A][start]:"<<A_addr[DDR_A][start]<<endl;
        cout << " After write to dmi MatrixA end addr: " << A_addr[DDR_A][end] << endl;

        cout << "=== MatrixB Write to DDR ===" << endl;
        DDR_data.clear();
        load_from_file<T>(DDR_data, matrixB_file_path);
        check_all_zero(DDR_data);
        int MatrixB_num = B_rows * B_cols;
        write_data(B_addr[DDR_BC][start], B_addr[DDR_BC][end], DDR_data, ddr_dmi, MatrixB_num);
        cout <<"B_addr[DDR_BC][start]:"<<B_addr[DDR_BC][start]<<endl;
        cout << "MatrixB end addr: " << B_addr[DDR_BC][end] << endl;

        cout << "=== MatrixC Write to DDR ===" << endl;
        DDR_data.clear();
        load_from_file<T>(DDR_data, matrixC_file_path);
        int MatrixC_num = C_rows * C_cols;
        write_data(C_addr[DDR_BC][start], C_addr[DDR_BC][end], DDR_data, ddr_dmi, MatrixC_num);
        cout <<"C_addr[DDR_BC][start]:"<<C_addr[DDR_BC][start]<<endl;
        cout << "MatrixC end addr: " << C_addr[DDR_BC][end] << endl;

        //双缓冲结构暂不实现
        //初始化时间
        init_time = sc_time_stamp();
        
        //初始化第一个分块的地址
        // 初始化地址

        //sc_stop();
        init_ready.notify();
    }
    void computing_process() {
        while(true) {
            // 等待新的计算任务
            wait(compute_ready);
            
            // 创建临时矩阵存储当前块的数据
            // vector<vector<T>> A_block(A_SM_size[row], vector<T>(A_SM_size[col]));
            // vector<vector<T>> B_block(B_AM_size[row], vector<T>(B_AM_size[col]));
            // vector<vector<T>> C_block(A_SM_size[row], vector<T>(B_AM_size[col]));
            vector<T> C_block_1D(A_SM_size[row]*B_AM_size[col]);
            vector<T> A_block_1D(A_SM_size[row]*A_SM_size[col]);
            vector<T> B_block_1D(B_AM_size[row]*B_AM_size[col]);
            // 从SM读取A块
            read_data(A_addr[SM][start], A_block_1D, vcore_dmi, A_SM_size[row]*A_SM_size[col]);
            //从AM读取B块
            read_data(B_addr[AM][start], B_block_1D, vcore_dmi, B_AM_size[row]*B_AM_size[col]);
            //从AM读取C块
            // 计算C块在AM中的偏移
            uint64_t C_offset = (A_GSM_addr_flag - A_addr[GSM][start]) / ( A_GSM_size[col]) * C_AM_size[col];
            read_data(C_addr[AM][start]+C_offset, C_block_1D, vcore_dmi, A_SM_size[row]*B_AM_size[col]);

            // convertTo2D(A_block_1D, A_block, A_SM_size[row], A_SM_size[col]);
            // convertTo2D(B_block_1D, B_block, B_AM_size[row], B_AM_size[col]);
            // convertTo2D(C_block_1D, C_block, A_SM_size[row], B_AM_size[col]);
            //cout << (sc_time_stamp() - init_time)<< "============准备开始一次kernel计算============="<<endl;
            // 执行矩阵乘法计算
            kernel_mul(
                A_block_1D, B_block_1D, C_block_1D,
                A_SM_size[row], A_SM_size[col],
                B_AM_size[row], B_AM_size[col],
                A_SM_size[row], B_AM_size[col]
            );
            cout << "kernel_mul done" << endl;
            //cout << (sc_time_stamp() - init_time)<< "============kernel计算完成============="<<endl;
            //convertTo1D(C_block, C_block_1D);
            // 将计算结果写回AM
            // 计算C块在AM中的偏移  
            uint64_t temp_end_addr;  // 临时变量存储写入结束地址
            write_data(C_addr[AM][start]+C_offset, temp_end_addr, C_block_1D, vcore_dmi, A_SM_size[row]*B_AM_size[col]);
            // cout<< (sc_time_stamp() - init_time)<<"============C矩阵写回AM============="<<endl;
            // cout<<"C_offset:"<<C_offset<<endl;
            // cout<<"end_addr:"<<temp_end_addr<<endl;
            // 通知计算完成
            kernel_com_finished.notify();
            //cout << "kernel计算完成" << endl;
            
        }
    }
    void writeback_C_process() {
        MatrixBlockTransfer<T> amCback_transfer("AMCback_Transfer", init_time);
        while(true) {
            wait(C_write_back_ready);
            
            bool writeback_end;
            amCback_transfer.transfer_back(
                C_addr[AM][start],
                C_addr[DDR_BC][start],
                C_addr[DDR_BC][end],
                C_AM_size[row],
                C_AM_size[col],
                C_rows,
                C_cols,
                vcore_dmi,
                ddr_dmi
            );
            // cout << "=====================C矩阵写回DDR======================"<<endl;
            // cout << "C_addr[AM][start]:"<<C_addr[AM][start]<<"\n"
            //     << "C_addr[AM][end]:"<<C_addr[AM][end]<<"\n"
            //     << "C_addr[DDR_BC][start]:"<<C_addr[DDR_BC][start]<<"\n"
            //     << "C_addr[DDR_BC][end]:"<<C_addr[DDR_BC][end]<<"\n"
            //     << "C_AM_size[row]:"<<C_AM_size[row]<<"\n"
            //     << "C_AM_size[col]:"<<C_AM_size[col]<< endl;
            C_writeback_done.notify();
        }
    }
    void output_result() {
        wait(computation_done);  // 等待计算完成
        cout << "\n=== C矩阵读取调试信息 ===" << endl;
        cout << "MatrixC_DDR_start_addr: 0x" << hex << Matrix_addr[C][start] << endl;
        cout << "MatrixC_DDR_end_addr: 0x" << hex << Matrix_addr[C][end] << endl;
        cout << "C矩阵大小: " << dec << C_rows << " x " << C_cols << endl;

        // 在读取数据后添加数据检查
        uint64_t C_size = C_rows * C_cols;
        vector<T> result_C(C_size); 
        
        cout << "\n=====================开始输出结果============================" << endl;
        cout << (sc_time_stamp() - init_time) << " 矩阵形状信息：" << endl;
        cout << "MatrixA: [" << A_rows << ", " << A_cols << "]" << endl;
        cout << "MatrixB: [" << B_rows << ", " << B_cols << "]" << endl;
        cout << "MatrixC: [" << C_rows << ", " << C_cols << "]" << endl;

        try {
            // 读取整个C矩阵
            read_data(
                Matrix_addr[C][start],
                result_C,
                ddr_dmi,
                C_size
            );
            write_matrix_in_file<T>(result_C, "./data512-4608-784/MatrixC_output.txt", C_rows, C_cols);
            cout << "=====================输出完成,结果保存到MatrixC_output.txt============================" << endl;
            output_result_done.notify();

        } catch (const std::exception& e) {
            cout << "ERROR in output_result: " << e.what() << endl;
            sc_stop();
            return;
        }
    }
    virtual void invalidate_direct_mem_ptr(sc_dt::uint64 start_range, sc_dt::uint64 end_range) {
        cout << "DMI invalidated. Range: " << hex << start_range << " - " << end_range << endl;
    }
    
    public:
        tlm::tlm_dmi ddr_dmi, gsm_dmi, vcore_dmi;
        sc_event init_ready,compute_ready,C_write_back_ready;
        sc_event C_writeback_done,kernel_com_finished;  
        sc_event computation_done, output_result_done; 

        vector<T> DDR_data;
        
        sc_time init_time;
        int GSM_row_cnt = 0;
        enum Matrix_flag{A=0,B=1,C=2};
        enum flag_A {DDR_A=0,GSM=1,GSMSM=2,SM=3};
        enum flag_BC{DDR_BC=0,AM=1};
        enum flag_start_end{start=0,end=1};
        enum flag_size{row=0,col=1};

        //初始化,实际矩阵块大小0_row,1_col
        array<int, 2> A_GSM_size = {0, 0};
        array<int, 2> A_SM_size = {0, 0};
        array<int, 2> B_AM_size = {0, 0};
        array<int, 2> C_AM_size = {0, 0};
        array<int, 2> C_DDR_size = {0, 0}; // 新增：C矩阵写回DDR的大小记录
        //实际矩阵大小
        int A_rows = 0, A_cols = 0, B_rows = 0, B_cols = 0, C_rows = A_rows, C_cols = B_cols;

        // 用于跟踪当前处理的块位置
        int current_M_block = 0;
        int current_K_block = 0;
        int current_N_block = 0;
        bool M_complete = false;
        bool K_complete = false;
        bool N_complete = false;
        bool SM_complete = false;

        uint64_t A_GSM_addr_flag;
        //实际DDR中矩阵起始结束地址
        array<array<uint64_t, 2>, 3> Matrix_addr = {{{0, 0}, {0, 0}, {0, 0}}};
        //分块矩阵的起始结束地址
        array<array<uint64_t, 2>, 4> A_addr = {{{0, 0}, {0, 0}, {0, 0}, {0, 0}}};
        array<array<uint64_t, 2>, 2> B_addr = {{{0, 0}, {0, 0}}};
        array<array<uint64_t, 2>, 2> C_addr = {{{0, 0}, {0, 0}}};   
        array<array<uint64_t, 2>, 3> A_next_addr = {{{0, 0}, {0, 0}, {0, 0}}};
        array<array<uint64_t, 2>, 2> B_next_addr = {{{0, 0}, {0, 0}}};
        array<array<uint64_t, 2>, 2> C_next_addr = {{{0, 0}, {0, 0}}}; 
        uint64_t empty_addr = 0;
        //输入数据
        string matrixA_file_path = "./data512-4608-784/matrixA_input.txt";
        string matrixB_file_path = "./data512-4608-784/matrixB_input.txt"; 
        string matrixC_file_path = "./data512-4608-784/matrixC_input.txt";
};
#endif
