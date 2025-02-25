#ifndef Gemm_H
#define Gemm_H

#include "./util/const.h"
#include "./util/tools.h"

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
        // // 参数验证
        // if (block_rows <= 0 || block_cols <= 0 || m_rows <= 0 || m_cols <= 0) {
        //     cout << transfer_name << " ERROR: Invalid dimensions detected!\n"
        //         << "block_rows=" << block_rows << ", block_cols=" << block_cols << "\n"
        //         << "m_rows=" << m_rows << ", m_cols=" << m_cols << endl;
        //     sc_stop();
        //     return;
        // }

        // ... [其余验证代码保持不变，只是在错误输出时加上transfer_name] ...
        // 确保起始地址在有效范围内
        // if (start_addr > matrix_end_addr || start_addr < matrix_start_addr) {
        //     cout<<"Error transfer_matrixblock: "<<transfer_name<<endl;
        //     cout << "ERROR: Invalid start address: 0x" << hex << start_addr << endl;
        //     cout<<"matrix_start_addr:"<<matrix_start_addr<<",matrix_end_addr:"<<matrix_end_addr<<endl;
        //     sc_stop();
        //     return;
        // }

        // 计算当前位置
        uint64_t offset = start_addr - matrix_start_addr;
        int start_row = (offset / sizeof(T)) / m_cols;
        int start_col = (offset / sizeof(T)) % m_cols;

        // 计算实际块大小
        real_block_rows = std::min(block_rows, m_rows - start_row);
        real_block_cols = std::min(block_cols, m_cols - start_col);

        // 验证计算结果
        // if (real_block_rows <= 0 || real_block_cols <= 0) {
        //     cout <<(sc_time_stamp() - init_time)<< ":ERROR: Invalid block size calculated: [" << real_block_rows 
        //         << "," << real_block_cols << "]" << endl;
        //     cout<<"start_row:"<<start_row<<",start_col:"<<start_col<<endl;
        //     cout<<hex<<"start_addr:"<<start_addr<<",matrix_start_addr:"<<matrix_start_addr<<",matrix_end_addr:"<<matrix_end_addr<<endl;
        //     sc_stop();
        //     return;
        // }

        vector<T> block_buffer(real_block_cols);
        for(int i = 0; i < real_block_rows; i++){
            read_data(start_addr + i * m_cols * sizeof(T), block_buffer, source_dmi, real_block_cols);
            //check_all_zero(block_buffer);
            write_data(target_start_addr + i * real_block_cols * sizeof(T), target_end_addr, block_buffer, target_dmi, real_block_cols);
        }
        end_addr = start_addr + (((real_block_rows-1) * m_cols+real_block_cols)) * sizeof(T) - 1;
        //先行后列循环
        next_block_start_addr = start_addr + (real_block_cols * sizeof(T));
        rowloop_complete = false;
        if ((start_col + real_block_cols) == m_cols) {
            next_block_start_addr = matrix_start_addr + 
                                ((start_row + real_block_rows) * m_cols * sizeof(T));
            rowloop_complete = true;
        }
        // // 计算下一个块的起始地址
        // if (traverse_by_row) {
        //     //先行后列循环
        //     next_block_start_addr = start_addr + (real_block_cols * sizeof(T));
        //     rowloop_complete = false;
        //     if ((start_col + real_block_cols) == m_cols) {
        //         next_block_start_addr = matrix_start_addr + 
        //                             ((start_row + real_block_rows) * m_cols * sizeof(T));
        //         rowloop_complete = true;
        //     }
        // } else {
        //     //先列后行循环
        //     next_block_start_addr = start_addr + (real_block_rows * m_cols * sizeof(T));
        //     rowloop_complete = false;
        //     if ((start_row + real_block_rows) == m_rows) {
        //         next_block_start_addr = matrix_start_addr + 
        //                             (start_col + real_block_cols) * sizeof(T);
        //         rowloop_complete = true;
        //     }
        // }

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
        // if(end_addr != target_end_addr){
        //     // cout<<"Error transfer_back: "<<transfer_name<<endl;
        //     // cout<<"start_addr:"<<start_addr<<",target_start_addr:"<<target_start_addr<<endl;
        //     // cout<<"end_addr:"<<end_addr<<",target_end_addr:"<<target_end_addr<<endl;
        //     // cout<<"am_rows:"<<am_rows<<",am_cols:"<<am_cols<<endl;
        //     // cout<<"ddr_rows:"<<ddr_rows<<",ddr_cols:"<<ddr_cols<<endl;  
        //     sc_stop();
        //     return;
        // }
        
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
        SC_THREAD(compute_matrix_thread);
    }

    // 添加标志变量
    bool computation_in_progress = false;

    void compute_matrix_thread() {
        while(true) {
            wait(start_compute);

            start_gemm.notify();
            init_time = sc_time_stamp();
            cout << (sc_time_stamp() - init_time) << "=====================启动gemm============================" << endl;
            wait(output_result_done);
            computation_in_progress = false;

        }

    }

    void compute_matrix(
        const std::vector<std::vector<T>>& A,
        const std::vector<std::vector<T>>& B,
        std::vector<std::vector<T>>& C,
        int M, int N, int K
    ) {
        computation_in_progress = true;
        
        // 设置矩阵维度
        A_rows = M;
        A_cols = K;
        B_rows = K;
        B_cols = N;
        C_rows = M;
        C_cols = N;

        convertTo1D(A, MatrixA);
        convertTo1D(B, MatrixB);
        convertTo1D(C, MatrixC);
        // write_matrix_in_file(MatrixC, "MatrixC_bias.txt", C_rows, C_cols);
        // cout << "打印出gemm前的C的值" << endl;
        //正确

        start_compute.notify();
        // 等待计算完成
        while(computation_in_progress) {
            sc_start(1, SC_NS);
        }
        // 检查计算结果
        //write_matrix_in_file(MatrixC, "block1_conv1_gemm_output.txt", C_rows, C_cols);
        convertTo2D(MatrixC, C, C_rows,C_cols);    
    }
    void compute_fc(
        const std::vector<T>& A,
        const std::vector<std::vector<T>>& B,
        std::vector<T>& C,
        int K, int N
    ) {
        computation_in_progress = true;
        int M = 1;
        // 设置矩阵维度
        A_rows = M;
        A_cols = K;
        B_rows = K;
        B_cols = N;
        C_rows = M;
        C_cols = N;
        cout << "M:" << M << ",N:" << N << ",K:" << K << endl;
        MatrixA.resize(M*K);
        MatrixC.resize(M*N);
        memcpy(MatrixA.data(), A.data(), K * M* sizeof(T));
        memcpy(MatrixC.data(), C.data(), M * N * sizeof(T));
        convertTo1D(B, MatrixB);

        start_compute.notify();
        // 等待计算完成
        while(computation_in_progress) {
            sc_start(1, SC_NS);
        }
        // 检查计算结果
        memcpy(C.data(), MatrixC.data(), M * N * sizeof(T));   

    }
    void setup_dmi(uint64_t base_addr, uint64_t size, tlm::tlm_dmi& dmi) {
        tlm::tlm_generic_payload trans;
        trans.set_address(base_addr);
        trans.set_command(tlm::TLM_READ_COMMAND);
        trans.set_data_length(sizeof(T));

        if (socket->get_direct_mem_ptr(trans, dmi)) {
            // cout << "DMI setup successful for range: 0x" << hex
            //      << dmi.get_start_address() << " - 0x" << dmi.get_end_address() << endl;
            return;
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
    //修改为输入为一维数组
    void kernel_mul(
        vector<T>& A_1D,
        vector<T>& B_1D,
        vector<T>& C_1D,
        const int rows_a,
        const int cols_a,
        const int rows_b,
        const int cols_b,
        const int rows_c,
        const int cols_c
    ){
                // 定义事务和延迟
        tlm_generic_payload trans;
        sc_time delay = SC_ZERO_TIME;
        //向量外积
        // 遍历 A 的每一行M
        vector<T> vec1(cols_b);
        vector<T> vec2(cols_b);
        vector<T> vec3(cols_b);
        vector<T> merged_vec(cols_b*3);
        for(int m = 0; m < rows_a; m++){
            //遍历A的每一列K,B的每一行K，根据规定，B一行的元素不会超过macs_per_vpu
            for(int k = 0; k < cols_a; k++){
                //C[m][start:end] += A[m][k]*B[k][start:end]
                //A[m][k]重复cols_b次,构成vec1
                //B[k][:]取出一整行，构成vec2
                //C[m][:]取出一整行，构成vec3
                for(int i = 0; i < cols_b; i++){
                    vec1[i] = A_1D[m*cols_a+k];
                    vec2[i] = B_1D[k*cols_b+i];
                    vec3[i] = C_1D[m*cols_c+i];
                }
                merge_vectors(vec1, vec2, vec3, merged_vec);
                // 设置事务属性
                trans.set_command(tlm::TLM_WRITE_COMMAND);
                trans.set_address(VPU_BASE_ADDR);  // 设置起始地址
                trans.set_data_length(sizeof(T) * merged_vec.size());
                trans.set_data_ptr(reinterpret_cast<unsigned char*>(merged_vec.data()));
                socket->b_transport(trans, delay);
                split_vector(merged_vec, vec1, vec2, vec3);
                //C的中间结果写回
                for(int i = 0; i < cols_b; i++){
                    C_1D[m*cols_c+i] = vec3[i];
                }
            }
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
        while(true){
            wait(init_ready);
            //cout<<(sc_time_stamp() - init_time)<<"=====================准备开始分块计算============================" << endl;
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



            // M方向循环 (处理MatrixA的行)
            for(m = 0; m < M_blocks && !M_complete; m++) {
                //cout<<(sc_time_stamp() - init_time)<<"=====================开始M方向循环============================" << endl;
                // 计算当前M块的实际大小
                int current_m_size = min(m_gsm_max, A_rows - m * m_gsm_max);
                for(k = 0; k < K_blocks && !K_complete; k++) {
                    //cout<<(sc_time_stamp() - init_time)<<"=====================开始K方向循环============================" << endl;
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

                    for(n = 0; n < N_blocks && !N_complete; n++) {
                        //cout<<(sc_time_stamp() - init_time)<<"=====================开始N方向循环============================" << endl;
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
                            ddr_dmi, am_dmi,
                            true,
                            N_complete
                        );
                        C_addr[AM][start] =B_addr[AM][end] + 1;
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
                            ddr_dmi, am_dmi,
                            true,
                            N_complete
                        );

                        int SM_blocks = (current_m_size + sm_max - 1) / sm_max;

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
                                gsm_dmi, sm_dmi,
                                false, //列循环
                                SM_complete
                            );
                            A_GSM_addr_flag = A_addr[GSMSM][start];
                            A_addr[GSMSM][start] = A_next_addr[GSMSM][start];
                            // 执行计算
                            compute_ready.notify();
                            wait(kernel_com_finished);
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
                    //B_addr[DDR_BC][start] = B_next_addr[DDR_BC][start]  
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
            computation_done.notify();
            cout << (sc_time_stamp() - init_time) << "=====================完成计算============================" << endl;
        }
    }
    void init_process(){
        // Setup DMI for DDR and GSM

        while(true){
            wait(start_gemm);
            
            setup_dmi(DDR_BASE_ADDR, DDR_SIZE, ddr_dmi);
            setup_dmi(GSM_BASE_ADDR, GSM_SIZE, gsm_dmi);
            setup_dmi(SM_BASE_ADDR, SM_SIZE, sm_dmi);
            setup_dmi(AM_BASE_ADDR, AM_SIZE, am_dmi);
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
            
            
            // Write batch data to DDR
            //cout << "=== MatrixA Write to DDR ===" << endl;
            int MatrixA_num = A_rows * A_cols;
            write_data(A_addr[DDR_A][start], A_addr[DDR_A][end], MatrixA, ddr_dmi, MatrixA_num);
            //cout <<"MatrixA start addr:"<<A_addr[DDR_A][start]<<endl;
            // cout << " MatrixA end addr: " << A_addr[DDR_A][end] << endl;
            // //打印MatrixA的前10个元素
            // cout<<"MatrixA的前10个元素:"<<endl;
            // for(int i=0;i<10;i++){
            //     cout<<MatrixA[i]<<" ";
            // }
            // cout<<endl;

            //cout << "=== MatrixB Write to DDR ===" << endl;
            int MatrixB_num = B_rows * B_cols;
            write_data(B_addr[DDR_BC][start], B_addr[DDR_BC][end], MatrixB, ddr_dmi, MatrixB_num);
            // cout <<"MatrixB start addr:"<<B_addr[DDR_BC][start]<<endl;
            // cout << "MatrixB end addr: " << B_addr[DDR_BC][end] << endl;
            // //打印MatrixB的前10个元素
            // cout<<"MatrixB的前10个元素:"<<endl;
            // for(int i=500;i<510;i++){
            //     cout<<MatrixB[i]<<" ";
            // }
            // cout<<endl;

            //cout << "=== MatrixC Write to DDR ===" << endl;
            int MatrixC_num = C_rows * C_cols;
            write_data(C_addr[DDR_BC][start], C_addr[DDR_BC][end], MatrixC, ddr_dmi, MatrixC_num);
            // cout <<"MatrixC start addr:"<<C_addr[DDR_BC][start]<<endl;
            // cout << "MatrixC end addr: " << C_addr[DDR_BC][end] << endl;
            // //打印MatrixC的前10个元素   
            // cout<<"MatrixC的前10个元素:"<<endl;
            // for(int i=0;i<10;i++){
            //     cout<<MatrixC[i]<<" ";
            // }
            // cout<<endl;

            M_complete = false;
            K_complete = false;
            N_complete = false;
            SM_complete = false;
        
            init_ready.notify();
            //cout << (sc_time_stamp() - init_time) << "=====================初始化完成============================" << endl; 
        }
    }
    void computing_process() {
        while(true) {
            // 等待新的计算任务
            wait(compute_ready);
            
            // 根据当前块大小调整vector大小
            A_block_1D.resize(A_SM_size[row] * A_SM_size[col]);
            B_block_1D.resize(B_AM_size[row] * B_AM_size[col]);
            C_block_1D.resize(A_SM_size[row] * B_AM_size[col]);
            
            // A_block.resize(A_SM_size[row], vector<T>(A_SM_size[col]));
            // B_block.resize(B_AM_size[row], vector<T>(B_AM_size[col]));
            // C_block.resize(A_SM_size[row], vector<T>(B_AM_size[col]));

            // 从SM读取A块
            read_data(A_addr[SM][start], A_block_1D, sm_dmi, A_SM_size[row]*A_SM_size[col]);
            read_data(B_addr[AM][start], B_block_1D, am_dmi, B_AM_size[row]*B_AM_size[col]);
            
            // 计算C块在AM中的偏移
            uint64_t C_offset = (A_GSM_addr_flag - A_addr[GSM][start]) / ( A_GSM_size[col]) * C_AM_size[col];
            read_data(C_addr[AM][start]+C_offset, C_block_1D, am_dmi, A_SM_size[row]*B_AM_size[col]);

            // convertTo2D(A_block_1D, A_block, A_SM_size[row], A_SM_size[col]);
            // convertTo2D(B_block_1D, B_block, B_AM_size[row], B_AM_size[col]);
            // convertTo2D(C_block_1D, C_block, A_SM_size[row], B_AM_size[col]);
            // 执行矩阵乘法计算
            kernel_mul(
                A_block_1D, B_block_1D, C_block_1D,
                A_SM_size[row], A_SM_size[col],
                B_AM_size[row], B_AM_size[col],
                A_SM_size[row], B_AM_size[col]
            );
            
            // convertTo1D(C_block, C_block_1D);
            // 将计算结果写回AM

            write_data(C_addr[AM][start]+C_offset, temp_end_addr, C_block_1D, am_dmi, A_SM_size[row]*B_AM_size[col]);
            
            // 通知计算完成
            kernel_com_finished.notify();

            
        }
    }
    void writeback_C_process() {
        MatrixBlockTransfer<T> amCback_transfer("AMCback_Transfer", init_time);
        while(true) {
            wait(C_write_back_ready);
            amCback_transfer.transfer_back(
                C_addr[AM][start],
                C_addr[DDR_BC][start],
                C_addr[DDR_BC][end],
                C_AM_size[row],
                C_AM_size[col],
                C_rows,
                C_cols,
                am_dmi,
                ddr_dmi
            );
            C_writeback_done.notify();
        }
    }
    void output_result() {
        while(true) {
            wait(computation_done);  
            
            // 从DDR一次性读取整个结果
            read_data(Matrix_addr[C][start], MatrixC, ddr_dmi, C_rows * C_cols);

            output_result_done.notify();
            //cout<<(sc_time_stamp() - init_time)<<"=====================GEMM执行结束============================"<<endl;
            // //打印C矩阵的前10个元素
            // cout<<"MatrixC的前10个元素:"<<endl;
            // for(int i=0;i<10;i++){
            //     cout<<MatrixC[i]<<" ";
            // }
            //cout<<endl;
        }
    }
    virtual void invalidate_direct_mem_ptr(sc_dt::uint64 start_range, sc_dt::uint64 end_range) {
        cout << "DMI invalidated. Range: " << hex << start_range << " - " << end_range << endl;
    }
    
    public:
        tlm::tlm_dmi ddr_dmi, gsm_dmi, sm_dmi, am_dmi, vpu_dmi;
        sc_event start_gemm,start_compute;
        sc_event init_ready,compute_ready,C_write_back_ready;
        sc_event C_writeback_done,kernel_com_finished;  
        sc_event computation_done, output_result_done; 

        vector<T> MatrixA;
        vector<T> MatrixB;
        vector<T> MatrixC;
        // 将vector声明移到循环外，避免重复创建和销毁
        vector<T> A_block_1D;
        vector<T> B_block_1D;
        vector<T> C_block_1D;
        // vector<vector<T>> A_block;
        // vector<vector<T>> B_block;
        // vector<vector<T>> C_block;
        
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
        uint64_t temp_end_addr;  // 临时变量存储写入结束地址

        //实际DDR中矩阵起始结束地址
        array<array<uint64_t, 2>, 3> Matrix_addr = {{{0, 0}, {0, 0}, {0, 0}}};
        //分块矩阵的起始结束地址
        array<array<uint64_t, 2>, 4> A_addr = {{{0, 0}, {0, 0}, {0, 0}, {0, 0}}};
        array<array<uint64_t, 2>, 2> B_addr = {{{0, 0}, {0, 0}}};
        array<array<uint64_t, 2>, 2> C_addr = {{{0, 0}, {0, 0}}};   
        array<array<uint64_t, 2>, 3> A_next_addr = {{{0, 0}, {0, 0}, {0, 0}}};
        array<array<uint64_t, 2>, 2> B_next_addr = {{{0, 0}, {0, 0}}};
        array<array<uint64_t, 2>, 2> C_next_addr = {{{0, 0}, {0, 0}}}; 

};
#endif
