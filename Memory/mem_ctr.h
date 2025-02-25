#ifndef INITIATOR_H
#define INITIATOR_H

#include "const.h"
#include "tools.h"

template <typename T>
struct Mem_ctr : sc_module {
    tlm_utils::simple_initiator_socket<Mem_ctr> socket;

    SC_CTOR(Mem_ctr) : socket("socket") {
        socket.register_invalidate_direct_mem_ptr(this, &Mem_ctr::invalidate_direct_mem_ptr);
        SC_THREAD(test_process);
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
            SC_REPORT_ERROR("Mem_ctr", "DMI setup failed");
        }
    }

    void write_to_dmi(uint64_t start_addr, uint64_t& end_addr, const std::vector<T>& values, const tlm::tlm_dmi& dmi, unsigned int data_num) {
        const unsigned int bytes_per_block = DDR_DATA_WIDTH; // Block size in bytes
        const unsigned int elements_per_block = bytes_per_block / sizeof(T);

        // Ensure the provided data_num matches the size of the values vector
        if (data_num != values.size()) {
            SC_REPORT_ERROR("Mem_ctr", "Mismatch between data_num and values size");
            return;
        }

        // Calculate the end address based on the start address and data_num
        end_addr = start_addr + data_num * sizeof(T) - 1;

        // Check if the address range is within DMI bounds
        if (end_addr <= dmi.get_end_address() && start_addr >= dmi.get_start_address()) {
            unsigned char* dmi_addr = dmi.get_dmi_ptr() + (start_addr - dmi.get_start_address());
            unsigned int total_blocks = (data_num + elements_per_block - 1) / elements_per_block;

            for (unsigned int block = 0; block < total_blocks; ++block) {
                unsigned int block_start = block * elements_per_block;
                unsigned int block_end = std::min(block_start + elements_per_block, data_num);

                for (unsigned int i = block_start; i < block_end; ++i) {
                    memcpy(dmi_addr + i * sizeof(T), &values[i], sizeof(T));
                }
                wait(dmi.get_write_latency());
            }
        } else {
            SC_REPORT_ERROR("Mem_ctr", "DMI write failed: Address out of range");
        }
    }

    void read_from_dmi(uint64_t addr, vector<T>& values, const tlm::tlm_dmi& dmi, unsigned int data_num) {
        const unsigned int bytes_per_block = DDR_DATA_WIDTH; // Block size in bytes
        const unsigned int elements_per_block = bytes_per_block / sizeof(T);

        if (addr + data_num * sizeof(T) - 1 <= dmi.get_end_address()) {
            unsigned char* dmi_addr = dmi.get_dmi_ptr() + (addr - dmi.get_start_address());
            unsigned int total_blocks = (data_num + elements_per_block - 1) / elements_per_block;

            for (unsigned int block = 0; block < total_blocks; ++block) {
                unsigned int block_start = block * elements_per_block;
                unsigned int block_end = std::min(block_start + elements_per_block, data_num);

                for (unsigned int i = block_start; i < block_end; ++i) {
                    memcpy(&values[i], dmi_addr + i * sizeof(T), sizeof(T));
                    //cout << sc_time_stamp() << ": DMI READ Addr = " << hex << addr + i * sizeof(T) << ", Data = " << values[i] << endl;
                }
                wait(dmi.get_read_latency());
            }
        } else {
            SC_REPORT_ERROR("Mem_ctr", "DMI read failed: Address out of range");
        }
    }

    void transfer_matrixblock(
        //本块矩阵的起始地址
        uint64_t start_addr, 
        //大矩阵中下一块的起始地址
        uint64_t& next_block_start_addr,
        // 大矩阵的起始终止地址
        uint64_t matrix_start_addr,
        uint64_t matrix_end_addr,
        //目标内存的起始终止地址
        const uint64_t target_start_addr,
        uint64_t& target_end_addr,
        //大矩阵的形状
        int m_rows, 
        int m_cols, 
        //最大分块的形状
        int block_rows, 
        int block_cols,
        //实际分块形状
        int& real_block_rows, 
        int& real_block_cols,
        //dmi对象
        const tlm::tlm_dmi& source_dmi, 
        const tlm::tlm_dmi& target_dmi,
        bool traverse_by_row,
        bool& is_complete
    ) {
        // 确保起始地址在有效范围内
        if (start_addr > matrix_end_addr || start_addr < matrix_start_addr) {
            SC_REPORT_ERROR("transfer_matrixblock", "Start address out of matrix range.");
            return;
        }

        // 确定当前块的起始行列号
        int start_offset = (start_addr - matrix_start_addr) / sizeof(T);
        int start_row = start_offset / m_cols;
        int start_col = start_offset % m_cols;

        // 计算当前块的大小
        int current_block_rows = std::min(block_rows, m_rows - start_row);
        int current_block_cols = std::min(block_cols, m_cols - start_col);
        real_block_rows = current_block_rows;
        real_block_cols = current_block_cols;

        // 确定当前块的元素数
        int block_elements = current_block_rows * current_block_cols;

        // 分配缓冲区
        std::vector<T> block_data(block_elements, 0.0);

        // 从源地址读取当前块
        for (int r = 0; r < current_block_rows; ++r) {
            uint64_t row_start_addr = start_addr + r * m_cols * sizeof(T);
            int row_elements = current_block_cols;

            if (row_start_addr < matrix_end_addr) {
                std::vector<T> row_data(row_elements); // 临时缓冲区存储一行
                read_from_dmi(row_start_addr, row_data, source_dmi, row_elements);
                std::copy(row_data.begin(), row_data.end(), block_data.begin() + r * current_block_cols);
            }
        }

        // 写入目标地址（一次性写入整个块）
        write_to_dmi(target_start_addr, target_end_addr, block_data, target_dmi, block_elements);

        if (traverse_by_row) {
            // 按行优先分块
            next_block_start_addr = start_addr + block_cols * sizeof(T);

            // 如果当前行已经结束，换行到下一块
            if ((start_offset % m_cols) + block_cols >= m_cols) {
                next_block_start_addr = matrix_start_addr 
                                    + ((start_offset / m_cols) + block_rows) * m_cols * sizeof(T);
            }
        } else {
            // 按列优先分块
            next_block_start_addr = start_addr + block_rows * m_cols * sizeof(T);

            // 如果当前列已经结束，换列到下一块
            if ((start_offset / m_cols) + block_rows >= m_rows) {
                next_block_start_addr = matrix_start_addr 
                                    + ((start_offset % m_cols) + block_cols) * sizeof(T);
            }
        }
        // 判断是否完成所有块的传输
        if (traverse_by_row) {
            // 行优先逻辑：检查是否已经到最后一行且最后一列
            is_complete = (start_row + real_block_rows >= m_rows) && (start_col + real_block_cols >= m_cols);
        } else {
            // 列优先逻辑：检查是否已经到最后一列且最后一行
            is_complete = (start_col + real_block_cols >= m_cols) && (start_row + real_block_rows >= m_rows);
        }
        
    }

    void test_process() {
        tlm::tlm_dmi ddr_dmi, gsm_dmi, vcore_dmi;
        bool MatrixA_DDR_empty, MatrixB_DDR_empty, MatrixC_DDR_empty;
        bool MatrixA_GSM_empty;
        //矩阵在DDR中的起始地址
        uint64_t MatrixA_DDR_start_addr, MatrixB_DDR_start_addr, MatrixC_DDR_start_addr;
        //矩阵在DDR中的结束地址
        uint64_t MatrixA_DDR_end_addr, MatrixB_DDR_end_addr, MatrixC_DDR_end_addr;
        //分块矩阵的起始地址
        uint64_t A0_DDR_start_addr, A1_DDR_start_addr;
        uint64_t A0_GSM_start_addr, A1_GSM_start_addr;
        uint64_t A0_GSMSM_start_addr, A1_GSMSM_start_addr;
        uint64_t A0_SM_start_addr, A1_SM_start_addr;
        uint64_t B0_DDR_start_addr, B1_DDR_start_addr;
        uint64_t B0_AM_start_addr, B1_AM_start_addr;
        uint64_t C0_DDR_start_addr, C1_DDR_start_addr; 
        uint64_t C0_AM_start_addr, C1_AM_start_addr;        
        //矩阵分块在的结束地址
        uint64_t A0_GSM_end_addr, A1_GSM_end_addr;
        uint64_t A0_SM_end_addr, A1_SM_end_addr;
        uint64_t B0_AM_end_addr, B1_AM_end_addr;
        uint64_t C0_AM_end_addr, C1_AM_end_addr;

        //输入数据
        string matrixA_file_path = "matrixA_input.txt";
        string matrixB_file_path = "matrixB_input.txt";
        string matrixC_file_path = "matrixC_input.txt";
        //初始化
        int real_A0_GSM_block_rows = 0, real_A0_GSM_block_cols = 0, real_A1_GSM_block_rows = 0, real_A1_GSM_block_cols = 0;
        int real_A0_SM_block_rows = 0, real_A0_SM_block_cols = 0, real_A1_SM_block_rows = 0, real_A1_SM_block_cols = 0;
        int real_B0_AM_block_rows = 0, real_B0_AM_block_cols = 0, real_B1_AM_block_rows = 0, real_B1_AM_block_cols = 0;
        int real_C0_AM_block_rows = 0, real_C0_AM_block_cols = 0, real_C1_AM_block_rows = 0, real_C1_AM_block_cols = 0;
        //初始化
        vector<T> DDR_data;
        int A_rows = 0, A_cols = 0, B_rows = 0, B_cols = 0, C_rows = 0, C_cols = 0;

        record_matrix_shape<T>(matrixA_file_path, A_rows, A_cols);
        record_matrix_shape<T>(matrixB_file_path, B_rows, B_cols);
        record_matrix_shape<T>(matrixC_file_path, C_rows, C_cols);

        // Setup DMI for DDR and GSM
        setup_dmi(DDR_BASE_ADDR, DDR_SIZE, ddr_dmi);
        setup_dmi(GSM_BASE_ADDR, GSM_SIZE, gsm_dmi);
        setup_dmi(VCORE_BASE_ADDR, VCORE_SIZE, vcore_dmi);

        // Write batch data to DDR
        cout << "=== MatrixA Write to DDR ===" << endl;
        load_from_file<T>(DDR_data, matrixA_file_path);
        //从DDR的起始地址写入MatrixA
        MatrixA_DDR_start_addr = DDR_BASE_ADDR;
        int MatrixA_num = DDR_data.size();
        write_to_dmi(MatrixA_DDR_start_addr, MatrixA_DDR_end_addr, DDR_data, ddr_dmi, MatrixA_num);
        //cout << " MatrixA end addr: " << MatrixA_DDR_end_addr << endl;

        cout << "=== MatrixB Write to DDR ===" << endl;
        DDR_data.clear();
        load_from_file<T>(DDR_data, matrixB_file_path);
        //在MatrixA后写入MatrixB
        MatrixB_DDR_start_addr = MatrixA_DDR_end_addr+1;
        int MatrixB_num = DDR_data.size();
        write_to_dmi(MatrixB_DDR_start_addr, MatrixB_DDR_end_addr, DDR_data, ddr_dmi, MatrixB_num);
        //cout << "MatrixB end addr: " << MatrixB_DDR_end_addr << endl;

        cout << "=== MatrixC Write to DDR ===" << endl;
        DDR_data.clear();
        load_from_file<T>(DDR_data, matrixC_file_path);
        //在MatrixB后写入MatrixC（结果矩阵，初始化为0）
        MatrixC_DDR_start_addr = MatrixB_DDR_end_addr+1;
        int MatrixC_num = DDR_data.size();
        write_to_dmi(MatrixC_DDR_start_addr, MatrixC_DDR_end_addr, DDR_data, ddr_dmi, MatrixC_num);
        //cout << "MatrixC end addr: " << MatrixC_DDR_end_addr << endl;
        sc_time reset_time = sc_time_stamp();

        cout << (sc_time_stamp() - reset_time) << ":=== MatrixA Transfer DDR to GSM ===" << endl;
        cout << (sc_time_stamp() - reset_time) << ":A0: 12*384 -> GSM" << endl;
        A0_DDR_start_addr = MatrixA_DDR_start_addr;
        A0_GSM_start_addr = GSM_BASE_ADDR;
        transfer_matrixblock(
            A0_DDR_start_addr, 
            A1_DDR_start_addr,
            MatrixA_DDR_start_addr,
            MatrixA_DDR_end_addr,
            A0_GSM_start_addr,
            A0_GSM_end_addr,
            A_rows, 
            A_cols, 
            m_gsm_max, 
            k_gsm_max,
            real_A0_GSM_block_rows,
            real_A0_GSM_block_cols,
            ddr_dmi, 
            gsm_dmi,
            true,
            MatrixA_DDR_empty
        );
        //cout << "MatrixA_DDR_end_addr:" << MatrixA_DDR_end_addr<<endl;
        //cout << "A1_GSM_start_addr:" << A1_GSM_start_addr<<endl;
        if(MatrixA_DDR_empty) 
            cout << (sc_time_stamp() - reset_time) << ":MatrixA finished transfer from DDR to GSM" << endl;
                else
            cout << (sc_time_stamp() - reset_time) << ":MatrixA doesn't finish transfer" << endl;

        //cout << "A1_GSM_start_addr:" << A1_GSM_start_addr << endl;
        //17280 ns
        cout << (sc_time_stamp() - reset_time) << ":=== A0 Transfer GSM to SM ===" << endl;
        cout << (sc_time_stamp() - reset_time) << ":A0: 12*384 -> SM" << endl;
        A0_GSMSM_start_addr = GSM_BASE_ADDR;
        A0_SM_start_addr = SM_BASE_ADDR;
        transfer_matrixblock(  
            A0_GSMSM_start_addr, 
            A1_GSMSM_start_addr,
            GSM_BASE_ADDR,
            A0_GSM_end_addr,
            A0_SM_start_addr,
            A0_SM_end_addr,
            real_A0_GSM_block_rows, 
            real_A0_GSM_block_cols, 
            sm_max, 
            k_gsm_max,
            real_A0_SM_block_rows, 
            real_A0_SM_block_cols, 
            gsm_dmi, 
            vcore_dmi,
            true,
            MatrixA_GSM_empty
        );
        if(MatrixA_GSM_empty) 
            cout << (sc_time_stamp() - reset_time) << ":MatrixA finished transfer from GSM to SM" << endl;
        else
            cout << (sc_time_stamp() - reset_time) << ":MatrixA doesn't finish transfer" << endl;
        //cout << "A1_SM_start_addr:" << A1_SM_start_addr << endl;
        //28800 ns
        cout << (sc_time_stamp() - reset_time) << ":=== MatrixB Transfer DDR to AM ===" << endl;
        cout << (sc_time_stamp() - reset_time) << ":B0_1: 384*64 -> AM" << endl;
        B0_DDR_start_addr = MatrixB_DDR_start_addr;
        B0_AM_start_addr = AM_BASE_ADDR;
        transfer_matrixblock(  
            B0_DDR_start_addr, 
            B1_DDR_start_addr,
            MatrixB_DDR_start_addr,
            MatrixB_DDR_end_addr,
            B0_AM_start_addr,
            B0_AM_end_addr,
            B_rows, 
            B_cols,  
            k_gsm_max,
            cu_max,
            real_B0_AM_block_rows, 
            real_B0_AM_block_cols, 
            ddr_dmi, 
            vcore_dmi,
            false,
            MatrixB_DDR_empty
        );
        //cout << "B1_AM_start_addr: " << B1_AM_start_addr << endl;
        if(MatrixB_DDR_empty) 
            cout << (sc_time_stamp() - reset_time) << ":MatrixB finished transfer from DDR to AM" << endl;
        else
            cout << (sc_time_stamp() - reset_time) << ":MatrixB doesn't finish transfer" << endl;

        cout << (sc_time_stamp() - reset_time) << ":=== MatrixC Transfer DDR to AM ===" << endl;
        cout << (sc_time_stamp() - reset_time) << ":C0_1: 12*64 -> AM" << endl;
        C0_DDR_start_addr = MatrixC_DDR_start_addr;
        C0_AM_start_addr = B0_AM_end_addr+1;//BC矩阵公用AM
        transfer_matrixblock(  
            C0_DDR_start_addr, 
            C1_DDR_start_addr,
            MatrixC_DDR_start_addr,
            MatrixC_DDR_end_addr,
            C0_AM_start_addr,
            C0_AM_end_addr,
            C_rows, 
            C_cols,  
            sm_max,
            cu_max,
            real_C0_AM_block_rows, 
            real_C0_AM_block_cols, 
            ddr_dmi, 
            vcore_dmi,
            true,
            MatrixC_DDR_empty
        );
        cout << "C1_AM_start_addr: " << C1_AM_start_addr << endl;
        if(MatrixC_DDR_empty) 
            cout << (sc_time_stamp() - reset_time) << ":MatrixC finished transfer from DDR to AM" << endl;
        else
            cout << (sc_time_stamp() - reset_time) << ":MatrixC doesn't finish transfer" << endl;

        cout<< "----------------------缓存完毕！可以开始计算：C=A*B+C-------------------------------------" << endl;

        sc_stop();
    }

    virtual void invalidate_direct_mem_ptr(sc_dt::uint64 start_range, sc_dt::uint64 end_range) {
        cout << "DMI invalidated. Range: " << hex << start_range << " - " << end_range << endl;
    }
};

#endif
