#include "./src/Soc.h"
#include "./gemm.h"
#include "./util/const.h"

using DataType = double;
SC_MODULE(Top){
    Soc<DataType>* soc;
    Gemm<DataType>* gemm;
    SC_CTOR(Top){
        soc = new Soc<DataType>("soc");
        gemm = new Gemm<DataType>("gemm");

        gemm->socket.bind(soc->target_socket);
    }
};

int sc_main(int argc, char* argv[])
{
    Top top("top");
    sc_start();
    return 0;
}
