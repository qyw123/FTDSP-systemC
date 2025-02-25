#include "systemc"
#include "DDR.h"
#include "initiator.h"

using namespace sc_core;

int sc_main(int argc, char* argv[]) {
    DDR<double> ddr("DDR");
    Initiator<double> initiator("Initiator");

    initiator.set_target(&ddr);  // 将 DDR 的实例传递给 Initiator
    // 绑定 Initiator 和 DDR 的 socket
    initiator.socket.bind(ddr.socket);

    sc_start();

    return 0;
}
