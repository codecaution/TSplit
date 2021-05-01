#include <cmath>
#include <unordered_map>
#include "ps/ps.h"
#include "../common/dlarray.h"
#include <unistd.h>
#include "ps/internal/postoffice.h"
#include "ps/psf/PSFHandle.h"
using namespace ps;

extern "C" {
void Server_Init(){
    Start(0);
    if (!IsServer()) assert(false);
}

void StartServer() {
    auto server = new KVServer<float>(0);
    server->set_request_handle(KVServerMatrixHandle<float>());
    RegisterExitCallback([server]() { delete server; });
}

void Server_Finalize(){
    Finalize(0, true);
}

}
