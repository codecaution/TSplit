#include <cmath>
#include <unordered_map>
#include "ps/ps.h"
#include "../common/dlarray.h"
#include <unistd.h>
#include "ps/internal/postoffice.h"

using namespace ps;

extern "C" {

void Scheduler_Init(){
    Start(0);
    if (!IsScheduler()) assert(false);
}

void Scheduler_Finalize(){
    Finalize(0, true);

}

}

