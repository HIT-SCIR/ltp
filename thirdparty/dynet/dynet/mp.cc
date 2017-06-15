#if !_WINDOWS
#include "mp.h"
#include "dynet/except.h"
using namespace std;
using namespace boost::interprocess;

namespace dynet {
  namespace mp {
    // TODO: Pass these around instead of having them be global
    std::string queue_name = "dynet_mp_work_queue";
    std::string shared_memory_name = "dynet_mp_shared_memory";
    timespec start_time;
    bool stop_requested = false;
    SharedObject* shared_object = nullptr;

    std::string generate_queue_name() {
      std::ostringstream ss;
      ss << "dynet_mp_work_queue";
      ss << rand();
      return ss.str();
    }

    std::string generate_shared_memory_name() {
      std::ostringstream ss;
      ss << "dynet_mp_shared_memory";
      ss << rand();
      return ss.str();
    }

    dynet::real sum_values(const std::vector<dynet::real>& values) {
      return accumulate(values.begin(), values.end(), 0.0);
    }

    dynet::real mean(const std::vector<dynet::real>& values) {
      return sum_values(values) / values.size();
    }

    std::string elapsed_time_string(const timespec& start, const timespec& end) {
      std::ostringstream ss;
      time_t secs = end.tv_sec - start.tv_sec;
      long nsec = end.tv_nsec - start.tv_nsec;
      ss << secs << " seconds and " << nsec << "nseconds";
      return ss.str();
    }

    unsigned spawn_children(std::vector<Workload>& workloads) {
      const unsigned num_children = workloads.size();
      pid_t pid;
      unsigned cid;
      for (cid = 0; cid < num_children; ++cid) {
        pid = fork();
        if (pid == -1) {
          std::cerr << "Fork failed. Exiting ..." << std::endl;
          return 1;
        }
        else if (pid == 0) {
          // children shouldn't continue looping
          break;
        }
        workloads[cid].pid = pid;
      }
      return cid;
    }

    std::vector<Workload> create_workloads(unsigned num_children) {
      int err;
      std::vector<Workload> workloads(num_children);
      for (unsigned cid = 0; cid < num_children; cid++) { 
        err = pipe(workloads[cid].p2c);
        if(err != 0) DYNET_RUNTIME_ERR("Problem writing to p2c pipe " << cid << " in create_workloads");
        err = pipe(workloads[cid].c2p);
        if(err != 0) DYNET_RUNTIME_ERR("Problem writing to c2p pipe " << cid << " in create_workloads");
      }
      return workloads;
    }

    void cleanup(const std::vector<Workload>& workloads) {
      for (const Workload& workload : workloads) {
        close (workload.c2p[0]);
        close (workload.c2p[1]);
        close (workload.p2c[0]);
        close (workload.p2c[1]);
      }
    }

  }
}
#endif
