#pragma once
#if !_WINDOWS
#include "dynet/globals.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/dict.h"
#include "dynet/lstm.h"
#include <boost/algorithm/string.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/interprocess/anonymous_shared_memory.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/shm.h>
#include <iostream>
#include <limits>
#include <fstream>
#include <vector>
#include <utility>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>

namespace dynet {
  namespace mp {
    // TODO: Pass these around instead of having them be global
    extern std::string queue_name;
    extern std::string shared_memory_name;
    extern timespec start_time;
    extern bool stop_requested;

    struct WorkloadHeader {
      bool is_dev_set;
      bool end_of_epoch;
      unsigned report_frequency;
    };

    // A simple struct to hold information about a child process
    // TODO: Rename me!
    struct Workload {
      pid_t pid;
      int c2p[2]; // Child to parent pipe
      int p2c[2]; // Parent to child pipe
    };

    // This interface is used by the child processes and called
    // once per datum.
    template<class D, class S>
    class ILearner {
    public:
      virtual ~ILearner() {}
      virtual S LearnFromDatum(const D& datum, bool learn) = 0;
      virtual void SaveModel() = 0;
    };

    struct SharedObject {
      SharedObject() : update_mutex(1), counter_mutex(1), counter(0) {} 
      boost::interprocess::interprocess_semaphore update_mutex;
      boost::interprocess::interprocess_semaphore counter_mutex;
      unsigned counter;
    };
    extern SharedObject* shared_object;

    /// XXX: We never delete these objects
    template <class T>
    T* get_shared_memory() {
      /*std::cerr << "Creating shared memory named " << shared_memory_name << std::endl;
      auto shm = new boost::interprocess::shared_memory_object(boost::interprocess::create_only, shared_memory_name.c_str(), boost::interprocess::read_write);
      shm->truncate(sizeof(T));
      auto region = new boost::interprocess::mapped_region (*shm, boost::interprocess::read_write);*/
      auto region = new boost::interprocess::mapped_region(boost::interprocess::anonymous_shared_memory(sizeof(T)));
      void* addr = region->get_address();
      T* obj = new (addr) SharedObject();
      return obj;
    }

    // Some simple functions that do IO to/from pipes.
    // These are used to send data from child processes
    // to the parent process or vice/versa.
    template <class T>
    T read_data(int pipe) {
      T v;
      int err = read(pipe, (void*)&v, sizeof(T));
      DYNET_ASSERT(err != -1, "Failed to read data from pipe in multi-processing");
      return v;
    }

    template <class T>
    void write_data(int pipe, const T& v) {
      int err = write(pipe, (void*)&v, sizeof(T));
      DYNET_ASSERT(err != -1, "Failed to write data to pipe in multi-processing");
    }

    std::string generate_queue_name();
    std::string generate_shared_memory_name();

    dynet::real sum_values(const std::vector<dynet::real>& values);
    dynet::real mean(const std::vector<dynet::real>& values);

    std::string elapsed_time_string(const timespec& start, const timespec& end);

    unsigned spawn_children(std::vector<Workload>& workloads);
    std::vector<Workload> create_workloads(unsigned num_children);

    // Called by the parent to process a chunk of data
    template <class S>
    S run_data_set(std::vector<unsigned>::iterator begin, std::vector<unsigned>::iterator end, const std::vector<Workload>& workloads,
        boost::interprocess::message_queue& mq, const WorkloadHeader& header) {
      const unsigned num_children = workloads.size();

      // Tell all the children to start up
      for (unsigned cid = 0; cid < num_children; ++cid) {
        bool cont = true;
        write_data(workloads[cid].p2c[1], cont);
        write_data(workloads[cid].p2c[1], header);
      }

      // Write all the indices to the queue for the children to process
      for (auto curr = begin; curr != end; ++curr) {
        unsigned i = *curr;
        mq.send(&i, sizeof(i), 0);
        if (stop_requested) {
          break;
        }
      }

      // Send a bunch of stop messages to the children
      for (unsigned cid = 0; cid < num_children; ++cid) {
        unsigned stop = -1U;
        mq.send(&stop, sizeof(stop), (stop_requested ? 1 : 0));
      }

      // Wait for each child to finish training its load
      std::vector<S> losses(num_children);
      for(unsigned cid = 0; cid < num_children; ++cid) {
        losses[cid] = read_data<S>(workloads[cid].c2p[0]);
      }

      S total_loss = S();
      for (S& datum_loss : losses) {
        total_loss += datum_loss;
      }
      return total_loss;
    }

    template<class D, class S>
    void run_parent(const std::vector<D>& train_data, const std::vector<D>& dev_data, ILearner<D, S>* learner,
       std::vector<Workload>& workloads, unsigned num_iterations, unsigned dev_frequency, unsigned report_frequency) {
      const unsigned num_children = workloads.size();
      boost::interprocess::message_queue mq(boost::interprocess::open_or_create, queue_name.c_str(), 10000, sizeof(unsigned));
      std::vector<unsigned> train_indices(train_data.size());
      std::iota(train_indices.begin(), train_indices.end(), 0);

      std::vector<unsigned> dev_indices(dev_data.size());
      std::iota(dev_indices.begin(), dev_indices.end(), 0);

      S best_dev_loss = S();
      bool first_dev_run = true;
      for (unsigned iter = 0; iter < num_iterations && !stop_requested; ++iter) {
        // Shuffle the training data indices
        std::shuffle(train_indices.begin(), train_indices.end(), *rndeng);

        S train_loss = S();

        std::vector<unsigned>::iterator begin = train_indices.begin();
        while (begin != train_indices.end()) {
          std::vector<unsigned>::iterator end = begin + dev_frequency;
          if (end > train_indices.end()) {
            end = train_indices.end();
          }


          std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
          double fractional_iter = iter + 1.0 * distance(train_indices.begin(), end) / train_indices.size();
          S batch_loss = run_data_set<S>(begin, end, workloads, mq, {false, end == train_indices.end(), report_frequency});
          std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
          double seconds_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000000.0;
          train_loss += batch_loss;
          std::cerr << fractional_iter << "\t" << "loss = " << batch_loss << " (" << seconds_elapsed << "s)" << std::endl;

          if (stop_requested) {
            break;
          }

          S dev_loss = run_data_set<S>(dev_indices.begin(), dev_indices.end(), workloads, mq, {true, false, report_frequency});
          bool new_best = (first_dev_run || dev_loss < best_dev_loss);
          first_dev_run = false;
          std::cerr << fractional_iter << "\t" << "dev loss = " << dev_loss << (new_best ? " (New best!)" : "") << std::endl;
          if (stop_requested) {
            break;
          }
          if (new_best) {
            learner->SaveModel();
            best_dev_loss = dev_loss;
          }

          begin = end;
        }
      }

      // Kill all children one by one and wait for them to exit
      for (unsigned cid = 0; cid < num_children; ++cid) {
        bool cont = false;
        write_data(workloads[cid].p2c[1], cont);
        wait(NULL);
      }
    }

    template <class D, class S>
    int run_child(unsigned cid, ILearner<D, S>* learner, Trainer* trainer,
        std::vector<Workload>& workloads, const std::vector<D>& train_data,
        const std::vector<D>& dev_data) {
      const unsigned num_children = workloads.size();
      DYNET_ASSERT(cid >= 0 && cid < num_children, "Bad child ID " << cid << " in run_child()");
      unsigned i;
      unsigned priority;
      boost::interprocess::message_queue::size_type recvd_size;
      boost::interprocess::message_queue mq(boost::interprocess::open_or_create, queue_name.c_str(), 10000, sizeof(unsigned));
      while (true) {
        // Check if the parent wants us to exit
        bool cont = read_data<bool>(workloads[cid].p2c[0]);
        if (cont == 0) {
          break;
        }

        // Check if we're running on the training data or the dev data 
        WorkloadHeader header = read_data<WorkloadHeader>(workloads[cid].p2c[0]);

        // Run the actual training loop
        S total_loss = S();
        S batch_loss = S();
        unsigned batch_counter = 0;
        while (true) {
          mq.receive(&i, sizeof(unsigned), recvd_size, priority);
          if (i == -1U) {
            break;
          }

          DYNET_ASSERT(i < (header.is_dev_set ? dev_data.size() : train_data.size()), "Out-of-bounds ID in MP dev/train set");
          const D& datum = (header.is_dev_set ? dev_data[i] : train_data[i]);
          S datum_loss = learner->LearnFromDatum(datum, !header.is_dev_set);
          total_loss += datum_loss;
          batch_loss += datum_loss;
          batch_counter++;

          bool do_update = !header.is_dev_set && cid == 0;
          unsigned counter = 0;
          if (!header.is_dev_set) {
            shared_object->counter_mutex.wait();
            counter = ++shared_object->counter;
            if (do_update) { shared_object->counter = 0; }
            shared_object->counter_mutex.post();
          }
          if (do_update && trainer != nullptr) {
            shared_object->update_mutex.wait();
            trainer->update(1.0 / counter); 
            shared_object->update_mutex.post();
          }
          if (batch_counter == header.report_frequency) {
            if (cid == 0) {
              std::cerr << (header.is_dev_set ? "dev" : "train") << " loss: " << batch_loss << std::endl;
            }
            batch_loss = S();
            batch_counter = 0;
          }
        }
        if (header.end_of_epoch && trainer != nullptr) {
          trainer->update_epoch();
        }

        // Let the parent know that we're done and return the loss value
        write_data(workloads[cid].c2p[1], total_loss);
      }
      return 0;
    }

    template<class D, class S>
    void run_multi_process(unsigned num_children, ILearner<D, S>* learner, Trainer* trainer, const std::vector<D>& train_data,
        const std::vector<D>& dev_data, unsigned num_iterations, unsigned dev_frequency, unsigned report_frequency) {
      queue_name = generate_queue_name();
      boost::interprocess::message_queue::remove(queue_name.c_str());
      boost::interprocess::message_queue::remove(queue_name.c_str());
      shared_memory_name = generate_shared_memory_name();
      shared_object = get_shared_memory<SharedObject>();
      std::vector<Workload> workloads = create_workloads(num_children);
      unsigned cid = spawn_children(workloads);
      if (cid < num_children) {
        run_child(cid, learner, trainer, workloads, train_data, dev_data);
        exit(0);
      }
      else {
        run_parent(train_data, dev_data, learner, workloads, num_iterations, dev_frequency, report_frequency);
      }
    }

    template<class D, class S>
    void run_single_process(ILearner<D, S>* learner, Trainer* trainer, const std::vector<D>& train_data,
        const std::vector<D>& dev_data, unsigned num_iterations, unsigned dev_frequency, unsigned report_frequency, unsigned batch_size) {
      std::vector<unsigned> train_indices(train_data.size());
      std::iota(train_indices.begin(), train_indices.end(), 0);

      std::vector<unsigned> dev_indices(dev_data.size());
      std::iota(dev_indices.begin(), dev_indices.end(), 0);

      S best_dev_loss = S();
      bool first_dev_run = true;
      unsigned batch_counter = 0;
      for (unsigned iter = 0; iter < num_iterations && !stop_requested; ++iter) {
        // Shuffle the training data indices
        std::shuffle(train_indices.begin(), train_indices.end(), *rndeng);

        S train_loss = S();

        unsigned data_processed = 0;
        unsigned data_until_report = report_frequency;
        std::vector<unsigned>::iterator begin = train_indices.begin();
        while (begin != train_indices.end()) {
          std::vector<unsigned>::iterator end = begin + dev_frequency;
          if (end > train_indices.end()) {
            end = train_indices.end();
          }
          S batch_loss;
          for (auto it = begin; it != end; ++it) {
            unsigned i = *it;
            DYNET_ASSERT(i < train_data.size(), "Out-of-bounds ID in train set for multiprocessing");
            const D& datum = train_data[i];
            S datum_loss = learner->LearnFromDatum(datum, true);
            batch_loss += datum_loss;
            train_loss += datum_loss;
            if (++batch_counter == batch_size) {
              trainer->update(1.0 / batch_size);
              batch_counter = 0;
            }
            data_processed++;

            if (--data_until_report == 0) {
              data_until_report = report_frequency;
              double fractional_iter = iter + 1.0 * data_processed / train_indices.size();
              std::cerr << fractional_iter << "\t" << "loss = " << batch_loss << std::endl;
              batch_loss = S();
            }
          }

          if (stop_requested) {
            break;
          }

          S dev_loss;
          for (auto it = dev_indices.begin(); it != dev_indices.end(); ++it) {
            unsigned i = *it;
            DYNET_ASSERT(i < dev_data.size(), "Out-of-bounds ID in dev set for multiprocessing");
            const D& datum = dev_data[i];
            S datum_loss = learner->LearnFromDatum(datum, false);
            dev_loss += datum_loss;
          }
          bool new_best = (first_dev_run || dev_loss < best_dev_loss);
          first_dev_run = false;
          double fractional_iter = iter + 1.0 * data_processed / train_indices.size();
          std::cerr << fractional_iter << "\t" << "dev loss = " << dev_loss << (new_best ? " (New best!)" : "") << std::endl;
          if (stop_requested) {
            break;
          }
          trainer->update_epoch();
          if (new_best) {
            learner->SaveModel();
            best_dev_loss = dev_loss;
          }

          begin = end;
        }
      }
    }

    void cleanup(const std::vector<Workload>& workloads);
    
    template<class D, class S>
    S run_simple_parent(const std::vector<D>& train_data, ILearner<D, S>* learner, std::vector<Workload>& workloads) {
      const unsigned num_children = workloads.size();
      boost::interprocess::message_queue mq(boost::interprocess::open_or_create, queue_name.c_str(), 10000, sizeof(unsigned));
      std::vector<unsigned> train_indices(train_data.size());
      std::iota(train_indices.begin(), train_indices.end(), 0);

      S train_loss = S();

      std::vector<unsigned>::iterator begin = train_indices.begin();
      std::vector<unsigned>::iterator end = train_indices.end();
      S batch_loss = run_data_set<S>(begin, end, workloads, mq, {false, true, (unsigned)-1});
      train_loss += batch_loss;

      // Kill all children one by one and wait for them to exit
      for (unsigned cid = 0; cid < num_children; ++cid) {
        bool cont = false;
        write_data(workloads[cid].p2c[1], cont);
        wait(NULL);
      }

      return train_loss;
    }

    template<class D, class S>
    S run_mp_minibatch(unsigned num_children, ILearner<D, S>* learner, const std::vector<D>& data) {
      queue_name = generate_queue_name();
      boost::interprocess::message_queue::remove(queue_name.c_str());
      boost::interprocess::message_queue::remove(queue_name.c_str());
      shared_memory_name = generate_shared_memory_name();
      shared_object = get_shared_memory<SharedObject>();
      std::vector<Workload> workloads = create_workloads(num_children);
      std::vector<D> dev_data;
      Trainer* trainer = nullptr;
      unsigned cid = spawn_children(workloads);
      if (cid < num_children) {
        run_child(cid, learner, trainer, workloads, data, dev_data);
        exit(0);
      }
      else {
        S return_value = run_simple_parent(data, learner, workloads);
        cleanup(workloads);
        return return_value;
      }
    }
    
    template<class D, class S>
    S run_sp_minibatch_trainer(ILearner<D, S>* learner, Trainer* inputTrainer, const std::vector<D>& data) {
      Trainer* trainer = inputTrainer;
      S total_loss;
      for (unsigned i = 0; i < data.size(); ++i) {
        const D& datum = data[i];
        S datum_loss = learner->LearnFromDatum(datum, (trainer != nullptr));
        total_loss += datum_loss;

        if (trainer != nullptr) {
          trainer->update();
        }
      }
      return total_loss;
    }

    template<class D, class S>
    S run_mp_minibatch_trainer(unsigned num_children, ILearner<D, S>* learner, Trainer* inputTrainer, const std::vector<D>& data) {
      queue_name = generate_queue_name();
      boost::interprocess::message_queue::remove(queue_name.c_str());
      boost::interprocess::message_queue::remove(queue_name.c_str());
      shared_memory_name = generate_shared_memory_name();
      shared_object = get_shared_memory<SharedObject>();
      std::vector<Workload> workloads = create_workloads(num_children);
      std::vector<D> dev_data;
      Trainer* trainer = inputTrainer;
      unsigned cid = spawn_children(workloads);
      if (cid < num_children) {
        run_child(cid, learner, trainer, workloads, data, dev_data);
        exit(0);
      }
      else {
        S return_value = run_simple_parent(data, learner, workloads);
        cleanup(workloads);
        return return_value;
      }
    }
  }
}
#endif // !_WINDOWS
