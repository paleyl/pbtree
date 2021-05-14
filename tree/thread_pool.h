#ifndef TREE_THREAD_POOL_H_
#define TREE_THREAD_POOL_H_

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <chrono>

#include "glog/logging.h"

namespace pbtree {

class ThreadPool {
 public:

  ThreadPool(int threads): m_is_shutdown_(false) {
    // Create the specified number of threads
    m_threads_vec_.reserve(threads);
    for (int i = 0; i < threads; ++i)
      m_threads_vec_.emplace_back(std::bind(&ThreadPool::thread_entry, this, i));
    for (int i = 0; i < threads; ++i) {
      m_thread_working_vec_.push_back(0);
    }
  }

  ~ThreadPool() {
    {
      // Unblock any threads and tell them to stop
      std::unique_lock<std::mutex> lock(m_lock_);

      m_is_shutdown_ = true;
      m_cond_var_.notify_all();
    }

    // Wait for all threads to stop
    LOG(WARNING) << "Joining threads";
    for (auto& thread : m_threads_vec_)
      thread.join();
  }

  void do_job(std::function<void(void)> func) {
    // Place a job on the queu and unblock a thread
    std::unique_lock<std::mutex> lock(m_lock_);

    m_jobs_queue_.emplace(std::move(func));
    m_cond_var_.notify_one();
  }

  bool is_jobs_queue_empty() {
    return m_jobs_queue_.empty();
  }

  uint32_t get_jobs_queue_size() {
    return m_jobs_queue_.size();
  }

  bool is_working() {
    bool result = false;
    for (unsigned int i = 0; i < m_thread_working_vec_.size(); ++i) {
      result = result || m_thread_working_vec_[i];
    }
    return result;
  }

  bool is_finished() {
    return !is_working() && is_jobs_queue_empty();
  }

  std::string get_worker_status() {
    std::string out = "Worker status: ";
    for (unsigned int i = 0; i < m_thread_working_vec_.size(); ++i) {
      out = out + std::to_string(i) + ":" +std::to_string( m_thread_working_vec_[i]) + ",";
    }
    return out;
  }

  protected:

  void thread_entry(int i) {
    std::function<void(void)> job;

    while (true) {
      {
        std::unique_lock<std::mutex> lock(m_lock_);

        while(!m_is_shutdown_ && m_jobs_queue_.empty())
          m_cond_var_.wait(lock);

        if (m_jobs_queue_.empty()) {
          // No jobs to do and we are shutting down
          LOG(WARNING) << "Thread " << i << " terminates";
          return;
        }

        // std::cout << "Thread " << i << " start a job" << std::endl;
        m_thread_working_vec_[i] = 1;
        job = std::move(m_jobs_queue_.front());
        m_jobs_queue_.pop();
        
      }

      // Do the job without holding any locks
      job();
      m_thread_working_vec_[i] = 0;
      // std::cout << "Thread " << i << " finished a job, adress " << &(m_thread_working_vec_) << " " << get_worker_status() << std::endl;
    }
  }

  // void split_thread_entry(int i) {
  //   std::function<bool(std::vector<uint64_t> &, uint64_t &, double *, double *)> job;

  //   while (true) {
  //     {
  //       std::unique_lock<std::mutex> lock(m_lock_);

  //       while(!m_is_shutdown_ && m_split_jobs_queue_.empty())
  //         m_cond_var_.wait(lock);

  //       if (m_split_jobs_queue_.empty()) {
  //         // No jobs to do and we are shutting down
  //         LOG(WARNING) << "Thread " << i << " terminates";
  //         return;
  //       }

  //       VLOG(202) << "Thread " << i << " does a job";
  //       auto job = std::move(m_split_jobs_queue_.front());
  //       m_jobs_queue_.pop();
  //     }

  //     // Do the job without holding any locks
  //     job();
  //   }
  // }

  std::mutex m_lock_;
  std::condition_variable m_cond_var_;
  bool m_is_shutdown_;
  std::queue<std::function<void(void)>> m_jobs_queue_;
  std::queue<std::function<bool(std::vector<uint64_t>&, uint64_t&, double*,
      double*)>> m_split_jobs_queue_;
  std::vector<std::thread> m_threads_vec_;
  std::vector<int> m_thread_working_vec_;
};

}  // pbtree

#endif  // TREE_THREAD_POOL_H_
