#ifndef ATHENA_THREAD_POOL_H_
#define ATEHNA_THREAD_POOL_H_

#include <assert.h>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>

class ThreadPool {
public:
  ThreadPool(size_t thread_num)
      : terminate_(false), thread_num_(thread_num), complete_task_num_(0) {
    for (size_t i = 0; i < thread_num; ++i) {
      threads_.emplace_back([this] {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->mutex_);
            this->cond_.wait(lock, [this] {
              return this->terminate_ || !this->tasks_.empty();
            });

            if (this->terminate_ && this->tasks_.empty())
              return;

            task = std::move(this->tasks_.front());
            this->tasks_.pop();
          }
          task();
          complete_task_num_++;
        }
      });
    }
  }
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      terminate_ = true;
    }
    cond_.notify_all();

    for (std::thread &thread : threads_) {
      thread.join();
    }
  }

  template <class F, class... Args>
  auto Enqueue(F &&f, Args &&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (terminate_)
        throw std::runtime_error("enqueue on stopped ThreadPool");
      tasks_.emplace([task]() { (*task)(); });
    }
    cond_.notify_one();
    return res;
  }

  void Wait(int task_num) {
    while (complete_task_num_ != task_num) {
      usleep(1000);
    }
    complete_task_num_ = 0;
  }

  size_t ThreadNum() { return thread_num_; }

private:
  bool terminate_;
  size_t thread_num_;
  std::atomic_int complete_task_num_;
  std::vector<std::thread> threads_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cond_;
};
#endif
