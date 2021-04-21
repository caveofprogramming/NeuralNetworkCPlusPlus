#pragma once

#include <functional>
#include <future>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace cop
{
    template <typename T>
    class ThreadPool
    {
    private:
        std::thread submissionThread_;
        std::mutex mtx_;
        std::condition_variable cond_;
        std::vector<std::function<T()>> tasks_;
        std::queue<std::shared_future<T>> futures_;
        int numberThreads_ = 1;

    public:
        ThreadPool(int threads) : numberThreads_{threads}
        {
        }

        void submit(std::function<T()> work)
        {
            tasks_.push_back(work);
        }

        void start()
        {
            std::thread t([this]() {
                for (auto task : tasks_)
                {
                    std::unique_lock<std::mutex> lock(mtx_);
                    cond_.wait(lock, [this]() {
                        return int(futures_.size()) + 1 < numberThreads_ || numberThreads_ == 1;
                    });

                    std::shared_future<T> f = async(std::launch::async, task);
                    futures_.push(f);

                    if(numberThreads_ == 1)
                    {
                        f.wait();
                    }

                    lock.unlock();
                    cond_.notify_one();
                }
            });

            submissionThread_ = std::move(t);
        }

        void awaitComplete()
        {
            if (submissionThread_.joinable())
                submissionThread_.join();
        }

        T get()
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cond_.wait(lock, [this]() { return !futures_.empty(); });

            std::shared_future<T> f = futures_.front();

            futures_.pop();

            lock.unlock();
            cond_.notify_one();

            return f.get();
        }
    };
}