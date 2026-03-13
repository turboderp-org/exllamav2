#include "ext_stloader.h"
#include "cpp/util.h"

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>

void stloader_read
(
    const char* filename,
    size_t offset,
    size_t size,
    torch::Tensor target
)
{
    c10::optional<torch::Device> device = torch::device_of(target);
    bool target_cpu = (device.has_value() && device->type() == torch::kCPU);
    cudaStream_t stream;

    // Buffers

    uint8_t* load_buffer;
    uint8_t* cuda_buffer;
    if (target_cpu)
    {
        load_buffer = (uint8_t*) target.data_ptr();
        cuda_buffer = nullptr;
    }
    else
    {
        load_buffer = (uint8_t*) calloc(1, size);
        TORCH_CHECK(load_buffer, "Can't allocate buffer for tensor");
        cuda_buffer = (uint8_t*) target.data_ptr();
        cudaSetDevice(device.value().index());
        stream = at::cuda::getCurrentCUDAStream(device.value().index()).stream();
    }

    // Synchronization

    Py_BEGIN_ALLOW_THREADS

    volatile bool load_failed = false;
    std::mutex mtx;
    std::deque<std::pair<size_t, size_t>> dq;
    std::condition_variable cv;

    // Load chunks

    auto load_worker = [&] (size_t pos_a)
    {
        FILE* file = fopen(filename, "rb");
        if (!file) goto error;

        while (pos_a < size && !load_failed)
        {
            size_t pos_b = pos_a + STLOADER_BLOCK_SIZE;
            if (pos_b > size) pos_b = size;

            #ifdef __linux__
                ssize_t br = pread(fileno(file), load_buffer + pos_a, pos_b - pos_a, offset + pos_a);
                if (br != pos_b - pos_a) goto error;
                int sr = fseek(file, offset + pos_a, SEEK_SET);
            #else
                int sr = _fseeki64(file, static_cast<__int64>(offset + pos_a), SEEK_SET);
                if (sr) goto error;
                size_t br = fread(load_buffer + pos_a, 1, pos_b - pos_a, file);
                if (br != pos_b - pos_a) goto error;
            #endif

            {
                std::lock_guard<std::mutex> lock(mtx);
                dq.push_back(std::pair<size_t, size_t>(pos_a, pos_b));
                cv.notify_one();
            }

            // DBGX3(pos_a, pos_b, br);
            pos_a += STLOADER_THREADS * STLOADER_BLOCK_SIZE;
        }

        fclose(file);
        return;

        error:
        if (file && ferror(file))
            printf("Error reading file: %s (errno: %d)\n", strerror(errno), errno);
        load_failed = true;
    };

    // Copy chunks to device

    auto copy_worker = [&] ()
    {
        cudaSetDevice(device.value().index());

        size_t total_blocks = DIVIDE(size, STLOADER_BLOCK_SIZE);
        while (total_blocks && !load_failed)
        {
            size_t pos_a, pos_b;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&dq] { return !dq.empty(); });

                auto pop = dq.front();
                dq.pop_front();
                total_blocks--;
                pos_a = std::get<0>(pop);
                pos_b = std::get<1>(pop);

                while (!dq.empty() && std::get<0>(dq.front()) == pos_b)
                {
                    pop = dq.front();
                    dq.pop_front();
                    pos_b = std::get<1>(pop);
                    total_blocks--;
                }
            }

            cudaError_t cr = cudaMemcpyAsync
            (
                cuda_buffer + pos_a,
                load_buffer + pos_a,
                pos_b - pos_a,
                cudaMemcpyHostToDevice,
                stream
            );
            if (cr != cudaSuccess)
            {
                fprintf(stderr,"GPUassert: %s\n", cudaGetErrorString(cr));
                goto error;
            }
        }
        return;

        error:
        load_failed = true;
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < STLOADER_THREADS && i * STLOADER_BLOCK_SIZE < size; ++i)
        threads.emplace_back(load_worker, i * STLOADER_BLOCK_SIZE);
    if (cuda_buffer)
        threads.emplace_back(copy_worker);
    for (auto& thread : threads)
        thread.join();

    TORCH_CHECK(!load_failed, "I/O error reading tensor");

    if (!target_cpu)
    {
        free(load_buffer);
        cudaDeviceSynchronize();
    }

    Py_END_ALLOW_THREADS
}

void tensor_remap
(
    torch::Tensor tensor,
    torch::Tensor index
)
{
    TORCH_CHECK_SHAPES(tensor, 1, index, 0, 1);
    TORCH_CHECK_DTYPE(tensor, kInt);
    TORCH_CHECK_DTYPE(index, kInt);

    int rows = tensor.size(0);
    int cols = tensor.size(1);
    uint32_t* temp = (uint32_t*) calloc(cols, sizeof(int));
    uint32_t* a = (uint32_t*) tensor.data_ptr();
    uint32_t* idx = (uint32_t*) index.data_ptr();

    for (int r = 0; r < rows; ++r)
    {
        memcpy(temp, a, sizeof(uint32_t) * cols);
        for (int c = 0; c < cols; ++c)
        {
            *a++ = temp[idx[c]];
        }
    }
    free(temp);
}

void tensor_remap_4bit
(
    torch::Tensor tensor,
    torch::Tensor index
)
{
    TORCH_CHECK_SHAPES(index, 0, tensor, 1, 8);
    TORCH_CHECK_DTYPE(tensor, kInt);
    TORCH_CHECK_DTYPE(index, kInt);

    int rows = tensor.size(0);
    int cols = index.size(0);
    uint32_t* temp = (uint32_t*) calloc(cols / 8, sizeof(int));
    uint32_t* a = (uint32_t*) tensor.data_ptr();
    uint32_t* idx = (uint32_t*) index.data_ptr();

    for (int r = 0; r < rows; ++r)
    {
        memcpy(temp, a, sizeof(uint32_t) * cols / 8);
        for (int c = 0; c < cols;)
        {
            uint32_t rv = 0;
            for (int b = 0; b < 8; ++b, ++c)
            {
                uint32_t i = idx[c];
                uint32_t v = (temp[i / 8] >> ((i & 7) * 4) & 0x0f);
                rv |= v << (b * 4);
            }
            *a++ = rv;
        }
    }
    free(temp);
}
