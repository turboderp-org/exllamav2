#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.h"
#include "ext_tp.h"
#include "cpp/util.h"

#define cuda_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

ExtTPContext::ExtTPContext
(
    std::vector<std::tuple<int, int, int>> _kv_split,
    std::vector<std::tuple<int, int, int>> _id_split,
    std::vector<std::tuple<int, int, int>> _vc_split,
    std::vector<std::tuple<int, int, int>> _rs_split,
    std::vector<std::tuple<int, int, int>> _q_split,
    std::vector<torch::Tensor> _pinned_temp,
    std::vector<cudaStream_t> _streams
) :
    kv_split(_kv_split),
    id_split(_id_split),
    vc_split(_vc_split),
    rs_split(_rs_split),
    q_split(_q_split),
    streams(_streams)
{
    for (const auto &pt : _pinned_temp)
    {
        void* ptp = (void*) pt.data_ptr();
        pinned_temp.push_back(ptp);
        pinned_size = pt.numel() * pt.element_size();
    }

    for (int i = 0; i < streams.size(); ++i)
        if (streams[i]) all_devices.push_back(i);

    sync_events.resize(streams.size());

    for (int i = 0; i < streams.size(); ++i)
    {
        if (!streams[i]) continue;
        cudaSetDevice(i);
        cuda_check(cudaEventCreateWithFlags(&sync_events[i], cudaEventDisableTiming));
    }

    #ifdef TP_MULTITHREADED

        int numdevs = all_devices.size();
        thread_pool = new ThreadPool(numdevs);

    #endif

    cudaHostAlloc((void**)&tp_data, sizeof(ExtTPData), cudaHostAllocMapped);
    init_tp_data(tp_data);

//    comms.resize(all_devices.size());
//    ncclCommInitAll(&comms[0], all_devices.size(), &all_devices[0]);
//    comms_index.resize(streams.size());
//    for (int i = 0; i < all_devices.size(); ++i)
//        comms_index[all_devices[i]] = i;

}

ExtTPContext::~ExtTPContext()
{
    #ifdef TP_MULTITHREADED
        delete thread_pool;
    #endif

//    for (int i = 0; i < comms.size(); ++i)
//        ncclCommDestroy(comms[i]);

    cudaFreeHost(tp_data);
}

uintptr_t make_tp_context
(
    std::vector<std::tuple<int, int, int>> kv_split,
    std::vector<std::tuple<int, int, int>> id_split,
    std::vector<std::tuple<int, int, int>> vc_split,
    std::vector<std::tuple<int, int, int>> rs_split,
    std::vector<std::tuple<int, int, int>> q_split,
    std::vector<torch::Tensor> pinned_temp,
    std::vector<uintptr_t> streams
)
{
    std::vector<cudaStream_t> streams_;
    for (int i = 0; i < streams.size(); ++i)
        streams_.push_back(reinterpret_cast<cudaStream_t> (streams[i]));

    ExtTPContext* ctx = new ExtTPContext
    (
        kv_split,
        id_split,
        vc_split,
        rs_split,
        q_split,
        pinned_temp,
        streams_
    );

    return reinterpret_cast<uintptr_t> (ctx);
}

void free_tp_context
(
    uintptr_t tp_context
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
    delete ctx;
}

void tp_broadcast
(
    uintptr_t tp_context,
    int buffer,
    torch::Tensor source,
    int broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int dim,
    int t_device
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    size_t size = source.numel() * 2;
    TORCH_CHECK(size <= ctx->pinned_size, "Temporary tensor is too small")

    void* source_g = NULL;

    int src_dev = source.device().index();

    if (src_dev >= 0)
    {
        cudaSetDevice(src_dev);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

        source_g = (void*) source.data_ptr();
        cuda_check(cudaMemcpyAsync(ctx->pinned_temp[buffer], source_g, size, cudaMemcpyDeviceToHost, stream));
    }

    std::vector<std::tuple<int, int, int>> split;
    switch(broadcast_type)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    for (int i = 0; i < split.size(); ++i)
    {
        int dev = std::get<0>(split[i]);
        if (t_device != -1 && t_device != dev) continue;

        void* target = (void*) targets[i].data_ptr();
        if (target == source_g) continue;

        cudaSetDevice(dev);
        cudaStream_t stream = ctx->streams[dev];
        cuda_check(cudaMemcpyAsync(target, ctx->pinned_temp[buffer], size, cudaMemcpyHostToDevice, stream));
    }
}

void tp_gather
(
    uintptr_t tp_context,
    int buffer,
    const std::vector<torch::Tensor> &inputs,
    int broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int broadcast_type_target,
    int dim,
    int t_device
)
{
    tp_gather_barrier
    (
        tp_context,
        buffer,
        inputs,
        broadcast_type,
        targets,
        broadcast_type_target,
        dim,
        t_device,
        nullptr
    );
}

void tp_gather_barrier
(
    uintptr_t tp_context,
    int buffer,
    const std::vector<torch::Tensor> &inputs,
    int broadcast_type,
    const std::vector<torch::Tensor> &targets,
    int broadcast_type_target,
    int dim,
    int t_device,
    Barrier* barrier
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    std::vector<std::tuple<int, int, int>> split;
    switch(broadcast_type)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    int out_rows = inputs[0].size(0);
    int out_cols = std::get<2>(split[split.size() - 1]) * dim;
    int esize = inputs[0].element_size();

    for (int i = 0; i < split.size(); ++i)
    {
        int dev = std::get<0>(split[i]);
        if (t_device != -1 && t_device != dev) continue;

        uint8_t* src = (uint8_t*) inputs[i].data_ptr();
        int src_cols = inputs[i].size(1);
        uint8_t* dst = ((uint8_t*) ctx->pinned_temp[buffer]) + std::get<1>(split[i]) * esize * dim;

        cudaSetDevice(dev);
        cuda_check(cudaMemcpy2DAsync
        (
            dst,
            out_cols * esize,
            src,
            src_cols * esize,
            src_cols * esize,
            out_rows,
            cudaMemcpyDeviceToHost,
            ctx->streams[dev]
        ));
    }

    if (broadcast_type_target == -2) return;

    if (barrier)
        barrier->arrive_and_wait();

    tp_cross_device_barrier(tp_context, broadcast_type, t_device);

    if (broadcast_type_target == -1) return;

    size_t size = targets[0].numel() * 2;

    switch(broadcast_type_target)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    for (int i = 0; i < split.size(); ++i)
    {
        int dev = std::get<0>(split[i]);
        if (t_device != -1 && t_device != dev) continue;

        void* target = (void*) targets[i].data_ptr();

        cudaSetDevice(dev);
        cudaStream_t stream = ctx->streams[dev];
        cuda_check(cudaMemcpyAsync(target, ctx->pinned_temp[buffer], size, cudaMemcpyHostToDevice, stream));
    }
}

void tp_cross_device_barrier
(
    uintptr_t tp_context,
    int broadcast_type,
    int t_device,
    int stage,
    int next_stage
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    std::vector<std::tuple<int, int, int>> split;
    switch(broadcast_type)
    {
        case BROADCAST_KV: split = ctx->kv_split; break;
        case BROADCAST_ID: split = ctx->id_split; break;
        case BROADCAST_VC: split = ctx->vc_split; break;
        case BROADCAST_RS: split = ctx->rs_split; break;
        case BROADCAST_Q: split = ctx->q_split; break;
    }

    if (stage == -1)
    {
        stage = ctx->tp_data->next_stage;
        ctx->tp_data->next_stage = (ctx->tp_data->next_stage + 1) % MAX_SYNC_STAGES;
        next_stage = ctx->tp_data->next_stage;
    }

    uint32_t* sync = ctx->tp_data->sync[stage];
    uint32_t* sync_next = ctx->tp_data->sync[next_stage];

//    for (int i = 0; i < ctx->all_devices.size(); ++i)
//    {
//        int dev = ctx->all_devices[i];
//        // if (t_device != -1 && t_device != dev) continue;
//        cross_device_barrier_cuda
//        (
//            ctx->streams[dev],
//            sync,
//            sync_next,
//            ctx->all_devices.size(),
//            i
//        );
//    }

//    for (int i = 0; i < ctx->all_devices.size(); ++i)
//    {
//        int dev = ctx->all_devices[i];
//        cudaSetDevice(dev);
//        // if (t_device != -1 && t_device != dev) continue;
//        cudaStreamSynchronize(ctx->streams[dev]);
//    }

    for (int i = 0; i < ctx->all_devices.size(); ++i)
    {
        int dev_i = ctx->all_devices[i];
        cudaSetDevice(dev_i);
        if (i > 0) {
            int j = i - 1;
            int dev_j = ctx->all_devices[j];
            cuda_check(cudaStreamWaitEvent(ctx->streams[dev_i], ctx->sync_events[dev_j], 0));
        }
        cuda_check(cudaEventRecord(ctx->sync_events[dev_i], ctx->streams[dev_i]));
    }
    for (int i = 0; i < ctx->all_devices.size() - 1; ++i)
    {
        int dev_i = ctx->all_devices[i];
        cudaSetDevice(dev_i);
        cuda_check(cudaStreamWaitEvent(ctx->streams[dev_i], ctx->sync_events[ctx->all_devices.back()], 0));
    }
}

//void tp_all_reduce_nccl
//(
//    uintptr_t tp_context,
//    const std::vector<torch::Tensor> &tensors
//)
//{
//    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);
//
//    ncclGroupStart();
//
//    for (int i = 0; i < tensors.size(); ++i)
//    {
//        int dev = tensors[i].device().index();
//        int comms_i = ctx->comms_index[dev];
//
//        ncclAllReduce
//        (
//            tensors[i].data_ptr(),
//            tensors[i].data_ptr(),
//            tensors[i].numel(),
//            ncclFloat16,
//            ncclSum,
//            ctx->comms[comms_i],
//            ctx->streams[dev]
//        );
//    }
//
//    ncclGroupEnd();
//}

//void tp_all_reduce
//(
//    uintptr_t tp_context,
//    const std::vector<torch::Tensor> &tensors
//)

void tp_all_reduce
(
    uintptr_t tp_context,
    int buffer,
    const std::vector<torch::Tensor> &tensors,
    const std::vector<torch::Tensor> &residuals
)
{
    ExtTPContext* ctx = reinterpret_cast<ExtTPContext*> (tp_context);

    size_t size = tensors[0].numel() * tensors[0].element_size();
    size_t num = tensors.size();

    // Reduction via host buffer

    for (int i = 0; i < num; ++i)
    {
        int dev = tensors[i].device().index();
        auto torch_stream = at::cuda::getStreamFromExternal(ctx->streams[dev], dev);
        cudaSetDevice(dev);
        at::cuda::setCurrentCUDAStream(torch_stream);

        if (i > 0)
        {
            int prev_dev = tensors[i - 1].device().index();

            // Copy host buffer to current residual

            cuda_check(cudaStreamWaitEvent
            (
                ctx->streams[dev],
                ctx->sync_events[prev_dev],
                0
            ));

            cuda_check(cudaMemcpyAsync
            (
                residuals[i].data_ptr(),
                ctx->pinned_temp[buffer],
                size,
                cudaMemcpyHostToDevice,
                ctx->streams[dev]
            ));
        }

        // Add current tensor to current residual

        residuals[i].add_(tensors[i]);

        // Copy current residual to host buffer

        cuda_check(cudaMemcpyAsync
        (
            ctx->pinned_temp[buffer],
            residuals[i].data_ptr(),
            size,
            cudaMemcpyDeviceToHost,
            ctx->streams[dev]
        ));

        cuda_check(cudaEventRecord
        (
            ctx->sync_events[dev],
            ctx->streams[dev]
        ));
    }

    // Broadcast result

    int last_dev = tensors[num - 1].device().index();

    for (int i = 0; i < num - 1; ++i)
    {
        int dev = tensors[i].device().index();
        cudaSetDevice(dev);

        cuda_check(cudaStreamWaitEvent
        (
            ctx->streams[dev],
            ctx->sync_events[last_dev],
            0
        ));
        cuda_check(cudaMemcpyAsync
        (
            residuals[i].data_ptr(),
            ctx->pinned_temp[buffer],
            size,
            cudaMemcpyHostToDevice,
            ctx->streams[dev]
        ));
    }
}
