#include "head_norm.cuh"
#include "util.cuh"
#include "compat.cuh"

#define MAX_HEAD_DIM 256
#if defined(USE_ROCM)
#define WARP_SIZE (warpSize)
#define FULL_MASK 0xffffffffffffffffULL
#else
#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu
#endif
#define MAX_WARPS (MAX_HEAD_DIM / 32)

__global__ void head_norm_kernel
(
    const half* __restrict__ x,
    const half* __restrict__ w,
    const half* __restrict__ b,
    half* __restrict__ y,
    const float epsilon,
    const float r_dim,
    const int rows,
    const int num_heads,
    const int head_dim,
    const bool rms
)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = DIVIDE(head_dim, 2 * WARP_SIZE);

    int t = threadIdx.x;
    const half* x_ptr = x + num_heads * head_dim * blockIdx.y + head_dim * blockIdx.x;
          half* y_ptr = y + num_heads * head_dim * blockIdx.y + head_dim * blockIdx.x;
    const half* w_ptr = w + head_dim * blockIdx.x;
    const half* b_ptr = b + head_dim * blockIdx.x;
    const half2* x_ptr2 = (const half2*) x_ptr;
          half2* y_ptr2 = (      half2*) y_ptr;
    const half2* w_ptr2 = (const half2*) w_ptr;
    const half2* b_ptr2 = (const half2*) b_ptr;

    __shared__ float sums[MAX_WARPS];
    float itemf[2];
    float sum = 0.0f;

    // RMS Norm

    if (rms)
    {
        half2 h01 = ((half2*)x_ptr)[t];
        float f0 = __half2float(__low2half(h01));
        float f1 = __half2float(__high2half(h01));
        f0 = fmaxf(-65504.0f, fminf(f0, 65504.0f));
        f1 = fmaxf(-65504.0f, fminf(f1, 65504.0f));
        itemf[0] = f0;
        itemf[1] = f1;
        sum = fma(f0, f0, sum);
        sum = fma(f1, f1, sum);

        // Shuffle to sum across lanes

        for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(FULL_MASK, sum, offset);
        if (lane_id == 0) sums[warp_id] = sum;
        __syncthreads();

        // Sum of partial sums

        sum = 0.0f;
        for(int i = 0; i < num_warps; ++i) sum += sums[i];

        // Get 1/sqrt(variance)

        float rsvar = rsqrtf(sum * r_dim + epsilon);

        // Normalize x, scaling by w

        half2 w01 = w_ptr2[t];
        float n0 = itemf[0] * __half2float(__low2half(w01)) * rsvar;
        float n1 = itemf[1] * __half2float(__high2half(w01)) * rsvar;
        half2 nh = __halves2half2(__float2half_rn(n0), __float2half_rn(n1));
        if (b) nh = __hadd2(nh, b_ptr2[t]);  // Optional bias
        y_ptr2[t] = nh;
    }

    // LayerNorm

    else
    {
        half2 h01 = ((half2*)x_ptr)[t];
        float f0 = __half2float(__low2half(h01));
        float f1 = __half2float(__high2half(h01));
        f0 = fmaxf(-65504.0f, fminf(f0, 65504.0f));
        f1 = fmaxf(-65504.0f, fminf(f1, 65504.0f));
        itemf[0] = f0;
        itemf[1] = f1;
        sum += f0;
        sum += f1;

        // Shuffle to sum across lanes

        for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(FULL_MASK, sum, offset);
        if (lane_id == 0) sums[warp_id] = sum;
        __syncthreads();

        // Sum of partial sums

        sum = 0.0f;
        for(int i = 0; i < num_warps; ++i) sum += sums[i];

        // Compute mean

        float mean = sum * r_dim;

        // Compute square of distance to mean

        sum = 0.0f;
        itemf[0] -= mean;
        itemf[1] -= mean;
        sum = fma(itemf[0], itemf[0], sum);
        sum = fma(itemf[1], itemf[1], sum);

        // Shuffle to sum across lanes

        for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(FULL_MASK, sum, offset);
        if (lane_id == 0) sums[warp_id] = sum;
        __syncthreads();

        // Sum of partial sums

        sum = 0.0f;
        for(int i = 0; i < num_warps; ++i) sum += sums[i];

        // Get 1/sqrt(variance)

        float rsvar = rsqrtf(sum * r_dim + epsilon);

        // Normalize x, scaling by w

        half2 w01 = w_ptr2[t];
        float n0 = itemf[0] * __half2float(__low2half(w01)) * rsvar;
        float n1 = itemf[1] * __half2float(__high2half(w01)) * rsvar;
        half2 nh = __halves2half2(__float2half_rn(n0), __float2half_rn(n1));
        if (b) nh = __hadd2(nh, b_ptr2[t]);  // Optional bias
        y_ptr2[t] = nh;
    }
}

void head_norm_cuda
(
    cudaStream_t stream,
    const half* x,
    const half* w,
    const half* b,
    half* y,
    const float epsilon,
    bool rms,
    const int rows,
    const int num_heads,
    const int head_dim,
    Graph* graph,
    int label
)
{
    dim3 blockDim, gridDim;
    blockDim.x = head_dim / 2;
    gridDim.x = num_heads;
    gridDim.y = rows;

    float r_dim = 1.0f / (float) head_dim;

    head_norm_kernel<<<gridDim, blockDim, 0, stream>>>(x, w, b, y, epsilon, r_dim, rows, num_heads, head_dim, rms);
    if (graph) graph->attach_label(stream, label, 0);
}

void head_norm_cuda_update_x
(
    Graph* graph,
    int label,
    void* x
)
{
    graph->update_param_ptr(label, 0, 0, x);
}

void head_norm_cuda_update_y
(
    Graph* graph,
    int label,
    void* y
)
{
    graph->update_param_ptr(label, 0, 3, y);
}
