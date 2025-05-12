#include "cache.cuh"

#include "quant/qdq_util.cuh"
#include "util.cuh"
#include "compat.cuh"
#include "../config.h"

#define THREADS 32
#define BLOCKSIZE_Q Q_CACHE_BLOCKSIZE_Q
#define SUPER_BLOCKSIZE_Q Q_CACHE_SUPER_BLOCKSIZE_Q
#define THREADS_Q (BLOCKSIZE_Q / 2)

#include "cache_q.cuh"

// The upper 8 bits of FP16 are equivalent to FP8 E5M2.
//
// The range of values typically cached seem to be in the range of +/- 16, with an exponent component (with bias) up to
// about 20. Empirically, the MSE over the whole range of observed values in the K/V cache works out the same for E4M3
// and E5M2. However, over 80% of values in the cache tensors fall within the range of -1..1, where E5M2 produces about
// a 25% lower MSE.

__device__ inline uint32_t compress(uint32_t v)
{
    uint32_t vh = (v & 0xff000000) >> 16;
    uint32_t vl = (v & 0x0000ff00) >> 8;
    return vh | vl;
}

__device__ inline uint32_t decompress(uint32_t v)
{
    uint32_t vh = (v & 0xff00) << 16;
    uint32_t vl = (v & 0x00ff) << 8;
    return vh | vl;
}

__global__ void fp16_to_fp8_kernel
(
    const half* __restrict__ pIn,
    unsigned char* __restrict__ pOut,
    int stride,
    int height,
    int min,
    int max
)
{
    int x = min + (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int y = blockIdx.y;
    if (x >= max) return;

    int4* in_ptr = (int4*) (pIn + y * stride + x);
    int2* out_ptr = (int2*) (pOut + y * stride + x);

    int4 in = *in_ptr;
    uint32_t c0 = compress(in.x);
    uint32_t c1 = compress(in.y);
    uint32_t c2 = compress(in.z);
    uint32_t c3 = compress(in.w);
    int2 out = make_int2(c0 | (c1 << 16), c2 | (c3 << 16));
    *out_ptr = out;
}

__global__ void fp8_to_fp16_kernel
(
    const unsigned char* __restrict__ pIn,
    half* __restrict__ pOut,
    int stride,
    int height,
    int min,
    int max
)
{
    int x = min + (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int y = blockIdx.y;
    if (x >= max) return;

    int2* in_ptr = (int2*) (pIn + y * stride + x);
    int4* out_ptr = (int4*) (pOut + y * stride + x);

    int2 in = *in_ptr;
    uint32_t c0 = decompress(in.x);
    uint32_t c1 = decompress(in.x >> 16);
    uint32_t c2 = decompress(in.y);
    uint32_t c3 = decompress(in.y >> 16);
    int4 out = make_int4(c0, c1, c2, c3);
    *out_ptr = out;
}

void array_fp16_to_fp8_cuda
(
    cudaStream_t stream,
    const half* pIn,
    unsigned char *pOut,
    int stride,
    int height,
    int offset,
    int width
)
{
    int min = offset;
    int max = offset + width;
    min = min / 8 * 8;
    max = min + (max - min + 7) / 8 * 8;

    dim3 blockDim, gridDim;
    blockDim.x = THREADS;
    gridDim.x = DIVIDE((max - min) / 8, THREADS);
    gridDim.y = height;

    fp16_to_fp8_kernel<<<gridDim, blockDim, 0, stream>>>(pIn, pOut, stride, height, min, max);
    // cuda_check( cudaPeekAtLastError() );
}

void array_fp8_to_fp16_cuda
(
    cudaStream_t stream,
    const unsigned char* pIn,
    half* pOut,
    int stride,
    int height,
    int offset,
    int width
)
{
    int min = offset;
    int max = offset + width;
    min = min / 8 * 8;
    max = min + (max - min + 7) / 8 * 8;

    dim3 blockDim, gridDim;
    blockDim.x = THREADS;
    gridDim.x = DIVIDE((max - min) / 8, THREADS);
    gridDim.y = height;

    fp8_to_fp16_kernel<<<gridDim, blockDim, 0, stream>>>(pIn, pOut, stride, height, min, max);
    // cuda_check( cudaPeekAtLastError() );
}

// -------------- FP16 -> Q

template <int wbits_k, int wbits_v>
__global__ void fp16_to_q_kv_paged_kernel
(
    const half* __restrict__ k_in,
    unsigned char* __restrict__ k_out,
    half* __restrict__ k_scales,
    const half* __restrict__ v_in,
    unsigned char* __restrict__ v_out,
    half* __restrict__ v_scales,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ block_table,
    int pages_per_seq,
    int page_size,
    int dim,
    int q_len
)
{
    int t = threadIdx.x;
    int kv = blockIdx.z & 1;
    const half* in = kv ? v_in : k_in;
    half* scales = kv ? v_scales : k_scales;
    unsigned char* out = kv ? v_out : k_out;

    int x = blockIdx.x;
    int y = blockIdx.z >> 1;

    int page = block_table[pages_per_seq * y + x];
    int seqlen = cache_seqlens[y];
    int vx_a = (int64_t)page_size * x;
    int px_a = (int64_t)seqlen - vx_a;
    int px_b = px_a + q_len;

    if (dim % BLOCKSIZE_Q)
    {
        while ((px_a * dim) % BLOCKSIZE_Q) px_a--;
        while ((px_b * dim) % BLOCKSIZE_Q) px_b++;
    }
    
    px_a = max(px_a, 0);
    px_b = min(px_b, page_size);

    int block_a = (page * page_size + px_a) * dim;
    int block_b = (page * page_size + px_b) * dim;

    for (int i = block_a; i < block_b; i += SUPER_BLOCKSIZE_Q)
    {
        int j = i + blockIdx.y * BLOCKSIZE_Q;
        if (j >= block_b) continue;
        if (kv)
            fp16_to_q<wbits_v>(t, in, out, scales, j, dim);
        else
            fp16_to_q<wbits_k>(t, in, out, scales, j, dim);
    }
}

template <int wbits_k, int wbits_v>
__global__ void fp16_to_q_kv_kernel
(
    const half* __restrict__ k_in,
    unsigned char* __restrict__ k_out,
    half* __restrict__ k_scales,
    const half* __restrict__ v_in,
    unsigned char* __restrict__ v_out,
    half* __restrict__ v_scales,
    int dim,
    int offset,
    int stride
)
{
    int t = threadIdx.x;
    int kv = blockIdx.z & 1;
    const half* in = kv ? v_in : k_in;
    unsigned char* out = kv ? v_out : k_out;
    half* scales = kv ? v_scales : k_scales;
    int block_offset = (offset + blockIdx.y * stride + blockIdx.x * BLOCKSIZE_Q);

    if (kv)
        fp16_to_q<wbits_v>(t, in, out, scales, block_offset, dim);
    else
        fp16_to_q<wbits_k>(t, in, out, scales, block_offset, dim);
}

void array_fp16_to_q_kv_paged_cuda
(
    cudaStream_t stream,
    const half* k_in,
    unsigned char* k_out,
    half* k_scales,
    const half* v_in,
    unsigned char* v_out,
    half* v_scales,
    int batch_size,
    int dim,
    int pages_per_seq,
    const int* cache_seqlens,
    const int* block_table,
    int page_size,
    int q_len,
    int wbits
)
{
    dim3 blockDim, gridDim;
    blockDim.x = THREADS_Q;
    gridDim.x = pages_per_seq;
    gridDim.y = SUPER_BLOCKSIZE_Q / BLOCKSIZE_Q;
    gridDim.z = batch_size * 2;

    if (wbits == 4)
        fp16_to_q_kv_paged_kernel<4, 4><<<gridDim, blockDim, 0, stream>>>
        (
            k_in, k_out, k_scales,
            v_in, v_out, v_scales,
            cache_seqlens, block_table,
            pages_per_seq, page_size,
            dim, q_len
        );
    else if (wbits == 6)
        fp16_to_q_kv_paged_kernel<8, 4><<<gridDim, blockDim, 0, stream>>>
        (
            k_in, k_out, k_scales,
            v_in, v_out, v_scales,
            cache_seqlens, block_table,
            pages_per_seq, page_size,
            dim, q_len
        );
    else if (wbits == 8)
        fp16_to_q_kv_paged_kernel<8, 8><<<gridDim, blockDim, 0, stream>>>
        (
            k_in, k_out, k_scales,
            v_in, v_out, v_scales,
            cache_seqlens, block_table,
            pages_per_seq, page_size,
            dim, q_len
        );
}

void array_fp16_to_q_kv_cuda
(
    cudaStream_t stream,
    const half* k_in,
    unsigned char* k_out,
    half* k_scales,
    const half* v_in,
    unsigned char* v_out,
    half* v_scales,
    int dim,
    int stride,
    int height,
    int offset,
    int width,
    int wbits
)
{
    dim3 blockDim, gridDim;
    blockDim.x = THREADS_Q;
    gridDim.x = width / BLOCKSIZE_Q;
    gridDim.y = height;
    gridDim.z = v_in ? 2 : 1;

    if (wbits == 4)
        fp16_to_q_kv_kernel<4, 4><<<gridDim, blockDim, 0, stream>>>(
            k_in, k_out, k_scales,
            v_in, v_out, v_scales,
            dim, offset, stride
        );
    else if (wbits == 6)
        fp16_to_q_kv_kernel<8, 4><<<gridDim, blockDim, 0, stream>>>(
            k_in, k_out, k_scales,
            v_in, v_out, v_scales,
            dim, offset, stride
        );
    else if (wbits == 8)
        fp16_to_q_kv_kernel<8, 8><<<gridDim, blockDim, 0, stream>>>(
            k_in, k_out, k_scales,
            v_in, v_out, v_scales,
            dim, offset, stride
        );
}

// --------------- Q -> FP16

template <int wbits_k, int wbits_v>
__global__ void q_to_fp16_kv_paged_kernel
(
    const unsigned char* __restrict__ k_in,
    const half* __restrict__ k_scales,
    half* __restrict__ k_out,
    const unsigned char* __restrict__ v_in,
    const half* __restrict__ v_scales,
    half* __restrict__ v_out,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ block_table,
    int pages_per_seq,
    int page_size,
    int dim
)
{
    int t = threadIdx.x;
    int kv = blockIdx.z & 1;
    const unsigned char* in = kv ? v_in : k_in;
    const half* scales = kv ? v_scales : k_scales;
    half* out = kv ? v_out : k_out;

    int x = blockIdx.x;
    int y = blockIdx.z >> 1;
    int page = block_table[pages_per_seq * y + x];
    int seqlen = cache_seqlens[y];
    if (!seqlen) return;

    int vx_a = (int64_t)page_size * x;
    int vx_b = min(vx_a + page_size, seqlen);

    if (dim < BLOCKSIZE_Q)
    {
        while ((vx_a * dim) % BLOCKSIZE_Q) vx_a--;
        while ((vx_b * dim) % BLOCKSIZE_Q) vx_b++;
    }

    int vnum = max(vx_b - vx_a, 0);
    int block_a = (page * page_size) * dim;
    int block_b = (page * page_size + vnum) * dim;

    for (int i = block_a; i < block_b; i += SUPER_BLOCKSIZE_Q)
    {
        int j = i + blockIdx.y * BLOCKSIZE_Q;
        if (j >= block_b) continue;
        if (kv)
            q_to_fp16<wbits_v>(t, in, scales, out, j, dim);
        else
            q_to_fp16<wbits_k>(t, in, scales, out, j, dim);
    }
}

template <int wbits_k, int wbits_v>
__global__ void q_to_fp16_kv_kernel
(
    const unsigned char* __restrict__ k_in,
    const half* __restrict__ k_scales,
    half* __restrict__ k_out,
    const unsigned char* __restrict__ v_in,
    const half* __restrict__ v_scales,
    half* __restrict__ v_out,
    int dim,
    int offset,
    int stride
)
{
    int t = threadIdx.x;
    int kv = blockIdx.z & 1;
    const unsigned char* in = kv ? v_in : k_in;
    const half* scales = kv ? v_scales : k_scales;
    half* out = kv ? v_out : k_out;
    int block_offset = (offset + blockIdx.y * stride + blockIdx.x * BLOCKSIZE_Q);

    if (kv)
        q_to_fp16<wbits_v>(t, in, scales, out, block_offset, dim);
    else
        q_to_fp16<wbits_k>(t, in, scales, out, block_offset, dim);
}

void array_q_to_fp16_kv_paged_cuda
(
    cudaStream_t stream,
    const unsigned char* k_in,
    const half* k_scales,
    half* k_out,
    const unsigned char* v_in,
    const half* v_scales,
    half* v_out,
    int batch_size,
    int dim,
    int pages_per_seq,
    const int* cache_seqlens,
    const int* block_table,
    int page_size,
    int wbits
)
{
    dim3 blockDim, gridDim;
    blockDim.x = THREADS_Q;
    gridDim.x = pages_per_seq;
    gridDim.y = SUPER_BLOCKSIZE_Q / BLOCKSIZE_Q;
    gridDim.z = batch_size * 2;

    if (wbits == 4)
        q_to_fp16_kv_paged_kernel<4, 4><<<gridDim, blockDim, 0, stream>>>
        (
            k_in, k_scales, k_out,
            v_in, v_scales, v_out,
            cache_seqlens, block_table,
            pages_per_seq, page_size,
            dim
        );
    else if (wbits == 6)
        q_to_fp16_kv_paged_kernel<8, 4><<<gridDim, blockDim, 0, stream>>>
        (
            k_in, k_scales, k_out,
            v_in, v_scales, v_out,
            cache_seqlens, block_table,
            pages_per_seq, page_size,
            dim
        );
    else if (wbits == 8)
        q_to_fp16_kv_paged_kernel<8, 8><<<gridDim, blockDim, 0, stream>>>
        (
            k_in, k_scales, k_out,
            v_in, v_scales, v_out,
            cache_seqlens, block_table,
            pages_per_seq, page_size,
            dim
        );
}

void array_q_to_fp16_kv_cuda
(
    cudaStream_t stream,
    const unsigned char* k_in,
    const half* k_scales,
    half* k_out,
    const unsigned char* v_in,
    const half* v_scales,
    half* v_out,
    int dim,
    int stride,
    int height,
    int offset,
    int width,
    int wbits
)
{
    dim3 blockDim, gridDim;
    blockDim.x = THREADS_Q;
    gridDim.x = width / BLOCKSIZE_Q;
    gridDim.y = height;
    gridDim.z = v_in ? 2 : 1;

    if (wbits == 4)
        q_to_fp16_kv_kernel<4, 4><<<gridDim, blockDim, 0, stream>>>(
            k_in, k_scales, k_out,
            v_in, v_scales, v_out,
            dim, offset, stride
        );
    else if (wbits == 6)
        q_to_fp16_kv_kernel<8, 4><<<gridDim, blockDim, 0, stream>>>(
            k_in, k_scales, k_out,
            v_in, v_scales, v_out,
            dim, offset, stride
        );
    else if (wbits == 8)
        q_to_fp16_kv_kernel<8, 8><<<gridDim, blockDim, 0, stream>>>(
            k_in, k_scales, k_out,
            v_in, v_scales, v_out,
            dim, offset, stride
        );
}