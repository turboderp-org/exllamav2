#include "util.cuh"

void print_global_mem(const half* ptr, int rows, int columns, int stride)
{
    half* temp = (half*) calloc(1, rows * columns * sizeof(half));

    cudaDeviceSynchronize();
    cudaMemcpyAsync(temp, ptr, rows * columns * sizeof(half), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int r = 0; r < rows; ++r)
    {
        printf("%4d: ", r);
        for (int c = 0; c < columns; c++)
        {
            float v = __half2float(temp[r * stride + c]);
            printf("%10.6f", v);
            if (c < columns - 1) printf("  ");
        }
        printf("\n");
    }

    free(temp);
}
