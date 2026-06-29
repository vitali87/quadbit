// Validates that the read-swizzle XOR (off ^= ((off>>7)&3)<<4) EXACTLY matches the
// hardware TMA CU_TENSOR_MAP_SWIZZLE_64B write layout. All-ones matmul data can't
// catch a wrong swizzle (every byte identical), so here global memory is filled with
// distinct bytes value=p&0xff at logical offset p=row*BKH+col; after a swizzled TMA
// load, logical p must live at smem[swizzle(p)]. If every position matches, the
// formula inverts TMA's swizzle and matmul_fp4_swz is computing the right product.

#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define BM 64
#define BKH 64
#define TILE (BM * BKH)

__global__ void __launch_bounds__(128) probe(const __grid_constant__ CUtensorMap tmap, int *wrong) {
    __shared__ __align__(128) uint8_t smem[TILE];
    __shared__ __align__(8) uint64_t bar;
    int tid = threadIdx.x;
    uint32_t sa = __cvta_generic_to_shared(&smem[0]);
    uint32_t ba = __cvta_generic_to_shared(&bar);

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(ba));
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();
    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(ba), "r"((uint32_t)TILE));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%2, %3}], [%4];" ::"r"(sa), "l"(&tmap), "r"(0), "r"(0), "r"(ba));
    }
    asm volatile("{\n\t.reg .pred p;\n\tW:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}\n" ::"r"(ba));
    __syncthreads();

    // logical p must be at smem[swizzle(p)] with value p&0xff
    for (int p = tid; p < TILE; p += 128) {
        int phys = p ^ (((p >> 7) & 3) << 4);
        if (smem[phys] != (uint8_t)(p & 0xff)) atomicAdd(wrong, 1);
    }
}

int main() {
    uint8_t *dA;
    int *dWrong, hWrong = 0;
    cudaMalloc(&dA, TILE);
    cudaMalloc(&dWrong, sizeof(int));
    uint8_t *hA = (uint8_t *)malloc(TILE);
    for (int p = 0; p < TILE; p++) hA[p] = (uint8_t)(p & 0xff);
    cudaMemcpy(dA, hA, TILE, cudaMemcpyHostToDevice);
    cudaMemset(dWrong, 0, sizeof(int));

    CUtensorMap tmap;
    uint64_t gdim[2] = {BKH, BM};
    uint64_t gstride[1] = {BKH};
    uint32_t bdim[2] = {BKH, BM};
    uint32_t estride[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, dA, gdim, gstride, bdim, estride,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("map failed: %s\n", s); return 1; }

    probe<<<1, 128>>>(tmap, dWrong);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
    cudaMemcpy(&hWrong, dWrong, sizeof(int), cudaMemcpyDeviceToHost);
    printf("swz_probe: %s (%d wrong of %d) -- read-swizzle %s TMA 64B layout\n",
           hWrong == 0 ? "PASS" : "FAIL", hWrong, TILE, hWrong == 0 ? "MATCHES" : "DOES NOT MATCH");
    return 0;
}
