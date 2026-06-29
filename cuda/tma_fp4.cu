// Raw-PTX track, warp-spec milestone, step 0: validate raw TMA on sm_120 for the
// u8-described FP4 tensor. CubeCL proved TMA works for e2m1x2 IF the CUtensorMap is
// built with a u8 dtype + 16-aligned innermost coords (see quadbit-fp4-tma); this
// reproduces that with the raw driver API (cuTensorMapEncodeTiled) + raw PTX
// (cp.async.bulk.tensor.2d + mbarrier), the foundation for the warp-spec kernel.
// One block loads a BM x BKH tile of all-ones (0x22) into shared via TMA; every byte
// must come back 0x22.

#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define BM 64
#define BKH 64          // bytes per row of the tile (multiple of 16 for u8 TMA)
#define TILE (BM * BKH)

__global__ void __launch_bounds__(128) tma_probe(const __grid_constant__ CUtensorMap tmap, int *wrong) {
    __shared__ __align__(128) uint8_t smem[TILE];
    __shared__ __align__(8) uint64_t bar;

    int tid = threadIdx.x;
    uint32_t smem_addr = __cvta_generic_to_shared(&smem[0]);
    uint32_t bar_addr = __cvta_generic_to_shared(&bar);

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar_addr));
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar_addr), "r"((uint32_t)TILE));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_addr),
            "l"(&tmap), "r"(0), "r"(0), "r"(bar_addr));
    }
    // all threads wait on phase 0
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
        "@!p bra WAIT;\n\t"
        "}\n" ::"r"(bar_addr));
    __syncthreads();

    for (int i = tid; i < TILE; i += 128)
        if (smem[i] != 0x22) atomicAdd(wrong, 1);
}

int main() {
    int M = BM, Kb = BKH;
    uint8_t *dA;
    int *dWrong, hWrong = 0;
    cudaMalloc(&dA, (size_t)M * Kb);
    cudaMalloc(&dWrong, sizeof(int));
    cudaMemset(dA, 0x22, (size_t)M * Kb);
    cudaMemset(dWrong, 0, sizeof(int));

    CUtensorMap tmap;
    uint64_t gdim[2] = {(uint64_t)Kb, (uint64_t)M};   // innermost = Kb bytes, then M rows
    uint64_t gstride[1] = {(uint64_t)Kb};             // row stride in bytes (multiple of 16)
    uint32_t bdim[2] = {(uint32_t)BKH, (uint32_t)BM};
    uint32_t estride[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(
        &tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, dA, gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        const char *s; cuGetErrorString(r, &s);
        printf("cuTensorMapEncodeTiled failed: %s\n", s);
        return 1;
    }

    tma_probe<<<1, 128>>>(tmap, dWrong);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    cudaMemcpy(&hWrong, dWrong, sizeof(int), cudaMemcpyDeviceToHost);
    printf("tma_fp4: tile %dx%d bytes  %s (%d wrong of %d)\n",
           M, Kb, hWrong == 0 ? "PASS" : "FAIL", hWrong, TILE);
    cudaFree(dA); cudaFree(dWrong);
    return 0;
}
