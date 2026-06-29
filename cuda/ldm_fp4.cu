// Raw-PTX track, milestone 1: ldmatrix-load A/B from shared into the FP4 MMA
// fragments and accumulate over K, hand-written. One warp, one 16x8 output tile,
// K=256 (4 MMA-K substeps), all-ones FP4 + unit scales => every output == K = 256.
// Validates: shared staging, m8n8.x4/.x2 b16 ldmatrix, the K-accumulation loop,
// and the block-scaled mma.sync chained as C=D. Layout (no swizzle) is the recipe
// derived in the CubeCL ldm_probe (A: arow=(lane/8%2)*8+lane%8, cblk=(lane/8)/2;
// B: nrow=lane%8, kblk=(lane/8)%2).

#include <cstdio>
#include <cstdint>

#define K 256
#define KSUB (K / 64)
#define KB (K / 2)          // e2m1x2 bytes per row
#define MMAKB 32            // bytes per MMA-K (64 fp4) slice

__global__ void ldm_fp4(float *out) {
    __shared__ __align__(16) uint8_t a_s[16 * KB];
    __shared__ __align__(16) uint8_t b_s[8 * KB];

    int lane = threadIdx.x & 31;
    // fill all-ones fp4 (0x22 = two e2m1 1.0); 32 threads cooperatively
    for (int i = lane; i < 16 * KB; i += 32) a_s[i] = 0x22;
    for (int i = lane; i < 8 * KB; i += 32) b_s[i] = 0x22;
    __syncwarp();

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7);
    int acblk = (lane >> 3) >> 1;
    int nrow = lane & 7;
    int bkblk = (lane >> 3) & 1;

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;
    uint16_t z = 0;

    for (int ks = 0; ks < KSUB; ks++) {
        int astart = arow * KB + ks * MMAKB + acblk * 16;
        int bstart = nrow * KB + ks * MMAKB + bkblk * 16;
        uint32_t aa = __cvta_generic_to_shared(&a_s[astart]);
        uint32_t ba = __cvta_generic_to_shared(&b_s[bstart]);
        uint32_t a0, a1, a2, a3, b0, b1;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(aa));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];"
                     : "=r"(b0), "=r"(b1) : "r"(ba));
        float d0, d1, d2, d3;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14},{%15,%16}, {%17},{%18,%19};"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3),
              "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
        c0 = d0; c1 = d1; c2 = d2; c3 = d3;
    }

    out[lane * 4 + 0] = c0;
    out[lane * 4 + 1] = c1;
    out[lane * 4 + 2] = c2;
    out[lane * 4 + 3] = c3;
}

int main() {
    const int N = 32 * 4;
    float *d_out, h_out[N];
    cudaMalloc(&d_out, N * sizeof(float));
    ldm_fp4<<<1, 32>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    int wrong = 0;
    for (int i = 0; i < N; i++) if (h_out[i] != (float)K) wrong++;
    printf("ldm_fp4: out[0]=%.1f out[1]=%.1f (expected %d)  %s (%d wrong of %d)\n",
           h_out[0], h_out[1], K, wrong == 0 ? "PASS" : "FAIL", wrong, N);
    cudaFree(d_out);
    return 0;
}
