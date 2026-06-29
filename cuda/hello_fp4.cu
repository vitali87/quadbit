// Raw-PTX track, milestone 0: validate the block-scaled FP4 mma.sync on sm_120
// hand-written (no CubeCL). One warp issues one m16n8k64 mxf4nvf4 MMA with all-ones
// FP4 operands (e2m1 1.0 = 0x2, packed byte 0x22, u32 0x22222222) and unit ue8m0
// scales (2^0 = bits 127, packed 0x7f7f7f7f). Every output must equal K = 64.
// This pins the exact PTX (scale operand layout) before building the full matmul.

#include <cstdio>
#include <cstdint>

__global__ void hello_fp4(float *out) {
    // all-ones fp4 fragments: A = 4 u32/lane, B = 2 u32/lane
    uint32_t a0 = 0x22222222u, a1 = 0x22222222u, a2 = 0x22222222u, a3 = 0x22222222u;
    uint32_t b0 = 0x22222222u, b1 = 0x22222222u;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0, d1, d2, d3;
    // unit ue8m0 scales (every byte = 127 => 2^0); byte/thread selectors = 0
    uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;
    uint16_t z = 0;

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14},{%15,%16}, {%17},{%18,%19};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));

    int lane = threadIdx.x & 31;
    out[lane * 4 + 0] = d0;
    out[lane * 4 + 1] = d1;
    out[lane * 4 + 2] = d2;
    out[lane * 4 + 3] = d3;
}

int main() {
    const int N = 32 * 4;
    float *d_out, h_out[N];
    cudaMalloc(&d_out, N * sizeof(float));
    hello_fp4<<<1, 32>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    int wrong = 0;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != 64.0f) wrong++;
    }
    printf("hello_fp4: out[0]=%.1f out[1]=%.1f (expected 64)  %s (%d wrong of %d)\n",
           h_out[0], h_out[1], wrong == 0 ? "PASS" : "FAIL", wrong, N);
    cudaFree(d_out);
    return 0;
}
