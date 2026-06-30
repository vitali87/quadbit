// Raw-PTX track, SPARSE milestone 0: validate the 2:4-sparse block-scaled FP4
// mma.sp on sm_120 (the 2x-throughput path that can beat dense CUTLASS). m16n8k128
// sparse = A compressed to 16x64 nonzero (4 u32/lane, all 1.0) + metadata selecting
// 2 of every 4 K; B is full 128x8 (4 u32/lane, all 1.0); unit ue8m0 scales (4X for
// k128/32=4 K-blocks). 64 nonzero contributions => every output should equal 64.

#include <cstdio>
#include <cstdint>

__global__ void hello_sp(float *out) {
    uint32_t a0 = 0x22222222u, a1 = 0x22222222u, a2 = 0x22222222u, a3 = 0x22222222u;       // 16x64 nonzero fp4=1.0
    uint32_t b0 = 0x22222222u, b1 = 0x22222222u, b2 = 0x22222222u, b3 = 0x22222222u;       // 128x8 fp4=1.0
    uint32_t e = 0x44444444u;     // 2:4 metadata: each 4-bit group picks 2 of 4 positions
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0, d1, d2, d3;
    uint32_t sa = 0x38383838u, sb = 0x38383838u;  // ue4m3 unit scale (e4m3 1.0 = 0x38)
    uint16_t z = 0;

    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(e),
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
    hello_sp<<<1, 32>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    int wrong = 0;
    for (int i = 0; i < N; i++) if (h_out[i] != 64.0f) wrong++;
    printf("hello_sp: out[0]=%.1f out[1]=%.1f (expected 64)  %s (%d wrong of %d)\n",
           h_out[0], h_out[1], wrong == 0 ? "PASS" : "FAIL", wrong, N);
    cudaFree(d_out);
    return 0;
}
