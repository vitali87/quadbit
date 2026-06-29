// Derive the scale_b layout (B's 8 columns x 2 K-blocks). Mirror of scale_probe:
// A all 1.0 (both kblocks), B kblock0 (K0-31)=1.0, kblock1 (K32-63)=2.0, scale_a unit.
// baseline output[*][col] = 32*1 + 32*2 = 96; perturb scale_b (lane,byte) -> exp1:
// col's kb0 -> 128, kb1 -> 160. Read row 0's 8 columns (lanes 0-3, regs d0/d1).

#include <cstdio>
#include <cstdint>

#define KB 32
__global__ void probe(float *out, int bidB, int tidB) {
    __shared__ __align__(16) uint8_t a_s[16 * KB];
    __shared__ __align__(16) uint8_t b_s[8 * KB];
    int lane = threadIdx.x & 31;
    int L = blockIdx.x >> 2, b = blockIdx.x & 3;

    for (int i = lane; i < 16 * KB; i += 32) a_s[i] = 0x22;                       // A all 1.0
    for (int i = lane; i < 8 * KB; i += 32) b_s[i] = (i % KB) < 16 ? 0x22 : 0x44; // B kb0=1.0 kb1=2.0
    __syncwarp();

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bkblk = (lane >> 3) & 1;
    uint32_t aa = __cvta_generic_to_shared(&a_s[arow * KB + acblk * 16]);
    uint32_t ba = __cvta_generic_to_shared(&b_s[nrow * KB + bkblk * 16]);
    uint32_t a0, a1, a2, a3, bf0, bf1;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(aa));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];"
                 : "=r"(bf0), "=r"(bf1) : "r"(ba));

    uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;
    if (lane == L) sb = (sb & ~(0xffu << (b * 8))) | (0x80u << (b * 8));
    uint16_t bb16 = (uint16_t)bidB, tb16 = (uint16_t)tidB, z = 0;

    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14},{%15,%16}, {%17},{%18,%19};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bf0), "r"(bf1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f),
          "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(bb16), "h"(tb16));

    // row 0 occupies lanes 0..3; col = (lane%4)*2 + {0,1} via d0,d1
    if ((lane >> 2) == 0) {
        out[blockIdx.x * 8 + (lane & 3) * 2 + 0] = d0;
        out[blockIdx.x * 8 + (lane & 3) * 2 + 1] = d1;
    }
}

int main() {
    float *d_out, *h_out;
    int P = 128;
    cudaMalloc(&d_out, P * 8 * sizeof(float));
    h_out = (float *)malloc(P * 8 * sizeof(float));
    for (int tid = 0; tid < 4; tid++)
        for (int bid = 0; bid < 4; bid++) {
            cudaMemset(d_out, 0, P * 8 * sizeof(float));
            probe<<<P, 32>>>(d_out, bid, tid);
            if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
            cudaMemcpy(h_out, d_out, P * 8 * sizeof(float), cudaMemcpyDeviceToHost);
            int hits = 0;
            for (int p = 0; p < P; p++) {
                int L = p >> 2, by = p & 3;
                for (int c = 0; c < 8; c++) {
                    float v = h_out[p * 8 + c];
                    if (v > 100.f) {
                        int kb = (v > 144.f) ? 1 : 0;
                        if (hits < 40) printf("  bid=%d tid=%d  lane=%2d byte=%d -> col=%d kb=%d (%.0f)\n",
                                              bid, tid, L, by, c, kb, v);
                        hits++;
                    }
                }
            }
            if (hits) printf("[bid=%d tid=%d] %d hits\n", bid, tid, hits);
        }
    printf("scale_probe_b done\n");
    return 0;
}
