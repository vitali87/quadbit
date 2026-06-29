// Derive the ue8m0 scale-register layout for the block-scaled FP4 mma empirically.
// One warp, one m16n8k64 mma, A/B loaded via the proven ldmatrix path. Values are
// rigged so each (row, K-block) is distinguishable: A kblock0 (K0-31)=1.0, kblock1
// (K32-63)=2.0, B all 1.0, scale_b unit. Baseline output[row] = 32*1 + 32*2 = 96.
// Each block perturbs ONE (lane L, byte b) of scale_a to exponent 1 (0x80) and we
// observe the output: row->128 means (L,b) drives that row's kblock0 (32*2+32*2... no:
// 32*2^1*1 + 32*2 = 64+64=128), row->160 means kblock1 (32 + 64*2^1=... 32+128=160).
// So 128 => (row, kb0), 160 => (row, kb1). 128 probes = 32 lanes x 4 bytes.

#include <cstdio>
#include <cstdint>

#define KB 32           // bytes per row for K=64 fp4
__global__ void probe(float *out, int bidA, int tidA) {
    __shared__ __align__(16) uint8_t a_s[16 * KB];
    __shared__ __align__(16) uint8_t b_s[8 * KB];
    int lane = threadIdx.x & 31;
    int L = blockIdx.x >> 2, b = blockIdx.x & 3;

    // A: each row = 16 bytes 0x22 (K0-31 = 1.0) then 16 bytes 0x44 (K32-63 = 2.0)
    for (int i = lane; i < 16 * KB; i += 32) a_s[i] = (i % KB) < 16 ? 0x22 : 0x44;
    for (int i = lane; i < 8 * KB; i += 32) b_s[i] = 0x22;
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

    // perturb: lane L sets scale_a byte b to 0x80 (exp 1); all else unit 0x7f
    uint32_t sa = 0x7f7f7f7fu;
    if (lane == L) sa = (sa & ~(0xffu << (b * 8))) | (0x80u << (b * 8));
    uint32_t sb = 0x7f7f7f7fu;
    uint16_t ba16 = (uint16_t)bidA, ta16 = (uint16_t)tidA, z = 0;

    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14},{%15,%16}, {%17},{%18,%19};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bf0), "r"(bf1),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f),
          "r"(sa), "h"(ba16), "h"(ta16), "r"(sb), "h"(z), "h"(z));

    int r = lane >> 2;
    if ((lane & 3) == 0) {  // one lane per row group writes the row's value
        out[blockIdx.x * 16 + r] = d0;
        out[blockIdx.x * 16 + r + 8] = d2;
    }
}

int main() {
    float *d_out, *h_out;
    int P = 128;
    cudaMalloc(&d_out, P * 16 * sizeof(float));
    h_out = (float *)malloc(P * 16 * sizeof(float));

    // sweep selectors too: try (bidA,tidA) in {0..3}x{0..3} until perturbations land
    for (int tid = 0; tid < 4; tid++)
        for (int bid = 0; bid < 4; bid++) {
            cudaMemset(d_out, 0, P * 16 * sizeof(float));
            probe<<<P, 32>>>(d_out, bid, tid);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
            cudaMemcpy(h_out, d_out, P * 16 * sizeof(float), cudaMemcpyDeviceToHost);
            int hits = 0;
            for (int p = 0; p < P; p++) {
                int L = p >> 2, by = p & 3;
                for (int r = 0; r < 16; r++) {
                    float v = h_out[p * 16 + r];
                    if (v > 100.f) {  // 128 (kb0) or 160 (kb1); baseline 96
                        int kb = (v > 144.f) ? 1 : 0;
                        if (hits < 40) printf("  bid=%d tid=%d  lane=%2d byte=%d -> row=%2d kb=%d (%.0f)\n",
                                              bid, tid, L, by, r, kb, v);
                        hits++;
                    }
                }
            }
            if (hits) printf("[bid=%d tid=%d] %d hits\n", bid, tid, hits);
        }
    printf("scale_probe done\n");
    return 0;
}
