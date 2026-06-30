// Derive the .sp::ordered_metadata metadata operand layout for m16n8k128 -- needed to
// support ARBITRARY 2:4 pruning (the kernel currently hardcodes 0x44444444 = pairs {0,1}
// of every 4, which only matches weights pruned to that fixed pattern). The metadata u32
// `e` per lane encodes which 2-of-4 b16-pairs are kept. Probe the ROW mapping: A all 1.0;
// B[k] = 1.0 if (k%8<4) else 2.0; baseline metadata 0x44444444 selects k%8<4 => out=64.
// Set ONE lane L's metadata to 0xeeeeeeee (selects k%8>=4 => value 2.0 => out=128). The
// rows that flip 64->128 are the rows lane L's metadata controls. 32 probes (one per lane).

#include <cstdio>
#include <cstdint>

#define AROWB 32
#define BROWB 64

__global__ void probe(float *out) {
    __shared__ __align__(16) uint8_t a_s[16 * AROWB];
    __shared__ __align__(16) uint8_t b_s[8 * BROWB];
    int lane = threadIdx.x & 31;
    int L = blockIdx.x;  // which lane gets perturbed metadata

    for (int i = lane; i < 16 * AROWB; i += 32) a_s[i] = 0x22;  // A all 1.0
    // B[col][k]: fp4 byte holds 2 k's (lo=2p, hi=2p+1). value = (k%8<4)?1.0(0x2):2.0(0x4)
    for (int i = lane; i < 8 * BROWB; i += 32) {
        int p = i % BROWB;            // byte index within row = pair index
        int klo = 2 * p, khi = 2 * p + 1;
        uint8_t lo = (klo % 8 < 4) ? 0x2 : 0x4, hi = (khi % 8 < 4) ? 0x2 : 0x4;
        b_s[i] = lo | (hi << 4);
    }
    __syncwarp();

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bsub = (lane >> 3) & 3;
    uint32_t aa = __cvta_generic_to_shared(&a_s[arow * AROWB + acblk * 16]);
    uint32_t ba = __cvta_generic_to_shared(&b_s[nrow * BROWB + bsub * 16]);
    uint32_t a0, a1, a2, a3, bf0, bf1, bf2, bf3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(aa));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(bf0), "=r"(bf1), "=r"(bf2), "=r"(bf3) : "r"(ba));

    uint32_t meta = (lane == L) ? 0xeeeeeeeeu : 0x44444444u;
    uint32_t sa = 0x38383838u, sb = 0x38383838u;
    uint16_t z = 0;
    float d0, d1, d2, d3;
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bf0), "r"(bf1), "r"(bf2), "r"(bf3),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f),
          "r"(meta), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));

    int r = lane >> 2;
    if ((lane & 3) == 0) { out[L * 16 + r] = d0; out[L * 16 + r + 8] = d2; }
}

int main() {
    int P = 32;
    float *d_out, *h_out = (float *)malloc(P * 16 * sizeof(float));
    cudaMalloc(&d_out, P * 16 * sizeof(float));
    probe<<<P, 32>>>(d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
    cudaMemcpy(h_out, d_out, P * 16 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("sp_meta_probe raw: L=0 rows0-7:");
    for (int r = 0; r < 8; r++) printf(" %.0f", h_out[0 * 16 + r]);
    printf("\n  per-lane min/max over 16 rows (baseline expect 64; 0xee row -> 128):\n");
    for (int L = 0; L < P; L++) {
        for(int r=0;r<16;r++) if(h_out[L*16+r] > 80.f) printf("  lane %2d -> row %2d (val %.0f)\n", L, r, h_out[L*16+r]);
    }
    free(h_out); cudaFree(d_out);
    return 0;
}
