// Arbitrary-2:4 correctness for sparse FP4: RANDOM per-group 2:4 selection (which 2 of 4
// pairs each group keeps), packed compressed A + per-lane metadata built from the derived
// layout. Confirms the metadata nibble<->group order and half assignment. One warp,
// 16x8x128, unit ue4m3 scales (isolates metadata), f32 out.
// Derived layout: row(L)=(L&1)*8+(L>>2), half H=(L>>1)&1; lane L's u32 e = 8 nibbles for
// groups [H*8 .. H*8+8) of row R; nibble = idx0 | (idx1<<2) (the 2 kept pair-indices,
// idx0<idx1). Compressed byte cs (row): group g=cs/2, slot s=cs%2 -> logical pair
// g*4 + idx[g][s] (fp4 2*pair, 2*pair+1).

#include <cstdio>
#include <cstdint>

#define AROWB 32
#define BROWB 64

__global__ void verify(const uint8_t *A, const uint8_t *B, const uint32_t *metaLane, float *out) {
    __shared__ __align__(16) uint8_t a_s[16 * AROWB];
    __shared__ __align__(16) uint8_t b_s[8 * BROWB];
    int lane = threadIdx.x & 31;
    for (int i = lane; i < 16 * AROWB; i += 32) a_s[i] = A[i];
    for (int i = lane; i < 8 * BROWB; i += 32) b_s[i] = B[i];
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

    uint32_t e = metaLane[lane];
    uint32_t sa = 0x38383838u, sb = 0x38383838u;
    uint16_t z = 0;
    float d0, d1, d2, d3;
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bf0), "r"(bf1), "r"(bf2), "r"(bf3),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f),
          "r"(e), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));

    int r = lane >> 2, c = (lane & 3) * 2;
    out[r * 8 + c] = d0; out[r * 8 + c + 1] = d1;
    out[(r + 8) * 8 + c] = d2; out[(r + 8) * 8 + c + 1] = d3;
}

static float dfp4(uint8_t n) {
    int s = (n >> 3) & 1, e = (n >> 1) & 3, m = n & 1;
    float v = (e == 0) ? (m ? 0.5f : 0.f) : ((float)(1 << (e - 1)) * (1.f + 0.5f * m));
    return s ? -v : v;
}

int main() {
    uint8_t hA[16 * AROWB], hB[8 * BROWB];
    uint8_t idx[16][16][2];   // [row][group] -> 2 kept pair-indices (idx0<idx1)
    uint32_t metaLane[32] = {0};
    uint32_t st = 0xBEEFu;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };

    // random 2:4 per (row, group): choose 2 of {0,1,2,3}
    for (int i = 0; i < 16; i++)
        for (int g = 0; g < 16; g++) {
            int a = rnd() % 4, b;
            do { b = rnd() % 4; } while (b == a);
            if (a > b) { int t = a; a = b; b = t; }
            idx[i][g][0] = a; idx[i][g][1] = b;
        }
    // compressed A: byte cs (row i) = group cs/2, slot cs%2 -> 2 random fp4 (lo,hi)
    for (int i = 0; i < 16; i++)
        for (int cs = 0; cs < AROWB; cs++) {
            uint8_t lo = rnd() & 0xf, hi = rnd() & 0xf;
            hA[i * AROWB + cs] = lo | (hi << 4);
        }
    for (int i = 0; i < 8 * BROWB; i++) hB[i] = (uint8_t)rnd();

    // per-lane metadata: lane L -> row R=(L&1)*8+(L>>2), half H=(L>>1)&1; nibble n = group H*8+n
    for (int L = 0; L < 32; L++) {
        int R = (L & 1) * 8 + (L >> 2), H = (L >> 1) & 1;
        uint32_t e = 0;
        for (int n = 0; n < 8; n++) {
            int g = H * 8 + n;
            uint32_t nib = idx[R][g][0] | (idx[R][g][1] << 2);
            e |= nib << (n * 4);
        }
        metaLane[L] = e;
    }

    uint8_t *dA, *dB; uint32_t *dM; float *dO, hO[16 * 8];
    cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dB, sizeof(hB)); cudaMalloc(&dM, sizeof(metaLane)); cudaMalloc(&dO, sizeof(hO));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
    cudaMemcpy(dM, metaLane, sizeof(metaLane), cudaMemcpyHostToDevice);
    verify<<<1, 32>>>(dA, dB, dM, dO);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
    cudaMemcpy(hO, dO, sizeof(hO), cudaMemcpyDeviceToHost);

    // reference: out[i][j] = sum over compressed byte cs of (lo*Blo + hi*Bhi) at the
    // logical pair g*4+idx[i][g][slot]
    int wrong = 0; float maxrel = 0.f;
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 8; j++) {
            float ref = 0.f;
            for (int cs = 0; cs < AROWB; cs++) {
                int g = cs / 2, slot = cs % 2;
                int pair = g * 4 + idx[i][g][slot];
                int klo = 2 * pair, khi = 2 * pair + 1;
                uint8_t ab = hA[i * AROWB + cs];
                uint8_t bl = hB[j * BROWB + klo / 2], bh = hB[j * BROWB + khi / 2];
                float blo = dfp4((klo & 1) ? bl >> 4 : bl & 0xf);
                float bhi = dfp4((khi & 1) ? bh >> 4 : bh & 0xf);
                ref += dfp4(ab & 0xf) * blo + dfp4(ab >> 4) * bhi;
            }
            float got = hO[i * 8 + j];
            float rel = fabsf(got - ref) / (fabsf(ref) + 1.f);
            if (rel > maxrel) maxrel = rel;
            if (rel > 1e-3f) { if (wrong < 6) printf("  [%d][%d] got %.3f ref %.3f\n", i, j, got, ref); wrong++; }
        }
    printf("sp_verify_meta (RANDOM per-group 2:4, 16x8x128): %s (%d wrong, maxrel %.5f)\n",
           wrong == 0 ? "PASS" : "FAIL", wrong, maxrel);
    return 0;
}
