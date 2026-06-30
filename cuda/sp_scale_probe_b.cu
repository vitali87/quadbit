// Derive the ue4m3 scale_vec::4X scale_B layout for the SPARSE block-scaled FP4 mma.sp
// (m16n8k128) empirically -- the last piece for arbitrary-scale (real NVFP4) deployment;
// so far the kernel is validated only at unit scales (0x38). Method mirrors the dense
// scale_probe: one warp, one mma.sp, A loaded via the matmul_sp ldmatrix path. The 64
// nonzero compressed-A K split into 4 NVFP4 blocks of 16 (= the "4X" = 4 scales). Rig the
// 4 blocks to distinct fp4 values {1.0,1.5,2.0,3.0} so baseline row-sum = 16*(1+1.5+2+3)
// = 120; B all 1.0; metadata 0x44 (uniform B => metadata-agnostic sum). Each probe sets
// ONE (lane L, byte b) of scale_a to 0x40 (ue4m3 2.0), doubling whichever (row, kblock)
// it drives: delta 16->kb0, 24->kb1, 32->kb2, 48->kb3. 128 probes = 32 lanes x 4 bytes.

#include <cstdio>
#include <cstdint>

#define AROWB 32   // compressed A bytes/row (64 nonzero fp4)
#define BROWB 64   // full B bytes/row (128 fp4)

__global__ void probe(float *out) {
    __shared__ __align__(16) uint8_t a_s[16 * AROWB];
    __shared__ __align__(16) uint8_t b_s[8 * BROWB];
    int lane = threadIdx.x & 31;
    int L = blockIdx.x >> 2, b = blockIdx.x & 3;

    // A: row r, byte j -> block kb=j/8 gets fp4 v[kb] in both nibbles. v={1,1.5,2,3}.
    const uint8_t v[4] = {0x2, 0x3, 0x4, 0x5};
    for (int i = lane; i < 16 * AROWB; i += 32) a_s[i] = 0x22;  // A all 1.0
    // B: col, byte j -> K-block kb=j/16 (32 fp4 each, 4X) gets fp4 v[kb] in both nibbles
    for (int i = lane; i < 8 * BROWB; i += 32) { uint8_t vv = v[(i % BROWB) / 16]; b_s[i] = vv | (vv << 4); }
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

    // perturb: lane L sets scale_a byte b to 0x40 (ue4m3 2.0); all else unit 0x38
    uint32_t sb = 0x38383838u;
    if (lane == L) sb = (sb & ~(0xffu << (b * 8))) | (0x40u << (b * 8));
    uint32_t sa = 0x38383838u, meta = 0x44444444u;
    uint16_t z = 0;

    float d0, d1, d2, d3;
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bf0), "r"(bf1), "r"(bf2), "r"(bf3),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f),
          "r"(meta), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));

    if ((lane >> 2) == 0) { int c0 = (lane & 3) * 2; out[blockIdx.x * 8 + c0] = d0; out[blockIdx.x * 8 + c0 + 1] = d1; }
}

int main() {
    int P = 128;
    float *d_out, *h_out = (float *)malloc(P * 8 * sizeof(float));
    cudaMalloc(&d_out, P * 8 * sizeof(float));
    probe<<<P, 32>>>(d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
    cudaMemcpy(h_out, d_out, P * 8 * sizeof(float), cudaMemcpyDeviceToHost);

    // baseline 120; delta -> kb: 16->0, 24->1, 32->2, 48->3
    printf("sp_scale_probe_b (scale_B ue4m3 4X): (lane,byte) -> (col,kblock)\n");
    for (int cfg = 0; cfg < P; cfg++) {
        int L = cfg >> 2, b = cfg & 3;
        for (int c = 0; c < 8; c++) {
            float val = h_out[cfg * 8 + c];
            int delta = (int)(val - 120.0f + 0.5f);
            if (delta == 0) continue;
            int kb = delta == 16 ? 0 : delta == 24 ? 1 : delta == 32 ? 2 : delta == 48 ? 3 : -1;
            printf("  lane %2d byte %d -> col %d  kb %d  (out=%.0f)\n", L, b, c, kb, val);
        }
    }
    free(h_out); cudaFree(d_out);
    return 0;
}
