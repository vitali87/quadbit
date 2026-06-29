//! Bisect: does CubeCL TMA work at all on sm_120 for a plain f32 element? This is
//! the CubeCL `tensormap_load` runtime test verbatim (f32, [16,32] tile from a
//! [64,64] tensor at col 8). If this PASSES, sm_120 TMA is fine and the
//! illegal-instruction in tma_probe is the e2m1x2 descriptor dtype (fix: TMA the
//! packed data viewed as u32). If this FAILS too, CubeCL's sm_120 TMA codegen is
//! the obstacle and must be patched/upgraded.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;
use cubecl::ir::features::Tma;

#[cube(launch)]
fn tma_f32<N: Size>(input: &TensorMap<f32, Tiled>, output: &mut Array<Vector<f32, N>>) {
    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_async_proxy_shared();
    let mut stage = SharedMemory::<Vector<f32, N>>::new_aligned(32usize * 16, 128usize);

    let type_size = f32::type_size();
    let expected = select(UNIT_POS == 0, comptime!(32 * 16) as u32 * type_size as u32, 0u32);
    if UNIT_POS == 0 {
        barrier.tma_load_2d(input, &mut stage.to_slice_mut(), 0, 8);
    }
    let token = barrier.arrive_and_expect_tx(1, expected);
    barrier.wait(token);

    let out_pos = UNIT_POS_Y * 32 + UNIT_POS_X;
    output[out_pos as usize] = stage[out_pos as usize];
}

fn main() {
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(not(feature = "cuda"))]
    panic!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    if !client.features().tma.contains(Tma::Base) {
        println!("TMA not supported; skipping");
        return;
    }
    let values: Vec<f32> = (0..64 * 64).map(|it| it as f32).collect();
    let handle = client.create_from_slice(f32::as_bytes(&values));
    let input = unsafe { TensorArg::from_raw_parts(handle, [64, 1].into(), [64, 64].into()) };
    let out = client.empty(16 * 32 * core::mem::size_of::<f32>());

    tma_f32::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 16),
        1,
        TensorMapArg::new(
            TiledArgs { tile_size: [16, 32].into() },
            input,
            f32::as_type_native_unchecked(),
        ),
        unsafe { ArrayArg::from_raw_parts(out.clone(), 32 * 16) },
    );

    let bytes = client.read_one(out).unwrap();
    let got = f32::from_bytes(&bytes);
    let expected: Vec<f32> = (0..16).flat_map(|i| i * 64..i * 64 + 32).map(|it| (it + 8) as f32).collect();
    let wrong = got.iter().zip(expected.iter()).filter(|(a, b)| a != b).count();
    println!("runtime: {:?}", R::name(&client));
    println!("tma_f32: {} ({wrong} wrong of {})", if wrong == 0 { "PASS" } else { "FAIL" }, 16 * 32);
}
