//! Minimal CubeCL kernel: prove the Rust + CubeCL + CUDA toolchain runs on SM120.
//! Each element gets +1.0; inputs are 0..n so output[i] must equal i+1 (exact check).

use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn add_one<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + F::new(1.0);
    }
}

fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let n = data.len();

    let input = client.create_from_slice(f32::as_bytes(&data));
    let output = client.empty(n * core::mem::size_of::<f32>());

    let threads = 256u32;
    unsafe {
        add_one::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static((n as u32).div_ceil(threads), 1, 1),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(input, n),
            ArrayArg::from_raw_parts(output.clone(), n),
        );
    }

    let bytes = client.read_one(output).unwrap();
    let result = f32::from_bytes(&bytes);

    let mut wrong = 0;
    for (i, v) in result.iter().enumerate() {
        if *v != i as f32 + 1.0 {
            wrong += 1;
        }
    }
    println!("runtime: {:?}", R::name(&client));
    println!(
        "n = {n}, output[0] = {}, output[{}] = {}",
        result[0],
        n - 1,
        result[n - 1]
    );
    println!(
        "correctness: {} ({wrong} wrong, expected output[i] == i + 1)",
        if wrong == 0 { "PASS" } else { "FAIL" }
    );
}

fn main() {
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "cpu")]
    run::<cubecl::cpu::CpuRuntime>(&Default::default());
    #[cfg(not(any(feature = "cuda", feature = "cpu")))]
    panic!("build with --features cuda (on a GPU) or --features cpu");
}
