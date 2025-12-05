use omnidsp::{Backend, Config};
use omnidsp::traits::dft::{Dft, DftDirection, DftSpec};
use num_complex::Complex;

#[test]
fn test_end_to_end_dft_architecture() {
    // 1. Setup Config
    // "I want a default setup"
    let config = Config::new();

    // 2. Create Backend
    // "Give me the manager"
    let backend = Backend::new(config);

    // 3. Create Spec
    // "I want a Forward DFT of size 4"
    let spec = DftSpec {
        length: 4,
        direction: DftDirection::Forward,
    };

    // 4. Create Plan
    // "Manager, find the best way to do this"
    // We verify that Backend implements Dft<Complex<f32>>
    let plan = backend
        .create_plan(spec)
        .expect("Failed to create plan");

    // 5. Prepare Data (DC Signal)
    let input = vec![
        Complex::new(1.0f32, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(1.0, 0.0),
    ];
    let mut output = vec![Complex::new(0.0, 0.0); 4];

    // 6. Execute
    // "Plan, do the work"
    plan.process(&input, &mut output).expect("Execution failed");

    // 7. Verify
    // RustFFT is unnormalized. Forward DFT of [1,1,1,1] is [4,0,0,0].
    let expected = vec![
        Complex::new(4.0f32, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    
    assert_eq!(output, expected);
    
    println!("Success: DC Input {:?} transformed to {:?} (Correct)", input, output);
}
