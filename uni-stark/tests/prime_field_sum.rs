use std::borrow::Borrow;
use std::ops::Mul;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, Field, PrimeField64};
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{prove, verify, StarkConfig};
use rand::thread_rng;

/// For testing the public values feature

pub struct FibonacciAir {}

impl<F: Field> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}

impl<F, AB> Air<AB> for FibonacciAir
where
    F: Field,
    AB: AirBuilderWithPublicValues<F = F>,
{
    // We'll assume distinct points
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];

        let x3_pi = pis[2];
        let y3_pi = pis[3];

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &FibonacciRow<AB::Var> = (*local).borrow();
        let next: &FibonacciRow<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.curve_coeff_a_weierstrass, a);
        when_first_row.assert_eq(local.curve_coeff_b_weierstrass, b);

        // Verify that (x1, y1), (x2, y2), and (x3, y3) lie on the curve
        when_first_row.assert_eq(
            local.point1_y.clone().into().square(),
            local.point1_x.clone().into().exp_u64(3)
                + a.clone().into() * local.point1_x.clone()
                + b.clone().into(),
        );
        when_first_row.assert_eq(
            local.point2_y.clone().into().square(),
            local.point2_x.clone().into().exp_u64(3)
                + a.clone().into() * local.point2_x.clone()
                + b.clone().into(),
        );
        when_first_row.assert_eq(
            y3_pi.clone().into().square(),
            x3_pi.clone().into().exp_u64(3)
                + a.clone().into() * x3_pi.clone().into()
                + b.clone().into(),
        );

        let mut when_transition = builder.when_transition();

        // Load inputs (x1, y1) and (x2, y2)
        let x1 = local.point1_x;
        let y1 = local.point1_y;
        let x2 = local.point2_x;

        // check lambda
        when_transition.assert_eq(AB::Expr::one(), next.tmp2.mul(next.tmp3));
        when_transition.assert_eq(next.tmp1.mul(next.tmp2), next.tmp4);

        let x3 = next.tmp4.into().square() - x1 - x2;
        let y3 = next.tmp4 * (x1 - x3.clone()) - y1;

        // Check the resulting output (x3, y3)
        when_transition.assert_eq(next.point3_x, x3.clone());
        when_transition.assert_eq(next.point3_y, y3.clone());

        let mut last_transition = builder.when_last_row();

        // check against public value result
        last_transition.assert_eq(local.point3_x, x3_pi);
        last_transition.assert_eq(local.point3_y, y3_pi);
    }
}

/// TODO parametrize trace by public and private inputs
pub fn generate_trace_rows<F: PrimeField64>(a: F, b: F, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace =
        RowMajorMatrix::new(vec![F::zero(); n * NUM_FIBONACCI_COLS], NUM_FIBONACCI_COLS);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<FibonacciRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    let point1_x = F::from_canonical_u64(1925856373);
    let point1_y = F::from_canonical_u64(487954017);
    let point2_x = F::from_canonical_u64(1838047255);
    let point2_y = F::from_canonical_u64(1978829721);
    let point3_x = F::zero();
    let point3_y = F::zero();

    rows[0] = FibonacciRow::new(
        a,
        b,
        F::zero(),
        F::zero(),
        F::zero(),
        F::zero(),
        point1_x,
        point1_y,
        point2_x,
        point2_y,
        point3_x,
        point3_y,
    );

    let x1 = rows[0].point1_x;
    let y1 = rows[0].point1_y;
    let x2 = rows[0].point2_x;
    let y2 = rows[0].point2_y;

    // Calculate lambda = (y2 - y1) / (x2 - x1)
    rows[1].tmp1 = y2 - y1;
    rows[1].tmp2 = (x2 - x1).inverse();
    rows[1].tmp3 = x2 - x1;
    let lambda = rows[1].tmp1 * rows[1].tmp2;
    rows[1].tmp4 = lambda;

    // Calculate the resulting x3 and y3
    let x3 = lambda * lambda - x1 - x2;
    let y3 = lambda * (x1 - x3) - y1;

    rows[1].point3_x = x3;
    rows[1].point3_y = y3;

    trace
}

const NUM_FIBONACCI_COLS: usize = 12;

pub struct FibonacciRow<F> {
    pub curve_coeff_a_weierstrass: F,
    pub curve_coeff_b_weierstrass: F,
    pub tmp1: F,
    pub tmp2: F,
    pub tmp3: F,
    pub tmp4: F,
    pub point1_x: F,
    pub point1_y: F,
    pub point2_x: F,
    pub point2_y: F,
    pub point3_x: F,
    pub point3_y: F,
}

impl<F> FibonacciRow<F> {
    const fn new(
        a: F,
        b: F,
        tmp1: F,
        tmp2: F,
        tmp3: F,
        tmp4: F,
        point1_x: F,
        point1_y: F,
        point2_x: F,
        point2_y: F,
        point3_x: F,
        point3_y: F,
    ) -> FibonacciRow<F> {
        FibonacciRow {
            curve_coeff_a_weierstrass: a,
            curve_coeff_b_weierstrass: b,
            tmp1,
            tmp2,
            tmp3,
            tmp4,
            point1_x,
            point1_y,
            point2_x,
            point2_y,
            point3_x,
            point3_y,
        }
    }
}

impl<F> Borrow<FibonacciRow<F>> for [F] {
    fn borrow(&self) -> &FibonacciRow<F> {
        debug_assert_eq!(self.len(), NUM_FIBONACCI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibonacciRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

type Val = BabyBear;
type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    FieldMerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

/// TODO setup property based test for valid and invalid inputs
/// Use the curve
/// ? a=Mod(2,p)
/// %8 = Mod(2, 2013265921)
/// ? Es = ellinit([a^4, a^6], a);
/// of order ellorder(Es,[Mod(1449395754, 2013265921), Mod(1781265000, 2013265921)])
/// %14 = 2013229300
///
/// generator: [[Mod(1449395754, 2013265921), Mod(1781265000, 2013265921)]]
/// two random points on the curve
/// [Mod(1925856373, 2013265921), Mod(487954017, 2013265921)]
/// [Mod(1838047255, 2013265921), Mod(1978829721, 2013265921)]
/// sum of two points above:
/// [Mod(987670558, 2013265921), Mod(401169798, 2013265921)]
#[test]
fn test_public_value() {
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft {};
    let a = BabyBear::from_canonical_u64(16); // a
    let b = BabyBear::from_canonical_u64(64); // b
    let trace = generate_trace_rows::<Val>(a, b, 2);
    let fri_config = FriConfig {
        log_blowup: 2,
        num_queries: 28,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let config = MyConfig::new(pcs);
    let mut challenger = Challenger::new(perm.clone());
    let pis = vec![
        a,                                       // a
        b,                                       // b
        BabyBear::from_canonical_u64(987670558), // x3
        BabyBear::from_canonical_u64(401169798), // y3
    ];
    let proof = prove(&config, &FibonacciAir {}, &mut challenger, trace, &pis);
    let mut challenger = Challenger::new(perm);
    verify(&config, &FibonacciAir {}, &mut challenger, &proof, &pis).expect("verification failed");
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "assertion `left == right` failed: constraints had nonzero value")]
fn test_incorrect_public_value() {
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        &mut thread_rng(),
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft {};
    let fri_config = FriConfig {
        log_blowup: 2,
        num_queries: 28,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    let a = BabyBear::from_canonical_u64(16); // a
    let b = BabyBear::from_canonical_u64(64); // b
    let trace = generate_trace_rows::<Val>(a, b, 2);
    let pcs = Pcs::new(dft, val_mmcs, fri_config);
    let config = MyConfig::new(pcs);
    let mut challenger = Challenger::new(perm.clone());
    let pis = vec![
        BabyBear::from_canonical_u64(0),
        BabyBear::from_canonical_u64(1),
        BabyBear::from_canonical_u64(123_123), // incorrect result
    ];
    prove(&config, &FibonacciAir {}, &mut challenger, trace, &pis);
}
