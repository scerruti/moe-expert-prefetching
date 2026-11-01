"""CPU tests for the k-scaling theory checks: each estimator against a case with a known answer."""
import numpy as np

from k_scaling_theory import fit_sqrt, hill, lattice_share, measure, recall_at, rmse, spacings, synthetic, tail_exponent


def test_tail_estimators_recover_a_known_exponent():
    rng = np.random.default_rng(0)
    values = rng.pareto(2.0, 2_000_000) + 1  # P(x > v) = v^-2
    assert abs(tail_exponent(values) - 2.0) < 0.1
    assert abs(hill(values) - 2.0) < 0.1


def test_exponential_scores_have_spacing_proportional_to_one_over_k():
    # The Renyi representation: gaps between the top order statistics of exponentials have mean 1/k
    rng = np.random.default_rng(0)
    gaps = spacings(rng.exponential(1.0, (20000, 60))).mean(axis=0)
    for k in (1, 2, 4, 8, 16):
        assert abs(gaps[k - 1] * k - 1) < 0.05


def test_measure_counts_a_swap_across_the_cut_off_only():
    a = np.array([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]], np.float32)
    b = np.array([[4.0, 2.0, 3.0, 1.0], [4.0, 3.0, 2.0, 1.0]], np.float32)  # ranks 2 and 3 swap, first row
    results = measure(a, b, [1, 2])
    assert results[1][0] == 0
    assert results[2][0] == 0.5
    assert results[2][1] == 0.95  # every change needs m = 1; interpolation puts the 95% target just below it


def test_no_noise_means_no_changes():
    rng = np.random.default_rng(0)
    a, b = synthetic(1000, 60, 1.0, np.array([0.0]), rng)
    assert all(rate == 0 for rate, _, _ in measure(a, b, [1, 4, 16]).values())


def test_lattice_share_separates_rounded_values_from_continuous_ones():
    rng = np.random.default_rng(0)
    step = 2.0 ** -9
    assert lattice_share(rng.integers(0, 50, 10000) * step) == 1
    assert abs(lattice_share(rng.uniform(0, 0.1, 100000)) - 0.12) < 0.01


def test_recall_counts_true_experts_inside_the_predicted_set():
    true = np.array([[0, 1, 2, 3]])       # top 2 are experts 0 and 1
    predicted = np.array([[0, 2, 1, 3]])  # expert 1 is ranked third
    assert recall_at(true, predicted, 2, 0) == 0.5
    assert recall_at(true, predicted, 2, 1) == 1


def test_fit_sqrt_recovers_a_known_curve():
    x = np.array([1.0, 4.0, 9.0, 16.0])
    m = 1 + 2 * np.sqrt(x)
    fit = fit_sqrt(x, m)
    assert np.allclose(fit, (1, 2))
    assert rmse(x, m, fit) < 1e-9


def test_near_tie_gaps_finds_each_tokens_first_changed_layer():
    from k_scaling_figures import near_tie_gaps, top4_gap

    # columns are unsorted; the gap is between the 4th and 5th largest
    probs = np.array([[.1, .4, .3, .05, .15], [.5, .2, .15, .1, .05], [.4, .3, .2, .06, .04]])
    gap = top4_gap(probs)
    assert np.allclose(gap, [.05, .05, .02])

    # token "a" changes at layers 1 and 2: only layer 1 is its first flip
    every, changed, first = near_tie_gaps(np.array([.9, .1, .2, .3]), np.array([False, True, True, True]),
                                          [("a", 0), ("a", 0), ("a", 0), ("b", 1)], np.array([0, 1, 2, 0]))
    assert len(every) == 4
    assert np.allclose(changed, [.1, .2, .3])
    assert sorted(first.tolist()) == [.1, .3]
