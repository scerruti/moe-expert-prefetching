import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_collection import get_prompt_text, run_collection, shuffled


def make_datasets(n=10):
    """10-example subsets shaped like the real loader output."""
    return {
        "gsm8k": [
            {"example_id": i, "question": f"q{i}", "source": "gsm8k"} for i in range(n)
        ],
        "mbpp": [{"task_id": i, "prompt": f"p{i}", "source": "mbpp"} for i in range(n)],
    }


def collect(datasets, **kwargs):
    calls = []
    summary = run_collection(
        datasets,
        process_fn=lambda run, name, pos, ex: calls.append((run, name, pos, ex)),
        show_progress=False,
        **kwargs,
    )
    return calls, summary


class DataCollectionTests(unittest.TestCase):
    def test_all_runs_cover_every_prompt(self):
        calls, summary = collect(make_datasets())

        self.assertEqual(len(calls), 5 * 2 * 10)
        self.assertEqual(summary["prompts_processed"], 100)
        self.assertEqual(summary["run_seeds"], [42, 43, 44, 45, 46])
        for run in range(5):
            for name in ["gsm8k", "mbpp"]:
                seen = [ex for r, n, _, ex in calls if r == run and n == name]
                self.assertCountEqual(seen, make_datasets()[name])

    def test_loop_nesting_order(self):
        calls, _ = collect(make_datasets(3), num_runs=2)
        sequence = [(run, name) for run, name, _, _ in calls]

        expected = [
            (r, n) for r in range(2) for n in ["gsm8k", "mbpp"] for _ in range(3)
        ]
        self.assertEqual(sequence, expected)

    def test_runs_use_different_orderings(self):
        calls, _ = collect(make_datasets())
        orders = {
            tuple(
                ex["example_id"] for r, n, _, ex in calls if r == run and n == "gsm8k"
            )
            for run in range(5)
        }
        self.assertGreater(len(orders), 1)

    def test_same_seed_is_reproducible(self):
        first, _ = collect(make_datasets(), base_seed=7)
        second, _ = collect(make_datasets(), base_seed=7)
        self.assertEqual(first, second)

    def test_shuffle_does_not_mutate_input(self):
        examples = make_datasets()["gsm8k"]
        original = list(examples)
        shuffled(examples, seed=1)
        self.assertEqual(examples, original)

    def test_get_prompt_text(self):
        data = make_datasets(1)
        self.assertEqual(get_prompt_text("gsm8k", data["gsm8k"][0]), "q0")
        self.assertEqual(get_prompt_text("mbpp", data["mbpp"][0]), "p0")


if __name__ == "__main__":
    unittest.main()
