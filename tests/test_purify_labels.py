import unittest
import numpy as np
import pandas as pd

from purify_labels_oof import purify_labels, FINDINGS, DEFAULT_ALPHA


class TestPurifyLabels(unittest.TestCase):
    def setUp(self):
        self.uids = [f"study_{i:04d}" for i in range(100)]
        self.gold_uids = self.uids[:10]  # First 10 are gold

        # Synthetic LLM labels in [0, 1]
        np.random.seed(42)
        llm_data = {"StudyInstanceUID": self.uids}
        oof_data = {"StudyInstanceUID": self.uids}
        gold_data = {"StudyInstanceUID": self.gold_uids}

        for col in FINDINGS:
            llm_data[col] = np.random.uniform(0.1, 0.9, size=100)
            oof_data[col] = np.random.uniform(0.1, 0.9, size=100)
            gold_data[col] = np.random.choice([0.0, 1.0], size=10)

        self.llm_df = pd.DataFrame(llm_data)
        self.oof_df = pd.DataFrame(oof_data)
        self.gold_df = pd.DataFrame(gold_data)

    def test_purification_mixing(self):
        """Test soft mixture formula on non-gold studies."""
        purified = purify_labels(self.llm_df, self.oof_df)
        self.assertEqual(len(purified), 100)
        self.assertEqual(list(purified.columns), ["StudyInstanceUID"] + FINDINGS)

        # Non-gold check
        for col in ["Synovitis", "PF OA", "Lateral OA"]:
            alpha = DEFAULT_ALPHA[col]
            expected = (1.0 - alpha) * self.llm_df[col].iloc[50] + alpha * self.oof_df[col].iloc[50]
            actual = purified[col].iloc[50]
            self.assertAlmostEqual(actual, expected, places=5)

    def test_gold_preservation(self):
        """Test that Gold studies retain exact ground truth values."""
        purified = purify_labels(self.llm_df, self.oof_df, gold_df=self.gold_df)
        gold_purified = purified.set_index("StudyInstanceUID").loc[self.gold_uids]
        gold_target = self.gold_df.set_index("StudyInstanceUID")

        for col in FINDINGS:
            np.testing.assert_array_almost_equal(
                gold_purified[col].values,
                gold_target[col].values,
                decimal=6
            )


if __name__ == "__main__":
    unittest.main()
