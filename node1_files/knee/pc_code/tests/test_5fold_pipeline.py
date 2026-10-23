import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from train_5fold_knee import aggregate_5fold_oofs, evaluate_oof_performance, FINDINGS


class Test5FoldPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tag = "test_tag"
        self.n_per_fold = 20
        self.total_folds = 5

        # Create synthetic npz files for folds 0..4
        all_uids = []
        for f in range(self.total_folds):
            uids = [f"study_{f}_{i:03d}" for i in range(self.n_per_fold)]
            all_uids.extend(uids)
            preds = np.random.uniform(0.1, 0.9, size=(self.n_per_fold, len(FINDINGS))).astype(np.float32)
            truth = np.random.uniform(0.1, 0.9, size=(self.n_per_fold, len(FINDINGS))).astype(np.float32)

            fpath = os.path.join(self.test_dir, f"raptor_oof_{self.tag}_fold{f}.npz")
            np.savez(fpath, pred=preds, truth=truth, ids=np.array(uids), fold=f, nfolds=self.total_folds)

        # Create labels DataFrame
        lbl_data = {"StudyInstanceUID": all_uids}
        for col in FINDINGS:
            lbl_data[col] = np.random.choice([0.0, 1.0], size=len(all_uids))
        self.labels_path = os.path.join(self.test_dir, "labels.parquet")
        pd.DataFrame(lbl_data).to_parquet(self.labels_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_aggregate_5fold_oofs(self):
        out_parquet = os.path.join(self.test_dir, "aggregated_oof.parquet")
        df = aggregate_5fold_oofs(self.tag, oof_dir=self.test_dir, num_folds=self.total_folds, out_parquet=out_parquet)

        self.assertEqual(len(df), self.total_folds * self.n_per_fold)
        self.assertTrue(os.path.exists(out_parquet))
        for col in FINDINGS:
            self.assertIn(col, df.columns)

    def test_evaluate_oof_performance(self):
        df = aggregate_5fold_oofs(self.tag, oof_dir=self.test_dir, num_folds=self.total_folds)
        aucs = evaluate_oof_performance(df, labels_path=self.labels_path)
        self.assertEqual(len(aucs), len(FINDINGS))
        for col, auc in aucs.items():
            self.assertTrue(0.0 <= auc <= 1.0)


if __name__ == "__main__":
    unittest.main()
