import tempfile
import unittest
from pathlib import Path

import numpy as np

from loopres.data import Dataset, collate, pack, write_dataset


class DataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "dataset"
        documents = [([10, 11, 12, 13, 14], 3), ([20, 21, 22], 1)] + [([30 + i] * (2 + i % 5), 1) for i in range(40)]
        write_dataset(self.path, documents, {})
        self.dataset = Dataset(self.path)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_a_row_labels_each_documents_answer_and_restarts_positions(self) -> None:
        batch = collate(self.dataset, [[0, 1]])
        self.assertEqual(batch["input_ids"][0].tolist(), [10, 11, 12, 13, 14, 20, 21, 22])
        self.assertEqual(batch["position_ids"][0].tolist(), [0, 1, 2, 3, 4, 0, 1, 2])
        self.assertEqual(batch["labels"][0].tolist(), [-100, -100, 13, 14, -100, 21, 22, -100])

    def test_packing_places_every_document_once_within_the_row_length(self) -> None:
        rows = pack(self.dataset.lengths, sample_length=9, seed=3)
        self.assertEqual(sorted(i for row in rows for i in row), list(range(len(self.dataset))))
        self.assertTrue(all(self.dataset.lengths[row].sum() <= 9 for row in rows))
        self.assertEqual(rows, pack(self.dataset.lengths, sample_length=9, seed=3))
        self.assertNotEqual(rows, pack(self.dataset.lengths, sample_length=9, seed=4))

    def test_store_round_trip(self) -> None:
        tokens, prompt_length = self.dataset.document(1)
        self.assertEqual((tokens.tolist(), prompt_length), ([20, 21, 22], 1))
        self.assertTrue(np.array_equal(self.dataset.lengths[:2], [5, 3]))


if __name__ == "__main__":
    unittest.main()
