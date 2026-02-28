"""Tests for LocalFeatureStore."""
import json
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage import LocalFeatureStore
from utils import EntityFeature


class LocalFeatureStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = LocalFeatureStore(root=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_save_and_load_roundtrip(self) -> None:
        features = [
            EntityFeature(
                entity_type="table",
                name="table_0_0",
                page_index=0,
                text_snippet="| col |",
                metadata={"bbox": {"x": 1}},
                embedding=[0.1, 0.2],
            ),
            EntityFeature(
                entity_type="text_block",
                name="page_0_text",
                page_index=0,
                text_snippet="Revenue up",
                metadata={"length": 11},
            ),
        ]

        output_path = self.store.save_features("doc123", features)
        self.assertTrue(Path(output_path).exists())

        loaded = self.store.load_features("doc123")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].name, "table_0_0")
        self.assertEqual(loaded[0].embedding, [0.1, 0.2])

    def test_save_requires_document_id(self) -> None:
        with self.assertRaises(ValueError):
            self.store.save_features("", [])

    def test_load_missing_document_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.store.load_features("missing")


if __name__ == "__main__":
    unittest.main()
