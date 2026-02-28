"""Tests for feature extraction layer."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import DocumentArtifacts, FeatureExtractor, PageArtifact, TableArtifact


class FeatureExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        pages = [
            PageArtifact(
                index=0,
                text="Revenue increased by 20%",
                tables=[
                    TableArtifact(page_index=0, markdown="| col |\n| 1 |", bbox={"x": 10})
                ],
            ),
            PageArtifact(index=1, text="Risks include liquidity issues"),
        ]
        self.artifacts = DocumentArtifacts(pages=pages, model="mistral-ocr-latest", usage_info={})

    def test_build_features_without_embeddings(self) -> None:
        extractor = FeatureExtractor(client=SimpleNamespace(embeddings=None))
        features = extractor.build_features(self.artifacts, embed=False)
        self.assertGreaterEqual(len(features), 3)
        types = {feature.entity_type for feature in features}
        self.assertIn("table", types)
        self.assertIn("text_block", types)

    def test_attach_embeddings_populates_vectors(self) -> None:
        mock_embeddings_client = SimpleNamespace()
        mock_embeddings_client.embeddings = SimpleNamespace(
            create=lambda model, inputs: SimpleNamespace(
                data=[SimpleNamespace(embedding=[float(i)]) for i, _ in enumerate(inputs)]
            )
        )
        extractor = FeatureExtractor(client=mock_embeddings_client)
        features = extractor.build_features(self.artifacts, embed=True)
        self.assertTrue(all(feature.embedding is not None for feature in features))
        self.assertEqual(features[0].embedding, [0.0])


if __name__ == "__main__":
    unittest.main()
