"""Tests for OCR parsing utilities."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (
    DocumentArtifacts,
    ImageArtifact,
    PageArtifact,
    TableArtifact,
    parse_ocr_response,
)


class ParseOcrResponseTests(unittest.TestCase):
    def test_parse_valid_response(self) -> None:
        response = {
            "model": "mistral-ocr-latest",
            "pages": [
                {
                    "index": 0,
                    "markdown": "Page 1 text",
                    "tables": [
                        {
                            "html": "<table><tr><td>1</td></tr></table>",
                            "markdown": "| c1 |\n|---|\n| 1 |",
                            "bbox": {"x": 10},
                        }
                    ],
                    "images": [
                        {
                            "base64": "aGVsbG8=",
                            "bbox": {"x": 5},
                        }
                    ],
                }
            ],
            "usage_info": {"tokens": 100},
        }

        artifacts = parse_ocr_response(response)

        self.assertIsInstance(artifacts, DocumentArtifacts)
        self.assertEqual(len(artifacts.pages), 1)
        page = artifacts.pages[0]
        self.assertIsInstance(page, PageArtifact)
        self.assertEqual(page.text, "Page 1 text")
        self.assertEqual(len(page.tables), 1)
        self.assertIsInstance(page.tables[0], TableArtifact)
        self.assertEqual(page.tables[0].html, "<table><tr><td>1</td></tr></table>")
        self.assertEqual(len(page.images), 1)
        self.assertIsInstance(page.images[0], ImageArtifact)
        self.assertEqual(artifacts.model, "mistral-ocr-latest")
        self.assertEqual(artifacts.usage_info["tokens"], 100)
        self.assertIn("Page 1 text", artifacts.combined_text)

    def test_parse_missing_pages_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_ocr_response({"pages": None})

    def test_missing_page_index_raises(self) -> None:
        response = {"pages": [{"markdown": "oops"}]}
        with self.assertRaises(ValueError):
            parse_ocr_response(response)


if __name__ == "__main__":
    unittest.main()
