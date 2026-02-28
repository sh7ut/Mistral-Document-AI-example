"""Demo-friendly persistence adapters for pipeline outputs."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from utils import EntityFeature


class LocalFeatureStore:
    """Persists features to JSON files under a local directory."""

    def __init__(self, root: Path | str = "demo_feature_store") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _serialize(self, features: Iterable[EntityFeature]) -> List[dict]:
        return [asdict(feature) for feature in features]

    def save_features(self, document_id: str, features: Iterable[EntityFeature]) -> Path:
        if not document_id:
            raise ValueError("document_id is required for persistence")
        feature_list = list(features)
        output = self.root / f"{document_id}.json"
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self._serialize(feature_list), handle, indent=2)
        return output

    def load_features(self, document_id: str) -> List[EntityFeature]:
        path = self.root / f"{document_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No features stored for document_id={document_id}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [EntityFeature(**item) for item in payload]


__all__ = ["LocalFeatureStore"]
