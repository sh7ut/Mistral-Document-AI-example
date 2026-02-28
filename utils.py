"""Utilities for parsing OCR responses into structured artifacts."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from mistralai import Mistral as MistralClient


@dataclass
class ImageArtifact:
    page_index: int
    base64_data: str
    bbox: Optional[Dict[str, Any]] = None


@dataclass
class TableArtifact:
    page_index: int
    html: Optional[str] = None
    markdown: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None


@dataclass
class PageArtifact:
    index: int
    text: str
    tables: List[TableArtifact] = field(default_factory=list)
    images: List[ImageArtifact] = field(default_factory=list)
    header: Optional[str] = None
    footer: Optional[str] = None


@dataclass
class DocumentArtifacts:
    pages: List[PageArtifact]
    model: Optional[str]
    usage_info: Dict[str, Any]

    @property
    def combined_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    def iter_tables(self) -> List[TableArtifact]:
        return [table for page in self.pages for table in page.tables]


def parse_ocr_response(ocr_response: Dict[str, Any]) -> DocumentArtifacts:
    pages = ocr_response.get("pages")
    if not isinstance(pages, list):
        raise ValueError("OCR response missing 'pages' list")

    page_artifacts: List[PageArtifact] = []
    for page in pages:
        index = page.get("index")
        if index is None:
            raise ValueError("Page entry missing 'index'")
        text = page.get("markdown") or ""
        header = page.get("header")
        footer = page.get("footer")

        tables_data = page.get("tables") or []
        tables: List[TableArtifact] = []
        for tbl in tables_data:
            tables.append(
                TableArtifact(
                    page_index=index,
                    html=tbl.get("html"),
                    markdown=tbl.get("markdown"),
                    bbox=tbl.get("bbox"),
                )
            )

        images_data = page.get("images") or []
        images: List[ImageArtifact] = []
        for img in images_data:
            base64_data = img.get("base64")
            if base64_data:
                images.append(
                    ImageArtifact(
                        page_index=index,
                        base64_data=base64_data,
                        bbox=img.get("bbox"),
                    )
                )

        page_artifacts.append(
            PageArtifact(
                index=index,
                text=text,
                tables=tables,
                images=images,
                header=header,
                footer=footer,
            )
        )

    return DocumentArtifacts(
        pages=page_artifacts,
        model=ocr_response.get("model"),
        usage_info=ocr_response.get("usage_info") or {},
    )


@dataclass
class EntityFeature:
    """Normalized entity extracted from OCR content."""

    entity_type: str
    name: str
    page_index: int
    text_snippet: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class FeatureExtractor:
    """Builds simple entity features and optional embeddings."""

    def __init__(self, client: Optional[MistralClient] = None, embed_model: str = "mistral-embed-latest"):
        self.embed_model = embed_model
        self.client = client or self._init_client()

    @staticmethod
    def _init_client() -> MistralClient:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set. Please set it before running.")
        return MistralClient(api_key=api_key)

    def build_features(self, artifacts: DocumentArtifacts, embed: bool = False) -> List[EntityFeature]:
        features: List[EntityFeature] = []
        for page in artifacts.pages:
            if page.tables:
                for idx, table in enumerate(page.tables):
                    features.append(
                        EntityFeature(
                            entity_type="table",
                            name=f"table_{page.index}_{idx}",
                            page_index=page.index,
                            text_snippet=table.markdown or table.html or "",
                            metadata={"bbox": table.bbox},
                        )
                    )
            if page.text:
                features.append(
                    EntityFeature(
                        entity_type="text_block",
                        name=f"page_{page.index}_text",
                        page_index=page.index,
                        text_snippet=page.text[:5000],
                        metadata={"length": len(page.text)},
                    )
                )
        if embed and features:
            self._attach_embeddings(features)
        return features

    def _attach_embeddings(self, features: Iterable[EntityFeature]) -> None:
        prompts = [feature.text_snippet for feature in features]
        response = self.client.embeddings.create(model=self.embed_model, inputs=prompts)
        vectors = response.data
        if len(vectors) != len(prompts):
            raise ValueError("Embedding count mismatch")
        for feature, vector in zip(features, vectors):
            feature.embedding = vector.embedding


__all__ = [
    "DocumentArtifacts",
    "PageArtifact",
    "TableArtifact",
    "ImageArtifact",
    "EntityFeature",
    "FeatureExtractor",
    "parse_ocr_response",
]
