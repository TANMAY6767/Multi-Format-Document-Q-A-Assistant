from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode
)

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)

from docling.datamodel.base_models import InputFormat
from pathlib import Path

def ingest_pdf(pdf_path: Path):

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Keep the demo focused on short reports so table extraction stays manageable.
    import pymupdf

    with pymupdf.open(pdf_path) as pdf:
        page_count = len(pdf)

    if page_count > 10:
        raise ValueError(
            f"This demo accepts PDFs with at most 10 pages; this one has {page_count}."
        )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options = pipeline_options
            )
        }
    )

    result = converter.convert(pdf_path)
    document = result.document

    return document

