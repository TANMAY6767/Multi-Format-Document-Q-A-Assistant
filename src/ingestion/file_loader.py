from io import BytesIO
from pathlib import Path
import tempfile

import pandas as pd
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.chunker import create_chunks
from src.ingestion.pdf_loader import ingest_pdf


CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
MAX_ROWS_PER_SHEET = 10_000
MAX_EXCEL_SHEETS = 5


def _split_text(text: str, filename: str, location: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return [
        Document(
            page_content=part,
            metadata={"source": filename, "location": location},
        )
        for part in splitter.split_text(text)
        if part.strip()
    ]


def _load_pdf(filename: str, file_bytes: bytes) -> list[Document]:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as file:
            file.write(file_bytes)
            temporary_path = file.name

        document = ingest_pdf(Path(temporary_path))
        chunks = create_chunks(document, filename)
        return [
            Document(
                page_content=chunk["content"],
                metadata={
                    "source": filename,
                    "location": "pages " + ", ".join(map(str, chunk["page_numbers"])),
                },
            )
            for chunk in chunks
            if chunk["content"].strip()
        ]
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


def _split_table(
    frame: pd.DataFrame, filename: str, location: str
) -> list[Document]:
    if frame.empty:
        return []

    lines = frame.to_csv(index=False).splitlines()
    header = lines[0]
    rows = []
    documents = []
    encoding = tiktoken.get_encoding("cl100k_base")

    # Repeat the column names in each chunk so retrieved table rows keep context.
    for row in lines[1:]:
        candidate = header + "\n" + "\n".join(rows + [row])
        if rows and len(encoding.encode(candidate)) > CHUNK_SIZE:
            documents.extend(
                _split_text(header + "\n" + "\n".join(rows), filename, location)
            )
            rows = [row]
        else:
            rows.append(row)

    if rows:
        documents.extend(
            _split_text(header + "\n" + "\n".join(rows), filename, location)
        )
    elif not documents:
        documents.extend(_split_text(header, filename, location))

    return documents


def _load_csv(filename: str, file_bytes: bytes) -> list[Document]:
    frame = pd.read_csv(BytesIO(file_bytes), nrows=MAX_ROWS_PER_SHEET + 1)
    if len(frame) > MAX_ROWS_PER_SHEET:
        raise ValueError(f"{filename} has more than {MAX_ROWS_PER_SHEET:,} rows.")
    return _split_table(frame, filename, "CSV table")


def _load_excel(filename: str, file_bytes: bytes) -> list[Document]:
    workbook = pd.ExcelFile(BytesIO(file_bytes))
    if len(workbook.sheet_names) > MAX_EXCEL_SHEETS:
        raise ValueError(f"{filename} has more than {MAX_EXCEL_SHEETS} sheets.")

    documents = []
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(
            workbook, sheet_name=sheet_name, nrows=MAX_ROWS_PER_SHEET + 1
        )
        if len(frame) > MAX_ROWS_PER_SHEET:
            raise ValueError(
                f"Sheet '{sheet_name}' in {filename} has more than "
                f"{MAX_ROWS_PER_SHEET:,} rows."
            )
        documents.extend(
            _split_table(frame, filename, f"Excel sheet: {sheet_name}")
        )
    return documents


def load_file(filename: str, file_bytes: bytes) -> list[Document]:
    """Read one supported file and return small LangChain chunks."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return _load_pdf(filename, file_bytes)
    if extension == ".txt":
        return _split_text(file_bytes.decode("utf-8-sig"), filename, "text file")
    if extension == ".csv":
        return _load_csv(filename, file_bytes)
    if extension in {".xlsx", ".xls"}:
        return _load_excel(filename, file_bytes)

    raise ValueError(f"Unsupported file type: {extension or 'unknown'}")


def count_embedding_tokens(documents: list[Document]) -> int:
    """Count the text that will be sent to OpenAI's embedding model."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(doc.page_content)) for doc in documents)
