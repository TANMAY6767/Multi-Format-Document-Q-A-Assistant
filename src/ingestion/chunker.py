
import tiktoken

from docling.chunking import HybridChunker

from docling_core.transforms.chunker.tokenizer.openai import (
    OpenAITokenizer
)

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider
)

from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer
)


# --------------------------------------------------
# 1. Configure Markdown serialization for tables
# --------------------------------------------------

class FinancialSerializerProvider(ChunkingSerializerProvider):

    def get_serializer(self, doc):

        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True)
        )


# --------------------------------------------------
# 2. Create the chunker
# --------------------------------------------------

def create_chunker():

    encoding = tiktoken.get_encoding("cl100k_base")

    tokenizer = OpenAITokenizer(
        tokenizer=encoding,
        max_tokens=700
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=FinancialSerializerProvider()
    )

    return chunker, tokenizer


# --------------------------------------------------
# 3. Extract metadata from each chunk
# --------------------------------------------------

def extract_metadata(chunk, source_name):

    page_numbers = sorted({
        provenance.page_no
        for item in chunk.meta.doc_items
        for provenance in item.prov
        if provenance.page_no is not None
    })

    headings = chunk.meta.headings or []

    labels = {
        getattr(item.label, "value", str(item.label))
        for item in chunk.meta.doc_items
    }

    return {
        "source": source_name,
        "page_numbers": page_numbers,
        "heading_path": " > ".join(headings),
        "contains_table": "table" in labels
    }


# --------------------------------------------------
# 4. Convert document into chunks
# --------------------------------------------------

def create_chunks(document, source_name):

    chunker, tokenizer = create_chunker()

    chunks = list(chunker.chunk(dl_doc=document))

    records = []

    for chunk in chunks:

        raw_text = chunk.text.replace("<!-- image -->", "").strip()

        content = chunker.contextualize(chunk)
        content = content.replace("<!-- image -->", "").strip()

        if not content:
            continue

        metadata = extract_metadata(chunk, source_name)

        records.append({
            "chunk_id": len(records),
            "content": content,
            "raw_text": raw_text,
            "token_count": tokenizer.count_tokens(content),
            **metadata
        })

    return records


