import asyncio
from functools import lru_cache

from rocrate_mcp.config import Settings

settings = Settings()

if settings.embeddings_provider == "local":
    from sentence_transformers import SentenceTransformer

    @lru_cache(maxsize=1)
    def get_model() -> SentenceTransformer:
        """Get or load the SentenceTransformer model and cache it."""
        return SentenceTransformer(settings.embeddings_model_name)


def chunk_text_by_tokens(
    text: str, model: SentenceTransformer, max_tokens: int | None = None, overlap: int = 20
) -> list[str]:
    """Chunk text into pieces based on token count for the given model."""
    tokenizer = model.tokenizer
    if max_tokens is None or max_tokens > model.max_seq_length:
        max_tokens = model.max_seq_length

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    step = max_tokens - overlap
    for start in range(0, len(token_ids), step):
        end = start + max_tokens
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks


async def get_embeddings(
    input: str,
    max_tokens: int | None = settings.embeddings_chunk_token_size,
    overlap: int = settings.embeddings_chunk_overlap,
    prompt_name: str | None = None,
) -> list[list[float]]:
    """Generate embeddings for the given input text using the configured provider."""
    if settings.embeddings_provider == "openai":
        NotImplementedError("OpenAI embeddings not implemented yet")
    elif settings.embeddings_provider == "local":
        return await sentencetransformer_embeddings(
            input, max_tokens=max_tokens, overlap=overlap, prompt_name=prompt_name
        )
    else:
        return [[]]


async def sentencetransformer_embeddings(
    input: str, max_tokens: int | None = None, overlap: int = 20, prompt_name: str | None = None
) -> list[list[float]]:
    """Generate embeddings using SentenceTransformer."""
    model = get_model()
    chunks = chunk_text_by_tokens(input, model, max_tokens=max_tokens, overlap=overlap)
    embeddings_list: list[list[float]] = []
    for chunk in chunks:
        emb = await asyncio.to_thread(
            model.encode, chunk, prompt_name=prompt_name, normalize_embeddings=True
        )
        embeddings_list.append(emb.tolist())
    return embeddings_list
