"""Fixed lexical rules and full-input frozen embeddings for the NLP experiment."""
from collections import defaultdict
import json
import re

import numpy as np

try:
    from .download_nlp_encoder import verify_model
    from .prepare_news_review import digest
    from .reserve_news_evaluation import save_json
except ImportError:
    from download_nlp_encoder import verify_model
    from prepare_news_review import digest
    from reserve_news_evaluation import save_json


def keyword_predictions(text, config):
    text = " ".join(text.split())
    matches = {key: list(re.finditer(config[key], text, re.I))
               for key in ("geography", "transport", "security", "threat", "disruption")}
    result = dict(relevant=0, threat=0, disruption=0)
    width = config["window_characters"]
    # A qualifying window is anchored on geography, transport and security.
    for geo in matches["geography"]:
        near = {k: [m for m in matches[k] if abs(m.start() - geo.start()) <= width]
                for k in ("transport", "security", "threat", "disruption")}
        for transport in near["transport"]:
            for security in near["security"]:
                start = min(m.start() for m in (geo, transport, security))
                end = max(m.end() for m in (geo, transport, security))
                if end - start > width:
                    continue
                result["relevant"] = 1
                for target in ("threat", "disruption"):
                    if any(max(end, m.end()) - min(start, m.start()) <= width for m in near[target]):
                        result[target] = 1
    return result


def partition_tokens(token_ids, offsets, content_capacity):
    if not token_ids or len(token_ids) != len(offsets) or content_capacity < 1:
        raise ValueError("Invalid token sequence or capacity")
    chunks = []
    for start in range(0, len(token_ids), content_capacity):
        stop = min(start + content_capacity, len(token_ids))
        chunks.append({"token_start": start, "token_end": stop,
                       "character_start": offsets[start][0], "character_end": offsets[stop - 1][1],
                       "token_ids": token_ids[start:stop]})
    if [token for chunk in chunks for token in chunk["token_ids"]] != token_ids:
        raise ValueError("Chunk coverage mismatch")
    return chunks


def pool_chunks(vectors, counts):
    vectors = np.asarray(vectors, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) != len(counts) or np.any(counts <= 0):
        raise ValueError("Invalid chunk vectors/counts")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("Non-finite or zero chunk embedding")
    result = np.average(vectors / norms, axis=0, weights=counts)
    norm = np.linalg.norm(result)
    if norm == 0:
        raise ValueError("Zero article embedding")
    return (result / norm).astype(np.float32)


def encode_articles(inputs, config, root):
    import torch
    from sentence_transformers import SentenceTransformer
    directory = root / config["model_directory"]
    model_lock = verify_model(directory)
    if model_lock["model_id"] != config["model_id"]:
        raise ValueError("Wrong encoder model")
    torch.manual_seed(0)
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    model = SentenceTransformer(str(directory), device=config["device"],
                                local_files_only=True, trust_remote_code=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = model.tokenizer
    maximum = config["max_sequence_tokens"]
    if maximum > model.max_seq_length or not tokenizer.is_fast:
        raise ValueError("Unsupported tokenizer/sequence capacity")
    capacity = maximum - tokenizer.num_special_tokens_to_add(pair=False)
    all_chunks, coverage = [], []
    for index, row in enumerate(inputs):
        encoded = tokenizer(row["input_text"], add_special_tokens=False, truncation=False,
                            return_offsets_mapping=True, verbose=False)
        chunks = partition_tokens(encoded["input_ids"], encoded["offset_mapping"], capacity)
        coverage.append({"source": row["source"], "document_id": row["document_id"],
                         "input_sha256": row["input_sha256"], "tokens": len(encoded["input_ids"]),
                         "chunks": [{k: v for k, v in c.items() if k != "token_ids"} for c in chunks],
                         "all_tokens_covered_once": True})
        all_chunks.extend((index, chunk) for chunk in chunks)
    chunk_vectors, weights = defaultdict(list), defaultdict(list)
    for start in range(0, len(all_chunks), config["batch_size"]):
        batch = all_chunks[start:start + config["batch_size"]]
        prepared = [tokenizer.prepare_for_model(c["token_ids"], add_special_tokens=True,
                    truncation=False, return_attention_mask=True, verbose=False) for _, c in batch]
        if any(len(item["input_ids"]) > maximum for item in prepared):
            raise ValueError("Tokenized chunk exceeds encoder capacity")
        features = tokenizer.pad(prepared, padding=True, return_tensors="pt", verbose=False)
        features = {k: v.to(config["device"]) for k, v in features.items()}
        with torch.inference_mode():
            vectors = model(features)["sentence_embedding"].cpu().numpy()
        for (index, chunk), vector in zip(batch, vectors):
            chunk_vectors[index].append(vector)
            weights[index].append(len(chunk["token_ids"]))
    matrix = np.stack([pool_chunks(chunk_vectors[i], weights[i]) for i in range(len(inputs))])
    if matrix.shape != (len(inputs), config["dimension"]):
        raise ValueError("Unexpected encoder output shape")
    return matrix, coverage


def embedding_cache(inputs, config, root, recompute=False):
    """Only fixed, label-free features may be cached across chronological folds."""
    from importlib.metadata import version
    directory = root / "data/processed/nlp_embeddings_v1"
    model_directory = root / config["model_directory"]
    verify_model(model_directory)
    identity = {"inputs": [{k: r[k] for k in ("source", "document_id", "input_sha256")} for r in inputs],
                "config": config, "code_sha256": digest(root / "scripts/nlp_features.py"),
                "model_manifest_sha256": digest(model_directory / "encoder_manifest.json"),
                "versions": {k: version(k) for k in ("numpy", "torch", "transformers", "sentence-transformers")}}
    if directory.exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError("Embedding cache identity changed; explicit versioned rebuild required")
        for name, expected in manifest["artifact_sha256"].items():
            if digest(directory / name) != expected:
                raise ValueError("Embedding cache artifact changed")
        matrix = np.load(directory / "embeddings.npy", allow_pickle=False)
        coverage = json.loads((directory / "coverage.json").read_text())
        if recompute:
            actual, actual_coverage = encode_articles(inputs, config, root)
            if not np.array_equal(actual, matrix) or actual_coverage != coverage:
                raise ValueError("Frozen embeddings do not reproduce exactly")
    else:
        matrix, coverage = encode_articles(inputs, config, root)
        directory.mkdir(parents=True, exist_ok=False)
        np.save(directory / "embeddings.npy", matrix, allow_pickle=False)
        save_json(directory / "coverage.json", coverage)
        save_json(directory / "manifest.json", {"identity": identity, "artifact_sha256": {
            name: digest(directory / name) for name in ("embeddings.npy", "coverage.json")}})
    if (matrix.shape != (len(inputs), config["dimension"]) or not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-6)):
        raise ValueError("Invalid cached embeddings")
    return matrix, coverage, directory
