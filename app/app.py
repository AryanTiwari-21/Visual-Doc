"""
Visual-Doc Assistant — Phase 3, Step 9D: True Multi-Vector Late-Interaction (MaxSim)
======================================================================================
Every prior step (9, 9B, 9C) stored ONE mean-pooled 128-dim vector per page and
retrieved with ChromaDB's plain cosine search — a deliberate simplification from
the ColPali paper's actual scoring function, which keeps every patch/token vector
and computes:

    LI(q, d) = sum_i  max_j  <Eq(i) | Ed(j)>          (Eq. 1, ColPali paper)

i.e. for each query token, find its single best-matching document patch, then sum
those best-match scores. Mean-pooling collapses all of a page's patches into one
average vector BEFORE any comparison happens — a page can average out to "close
enough" to a query while no single patch on it actually answers the question, and
conversely a page with one perfectly-matching patch buried among unrelated ones
can lose to a page that's merely uniformly-mediocre. This cell implements the
paper's real scoring function.

WHY THIS NEEDS NEW STORAGE, NOT JUST A NEW QUERY
ChromaDB's ANN index is built for ONE fixed-size vector per record — that's what
makes its cosine search fast. MaxSim needs the FULL patch matrix (~1024 x 128 per
page) available to score against, which isn't something an ANN index can hold or
search directly. So this cell:

  1. Keeps storing the mean-pooled vector in ChromaDB, but now ONLY as a cheap
     candidate-shortlist mechanism (fast, approximate) — not as the final score.
  2. ALSO saves each page's full patch matrix to disk as a float16 .npy file
     (~256 KB/page, matching the paper's own reported storage footprint, Table 4)
     next to its PNG, path stored in ChromaDB metadata.
  3. At query time, re-ranks the shortlist by computing the EXACT MaxSim formula
     above against those saved patch matrices — this is the real ColPali score,
     not an approximation.

This is the standard two-stage pattern real ColPali deployments use (the paper
itself points to PLAID/Vespa for this at Wikipedia-scale); we're doing the same
shortlist-then-exact-rerank idea with ChromaDB + local files instead of a
dedicated multi-vector engine. An "Exhaustive" mode is also included that skips
the ANN shortlist entirely and scores every indexed page exactly, for anyone who
wants zero approximation on a smaller library.

Since indexing now also has to compute and save a much bigger per-page tensor
(not just one small vector), this cell folds in Step 9C's batched-embedding and
parallel-rendering optimizations too — without them, MaxSim indexing would be
slower than 9B's single-page loop ever was. If you already applied 9C
separately, this is a strict superset and simply overwrites app.py the same way
every step here does.

EVERYTHING ELSE IS UNCHANGED: the Streamlit UI shape, PDF upload flow, Document
Library, chat history, Gemini prompt construction (build_multimodal_input),
generation (generate_answer), and SYSTEM_INSTRUCTION are identical to Step 9B.
Existing documents from Steps 9/9B/9C are NOT invalidated — they still retrieve
correctly using the fallback described below, and a one-click "Upgrade to
MaxSim" button (Document Library sidebar) backfills their full patch embeddings
from the page images already saved on disk — no re-upload of the original PDF
needed.
"""
import os
import shutil
import base64
import hashlib
import time

import numpy as np
import streamlit as st
import torch
from PIL import Image
import chromadb
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from colpali_engine.models import ColPali, ColPaliProcessor
from google import genai

# ============================================================================
# Config
# ============================================================================
MODEL_NAME       = "vidore/colpali-v1.2"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# Where page images, patch matrices and the ChromaDB index live. Set VISUALDOC_DATA_DIR
# to run outside Colab; the default keeps the original Google Drive location.
DATA_DIR         = os.environ.get("VISUALDOC_DATA_DIR", "/content/drive/MyDrive/Visual-Doc")
IMAGE_FOLDER     = os.path.join(DATA_DIR, "processed_images")
DB_PATH          = os.path.join(DATA_DIR, "chroma_db_storage")

GEMINI_MODEL     = "gemini-3.6-flash"
DEFAULT_DPI      = 200
DEFAULT_BATCH    = 4
RENDER_THREADS   = max(1, os.cpu_count() or 4)
PATCH_DTYPE      = np.float16   # matches the paper's own storage footprint (Table 4)
DEFAULT_SHORTLIST = 20

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

SYSTEM_INSTRUCTION = (
    "You are the Visual-Doc Assistant, a technical documentation Q&A system. "
    "Answer the user's question using ONLY the page image(s) provided below — "
    "do not use outside knowledge. Reference the source document and page "
    "number when relevant. If the provided pages don't contain the answer, "
    "say so directly instead of guessing."
)

st.set_page_config(page_title="Visual-Doc Assistant", page_icon="📄", layout="wide")


# ============================================================================
# Cached resources
# ============================================================================
@st.cache_resource(show_spinner="Loading ColPali model (first load only)...")
def load_colpali():
    model = ColPali.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
    ).eval()
    processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
    return model, processor


@st.cache_resource(show_spinner="Connecting to ChromaDB...")
def load_collection():
    db_client = chromadb.PersistentClient(path=DB_PATH)
    return db_client.get_or_create_collection(
        name="visual_doc_collection", metadata={"hnsw:space": "cosine"}
    )


model, processor = load_colpali()
collection = load_collection()


@st.cache_resource(show_spinner="Checking index for legacy records...")
def migrate_legacy_metadata():
    """One-time backfill for pre-Step-9B records missing doc_id_prefix.
    Unrelated to MaxSim — unchanged from Step 9B."""
    if collection.count() == 0:
        return 0
    all_data = collection.get(include=["metadatas"])
    updates_ids, updates_meta = [], []
    for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        if "doc_id_prefix" not in meta:
            legacy_prefix = f"{meta.get('doc_name', 'legacy')}__legacy"
            new_meta = dict(meta)
            new_meta["doc_id_prefix"] = legacy_prefix
            updates_ids.append(doc_id)
            updates_meta.append(new_meta)
    if updates_ids:
        collection.update(ids=updates_ids, metadatas=updates_meta)
    return len(updates_ids)


migrated_count = migrate_legacy_metadata()


# ============================================================================
# Embedding functions — now return TRUE multi-vector representations
# ============================================================================
@torch.no_grad()
def embed_page_images_batch(images: list):
    """
    Returns a list of (full, proxy) per image:
      full  — (N_patches, 128) float16 — the TRUE multi-vector patch embedding,
              saved to disk and used for exact MaxSim scoring.
      proxy — (128,) float32 — mean-pooled, kept ONLY as a fast ANN
              candidate-shortlist key (same role Steps 9B/9C used it for
              directly as the final score — it no longer is one).
    """
    inputs = processor.process_images(images).to(DEVICE)
    batch_patch_embeddings = model(**inputs)              # (B, N_patches, 128)
    results = []
    for patch_emb in batch_patch_embeddings:
        full32 = patch_emb.cpu().float().numpy()
        full = full32.astype(PATCH_DTYPE)                  # (N_patches, 128)
        proxy = full32.mean(axis=0).astype(np.float32)      # (128,)
        results.append((full, proxy))
    return results


@torch.no_grad()
def embed_query(query_text: str):
    """
    Returns (proxy, full_tokens):
      proxy       — (128,) mean-pooled, for ANN shortlisting only.
      full_tokens — (N_tokens, 128), the TRUE per-token query embedding used
                    for exact MaxSim scoring (includes ColBERT-style query
                    augmentation tokens, applied automatically by
                    ColPaliProcessor — unchanged from every prior phase).
    """
    inputs = processor.process_queries([query_text]).to(DEVICE)
    token_embeddings = model(**inputs)                      # (1, N_tokens, 128)
    full_tokens = token_embeddings.squeeze(0).cpu().float().numpy()
    proxy = full_tokens.mean(axis=0)
    return proxy, full_tokens


def maxsim_score(query_tokens: np.ndarray, doc_patches: np.ndarray) -> float:
    """
    Exact ColPali/ColBERT late interaction:  LI(q,d) = sum_i max_j <Eq(i)|Ed(j)>
    query_tokens: (Nq, 128)   doc_patches: (Nd, 128)   No padding needed —
    each document is scored independently, so patch counts can differ freely
    between pages.
    """
    sims = query_tokens.astype(np.float32) @ doc_patches.astype(np.float32).T   # (Nq, Nd)
    return float(sims.max(axis=1).sum())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


@st.cache_data(show_spinner=False)
def _load_patch_embeddings(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


# ============================================================================
# PDF ingestion pipeline — Step 9C's speed fixes, extended to save full patches
# ============================================================================
def _file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()[:10]


def index_pdf(uploaded_file, dpi: int, batch_size: int, progress_bar=None) -> dict:
    file_bytes = uploaded_file.getvalue()
    doc_hash = _file_hash(file_bytes)
    doc_name = os.path.splitext(uploaded_file.name)[0]
    doc_id_prefix = f"{doc_name}__{doc_hash}"
    doc_image_folder = os.path.join(IMAGE_FOLDER, doc_id_prefix)

    t_start = time.time()

    try:
        total_pages = pdfinfo_from_bytes(file_bytes)["Pages"]
    except Exception:
        total_pages = None

    if total_pages is not None:
        already = collection.get(where={"doc_id_prefix": doc_id_prefix}, include=[])
        if len(already["ids"]) >= total_pages:
            if progress_bar is not None:
                progress_bar.progress(1.0, text=f"'{uploaded_file.name}' already fully indexed — skipped")
            return {
                "doc_name": doc_name, "doc_id_prefix": doc_id_prefix,
                "total_pages": total_pages, "added": 0, "skipped": total_pages,
                "elapsed": time.time() - t_start,
            }

    os.makedirs(doc_image_folder, exist_ok=True)

    pages = convert_from_bytes(file_bytes, dpi=dpi, thread_count=RENDER_THREADS)
    total_pages = len(pages)

    candidate_ids = [f"{doc_id_prefix}__page_{i + 1}" for i in range(total_pages)]
    existing_ids = set(collection.get(ids=candidate_ids, include=[])["ids"])

    to_process = [
        (i + 1, page) for i, page in enumerate(pages)
        if f"{doc_id_prefix}__page_{i + 1}" not in existing_ids
    ]
    skipped = total_pages - len(to_process)
    added = 0

    for batch_start in range(0, len(to_process), batch_size):
        batch = to_process[batch_start: batch_start + batch_size]
        batch_page_nums = [pn for pn, _ in batch]
        batch_images_rgb = [pg.convert("RGB") for _, pg in batch]

        batch_paths = []
        for page_num, rgb_img in zip(batch_page_nums, batch_images_rgb):
            image_path = os.path.join(doc_image_folder, f"page_{page_num}.png")
            if not os.path.exists(image_path):
                rgb_img.save(image_path, "PNG")
            batch_paths.append(image_path)

        embedded = embed_page_images_batch(batch_images_rgb)   # [(full, proxy), ...]

        ids_batch, embs_batch, metas_batch, docs_batch = [], [], [], []
        for page_num, image_path, (full, proxy) in zip(batch_page_nums, batch_paths, embedded):
            patch_path = os.path.join(doc_image_folder, f"page_{page_num}_patches.npy")
            np.save(patch_path, full)

            ids_batch.append(f"{doc_id_prefix}__page_{page_num}")
            embs_batch.append(proxy.tolist())
            metas_batch.append({
                "page_number": page_num,
                "image_path": image_path,
                "filename": uploaded_file.name,
                "doc_name": doc_name,
                "doc_id_prefix": doc_id_prefix,
                "patch_embedding_path": patch_path,
                "num_patches": int(full.shape[0]),
            })
            docs_batch.append(f"{doc_name} — page {page_num}")

        collection.add(ids=ids_batch, embeddings=embs_batch, metadatas=metas_batch, documents=docs_batch)
        added += len(batch)

        if progress_bar is not None:
            done = batch_start + len(batch)
            progress_bar.progress(
                min(done / max(len(to_process), 1), 1.0),
                text=f"Embedding '{uploaded_file.name}' — {done}/{len(to_process)} new pages "
                     f"({skipped} already indexed)",
            )

    if progress_bar is not None:
        progress_bar.progress(1.0, text=f"'{uploaded_file.name}' done — {added} added, {skipped} skipped")

    return {
        "doc_name": doc_name, "doc_id_prefix": doc_id_prefix,
        "total_pages": total_pages, "added": added, "skipped": skipped,
        "elapsed": time.time() - t_start,
    }


def get_indexed_documents() -> dict:
    if collection.count() == 0:
        return {}
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    docs = {}
    for m in all_meta:
        key = (m.get("doc_name", "unknown"), m.get("doc_id_prefix", "unknown"))
        docs[key] = docs.get(key, 0) + 1
    return dict(sorted(docs.items(), key=lambda kv: kv[0][0].lower()))


def get_maxsim_readiness() -> dict:
    """Return {(doc_name, doc_id_prefix): (pages_with_full_embeddings, total_pages)}."""
    if collection.count() == 0:
        return {}
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    stats = {}
    for m in all_meta:
        key = (m.get("doc_name", "unknown"), m.get("doc_id_prefix", "unknown"))
        ready, total = stats.get(key, (0, 0))
        total += 1
        patch_path = m.get("patch_embedding_path")
        if patch_path and os.path.exists(patch_path):
            ready += 1
        stats[key] = (ready, total)
    return stats


def upgrade_document_to_maxsim(doc_id_prefix: str, batch_size: int, progress_bar=None) -> dict:
    """
    Backfill full patch embeddings for pages indexed before Step 9D (or an
    interrupted earlier upgrade). Reuses the page PNGs already saved on disk
    by Steps 9B/9C — the original PDF is NOT needed.
    """
    matches = collection.get(where={"doc_id_prefix": doc_id_prefix}, include=["metadatas"])
    to_upgrade = [
        (mid, meta) for mid, meta in zip(matches["ids"], matches["metadatas"])
        if not (meta.get("patch_embedding_path") and os.path.exists(meta.get("patch_embedding_path", "")))
    ]
    total = len(to_upgrade)
    upgraded = 0

    for batch_start in range(0, total, batch_size):
        batch = to_upgrade[batch_start: batch_start + batch_size]
        batch_ids = [mid for mid, _ in batch]
        batch_metas = [meta for _, meta in batch]
        batch_images = [Image.open(meta["image_path"]).convert("RGB") for meta in batch_metas]

        embedded = embed_page_images_batch(batch_images)

        new_metas, new_embs = [], []
        for meta, (full, proxy) in zip(batch_metas, embedded):
            folder = os.path.dirname(meta["image_path"])
            patch_path = os.path.join(folder, f"page_{meta['page_number']}_patches.npy")
            np.save(patch_path, full)
            m2 = dict(meta)
            m2["patch_embedding_path"] = patch_path
            m2["num_patches"] = int(full.shape[0])
            new_metas.append(m2)
            new_embs.append(proxy.tolist())

        collection.update(ids=batch_ids, embeddings=new_embs, metadatas=new_metas)
        upgraded += len(batch)

        if progress_bar is not None:
            progress_bar.progress(
                min(upgraded / max(total, 1), 1.0),
                text=f"Upgrading to MaxSim — {upgraded}/{total} pages",
            )

    if progress_bar is not None:
        progress_bar.progress(1.0, text=f"Upgrade complete — {upgraded} pages")

    return {"upgraded": upgraded, "total": total}


def delete_document(doc_id_prefix: str) -> int:
    matches = collection.get(where={"doc_id_prefix": doc_id_prefix})
    ids_to_delete = matches["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    folder = os.path.join(IMAGE_FOLDER, doc_id_prefix)
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)
    return len(ids_to_delete)


def reset_index():
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)
    if os.path.isdir(IMAGE_FOLDER):
        shutil.rmtree(IMAGE_FOLDER, ignore_errors=True)
        os.makedirs(IMAGE_FOLDER, exist_ok=True)


# ============================================================================
# Retrieval — two-stage: ANN shortlist (or exhaustive) + exact MaxSim rerank
# ============================================================================
def retrieve_top_k_pages(query_text: str, k: int = 3, doc_filter=None,
                          mode: str = "fast", shortlist_size: int = DEFAULT_SHORTLIST):
    """
    mode="fast" (default): ChromaDB's ANN index over the mean-pooled proxy
        vector produces a shortlist of `shortlist_size` candidates FAST, then
        every candidate is re-scored with the EXACT MaxSim formula and
        re-ranked — the score that determines the final top-k is always the
        true late-interaction score, never the proxy.
    mode="exhaustive": skips the ANN shortlist and MaxSim-scores every
        indexed page (respecting doc_filter) — exact by construction, no
        approximation anywhere, but scales linearly with library size.

    Pages indexed before Step 9D (no saved patch matrix yet) fall back to
    proxy-vector cosine similarity so they still participate in ranking,
    clearly labeled `score_type: "approx"` rather than silently mixed in
    as if it were a real MaxSim score.
    """
    proxy, query_tokens = embed_query(query_text)
    where = {"doc_id_prefix": {"$in": doc_filter}} if doc_filter else None

    if mode == "exhaustive":
        candidates = collection.get(where=where, include=["metadatas"])
        cand_ids, cand_metas = candidates["ids"], candidates["metadatas"]
    else:
        n_results = min(shortlist_size, max(collection.count(), 1))
        results = collection.query(
            query_embeddings=[proxy.tolist()], n_results=n_results,
            where=where, include=["metadatas"],
        )
        cand_ids = results["ids"][0] if results["ids"] else []
        cand_metas = results["metadatas"][0] if results["metadatas"] else []

    scored, legacy_idx = [], []
    for cid, meta in zip(cand_ids, cand_metas):
        patch_path = meta.get("patch_embedding_path")
        if patch_path and os.path.exists(patch_path):
            doc_patches = _load_patch_embeddings(patch_path)
            score = maxsim_score(query_tokens, doc_patches)
            scored.append({"id": cid, "meta": meta, "score": score, "score_type": "MaxSim"})
        else:
            scored.append({"id": cid, "meta": meta, "score": None, "score_type": "approx"})
            legacy_idx.append(len(scored) - 1)

    if legacy_idx:
        legacy_ids = [scored[i]["id"] for i in legacy_idx]
        fetched = collection.get(ids=legacy_ids, include=["embeddings"])
        emb_by_id = dict(zip(fetched["ids"], fetched["embeddings"]))
        for i in legacy_idx:
            doc_proxy = np.array(emb_by_id[scored[i]["id"]])
            scored[i]["score"] = _cosine(proxy, doc_proxy)

    scored.sort(key=lambda s: s["score"], reverse=True)

    pages = []
    for s in scored[:k]:
        meta = s["meta"]
        pages.append({
            "page_number": meta["page_number"],
            "image_path": meta["image_path"],
            "doc_name": meta["doc_name"],
            "score": s["score"],
            "score_type": s["score_type"],
        })
    return pages


def build_multimodal_input(query_text: str, pages: list):
    content = [{"type": "text", "text": f"Question: {query_text}"}]
    for p in pages:
        with open(p["image_path"], "rb") as f:
            image_bytes = f.read()
        content.append({
            "type": "image",
            "data": base64.b64encode(image_bytes).decode("utf-8"),
            "mime_type": "image/png",
        })
    return content


def generate_answer(gemini_client, query_text: str, pages: list) -> str:
    try:
        interaction = gemini_client.interactions.create(
            model=GEMINI_MODEL,
            input=build_multimodal_input(query_text, pages),
            system_instruction=SYSTEM_INSTRUCTION,
        )
        return interaction.output_text
    except Exception as e:
        return f"Gemini request failed: {e}"


# ============================================================================
# Session state
# ============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()


# ============================================================================
# Sidebar — Document Library
# ============================================================================
st.sidebar.title("📁 Document Library")

if migrated_count:
    st.sidebar.caption(f"Migrated {migrated_count} legacy record(s) from Phase 2.")

if shutil.which("pdftoppm") is None:
    st.sidebar.error(
        "poppler-utils not found on this system.\n\n"
        "Install it before uploading PDFs:\n"
        "`apt-get install -y poppler-utils` (Linux/Colab)\n"
        "`brew install poppler` (macOS)"
    )

with st.sidebar.expander("⚡ Indexing speed settings", expanded=False):
    st.caption(
        "Lower DPI speeds up rasterization without hurting retrieval quality "
        "(ColPali resizes internally regardless) — it only affects how sharp "
        "the page looks when Gemini reads it."
    )
    render_dpi = st.slider("Rendering DPI", min_value=100, max_value=300, value=DEFAULT_DPI, step=25)
    embed_batch_size = st.slider(
        "Embedding batch size", min_value=1, max_value=8, value=DEFAULT_BATCH,
        help="Pages embedded per model forward pass. Lower this if you hit a CUDA out-of-memory error "
             "— storing full patch matrices now uses more VRAM per page than mean-pooling did.",
    )

with st.sidebar.expander("🎯 Retrieval settings", expanded=False):
    st.caption(
        "Fast mode uses the mean-pooled vector only to shortlist candidates via "
        "ChromaDB's ANN index, then re-ranks that shortlist with the exact "
        "ColBERT-style MaxSim score — the score that decides the final answer is "
        "always the real late-interaction score. Exhaustive mode skips the "
        "shortlist and scores every indexed page exactly."
    )
    retrieval_mode_label = st.radio(
        "Mode",
        ["Fast (ANN shortlist + exact rerank)", "Exhaustive (exact over full library)"],
        index=0,
    )
    retrieval_mode = "exhaustive" if retrieval_mode_label.startswith("Exhaustive") else "fast"
    shortlist_size = st.slider(
        "Shortlist size (Fast mode)", min_value=5, max_value=50, value=DEFAULT_SHORTLIST,
        disabled=(retrieval_mode == "exhaustive"),
    )
    if retrieval_mode == "exhaustive":
        st.caption(f"⚠️ Will exactly score all {collection.count()} indexed pages per query.")

uploaded_files = st.sidebar.file_uploader(
    "Add PDFs to the knowledge base",
    type=["pdf"],
    accept_multiple_files=True,
    help="Any number of files. Each page is rendered as an image and embedded directly with ColPali — no OCR.",
)

if uploaded_files:
    for uf in uploaded_files:
        upload_key = f"{uf.name}_{uf.size}"
        if upload_key in st.session_state.processed_uploads:
            continue

        progress_bar = st.sidebar.progress(0.0, text=f"Preparing '{uf.name}'...")
        try:
            result = index_pdf(uf, dpi=render_dpi, batch_size=embed_batch_size, progress_bar=progress_bar)
            progress_bar.progress(
                1.0,
                text=(f"✅ {uf.name}: {result['added']} new pages, {result['skipped']} already indexed "
                      f"({result['elapsed']:.1f}s)"),
            )
        except Exception as e:
            progress_bar.empty()
            st.sidebar.error(f"Failed to index '{uf.name}': {e}")
        finally:
            st.session_state.processed_uploads.add(upload_key)

st.sidebar.divider()
st.sidebar.subheader("Indexed Documents")

docs = get_indexed_documents()
readiness = get_maxsim_readiness()

if not docs:
    st.sidebar.caption("No documents indexed yet — upload one or more PDFs above.")
else:
    for (doc_name, doc_id_prefix), page_count in docs.items():
        ready, total = readiness.get((doc_name, doc_id_prefix), (0, page_count))
        status = "✅ MaxSim-ready" if ready == total else f"⏳ {ready}/{total} MaxSim-ready"

        row = st.sidebar.columns([4, 1])
        row[0].markdown(f"**{doc_name}**  \n{page_count} pages · {status}")
        if row[1].button("🗑️", key=f"del_{doc_id_prefix}", help=f"Remove '{doc_name}'"):
            removed = delete_document(doc_id_prefix)
            st.sidebar.success(f"Removed {removed} pages from '{doc_name}'.")
            st.rerun()

        if ready < total:
            if st.sidebar.button(f"⏫ Upgrade '{doc_name}' to MaxSim", key=f"upg_{doc_id_prefix}"):
                pb = st.sidebar.progress(0.0, text="Starting upgrade...")
                res = upgrade_document_to_maxsim(doc_id_prefix, batch_size=embed_batch_size, progress_bar=pb)
                st.sidebar.success(f"Upgraded {res['upgraded']} of {res['total']} pages.")
                st.rerun()

    if st.sidebar.button("Clear ALL documents", type="secondary"):
        reset_index()
        st.session_state.chat_history = []
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("⚙️ Settings")

api_key = st.sidebar.text_input(
    "Gemini API key", type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
    help="Get one at aistudio.google.com/apikey",
)
top_k = st.sidebar.slider("Pages to retrieve (k)", min_value=1, max_value=8, value=3)

doc_names = [d[0] for d in docs.keys()]
selected_docs = st.sidebar.multiselect(
    "Search only in (optional)",
    options=doc_names,
    default=[],
    help="Leave empty to search across every indexed document.",
)
doc_filter = (
    [dip for (dn, dip) in docs.keys() if dn in selected_docs]
    if selected_docs else None
)

if st.sidebar.button("Clear chat history"):
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(f"Total pages indexed: {collection.count()} · Device: {DEVICE}")


# ============================================================================
# Main area
# ============================================================================
st.title("📄 Visual-Doc Assistant")
st.caption(
    "Multimodal RAG over your own documents — ColPali page-image retrieval with "
    "true multi-vector late-interaction (MaxSim) + Gemini generation. No OCR."
)

if not docs:
    st.info("👈 Upload one or more PDFs from the sidebar to get started. Any number of documents can be added, at any time.")


def _score_caption(p):
    label = "MaxSim" if p["score_type"] == "MaxSim" else "≈cos"
    return f"{p['doc_name']} · p.{p['page_number']} · {label} {p['score']:.3f}"


for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        with st.expander(f"📑 Source pages ({len(turn['pages'])})", expanded=False):
            cols = st.columns(min(len(turn["pages"]), 4) or 1)
            for i, p in enumerate(turn["pages"]):
                with cols[i % len(cols)]:
                    img = Image.open(p["image_path"]).convert("RGB")
                    st.image(img, caption=_score_caption(p), use_container_width=True)

query = st.chat_input("Ask a question about your documents...")

if query:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not docs:
        st.warning("Please upload at least one PDF first.")
    else:
        with st.chat_message("user"):
            st.markdown(query)

        gemini_client = genai.Client(api_key=api_key)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant pages (MaxSim)..."):
                pages = retrieve_top_k_pages(
                    query, k=top_k, doc_filter=doc_filter,
                    mode=retrieval_mode, shortlist_size=shortlist_size,
                )

            if not pages:
                st.warning("No relevant pages found in the selected document(s).")
            else:
                with st.spinner("Generating answer with Gemini..."):
                    answer = generate_answer(gemini_client, query, pages)

                st.markdown(answer)
                with st.expander(f"📑 Source pages ({len(pages)})", expanded=True):
                    cols = st.columns(min(len(pages), 4) or 1)
                    for i, p in enumerate(pages):
                        with cols[i % len(cols)]:
                            img = Image.open(p["image_path"]).convert("RGB")
                            st.image(img, caption=_score_caption(p), use_container_width=True)

                st.session_state.chat_history.append({
                    "query": query, "answer": answer, "pages": pages,
                })
