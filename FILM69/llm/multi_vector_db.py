import chromadb
import torch
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers.models.colpali import ColPaliForRetrieval, ColPaliProcessor
from datasets import load_dataset
import torch
from collections import Counter
import matplotlib.pyplot as plt
import time

def load_data(sample_size=500):
    """Load and sample the dataset."""
    ds = load_dataset("vidore/docvqa_test_subsampled", split="test")
    
    sample_ds = ds.select(range(sample_size))
    
    return sample_ds


def setup_model_and_collection():
    """Setup the model, processor, and ChromaDB collection."""
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(
        name="multivec",
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        })
    
    model_name = "vidore/colpali-v1.3-hf"
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    model = ColPaliForRetrieval.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    
    processor = ColPaliProcessor.from_pretrained(model_name, use_fast=True)
    
    return model, processor, collection, device


def embed_and_insert(sample_ds, model, processor, collection: chromadb.Collection):
    """Process images, generate embeddings, and insert into ChromaDB."""
    num_items = len(sample_ds)
    if collection.count() >= num_items*1024:
        return None
    
    for i, item in enumerate(sample_ds):
        image = item['image']
        doc_id = item['docId']
        
        res = collection.get(
            where={"doc_id": doc_id},
            include=["documents"],
        )
        
        
        if len(res["ids"]) > 0:
            continue
        
        batch_images = processor(images=[image]).to(model.device)
        
        with torch.no_grad():
            image_embeddings = model(**batch_images)
                        
        embedding_ids = []
        embedding_vectors = []
        metadatas = []
        
        for patch_idx, embedding in enumerate(image_embeddings.embeddings[0]):
            embedding_id = f"doc_{doc_id}_patch_{patch_idx}"
            embedding_vector = embedding.float().cpu().numpy().tolist()
            metadata = {
                'doc_id': int(doc_id),
                'patch_index': patch_idx
            }
            embedding_ids.append(embedding_id)
            embedding_vectors.append(embedding_vector)
            metadatas.append(metadata)
            
        collection.add(
            ids=embedding_ids,
            embeddings=embedding_vectors,
            metadatas=metadatas,
        )

def compute_maxsim_with_patch(query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, top_k=25):
    """
    Compute MaxSim score and most relevant patch indices.

    Args:
        query_embeddings: (num_query_tokens, dim)
        doc_embeddings: (num_patches, dim)
        top_k: number of top patches to return

    Returns:
        - maxsim_score: float
        - top_patch_indices: list of int (most frequently aligned patches)
    """
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)

    sims = torch.matmul(query_embeddings, doc_embeddings.T)

    max_vals, max_indices = sims.max(dim=1)

    maxsim_score = max_vals.mean().item()

    patch_counts = Counter(max_indices.tolist())
    top_patch_indices = [patch_idx for patch_idx, _ in patch_counts.most_common(top_k)]

    return maxsim_score, top_patch_indices


def query_and_rank(rand_query, model, processor, collection, device, num_results=4, fetch_factor=0.5):
    """Query the collection and rank documents using MaxSim."""
    
    processor_query = processor(text=[rand_query]).to(device)
    
    with torch.no_grad():
        query_embeddings = model(**processor_query)
            
    # this overfetches, then we will do maxsim on the unique set of doc ids to get the top num_results of documents
    results = collection.query(
        query_embeddings=query_embeddings.embeddings[0].tolist(),
        n_results=int(num_results*fetch_factor),
        include=["metadatas"],
    )
        
    candidate_docs = set()
    
    if results["metadatas"] is not None:
        for metadatas in results["metadatas"]:
            for metadata in metadatas:
                candidate_docs.add(metadata['doc_id'])
    
    print(f"Found {len(candidate_docs)} candidate documents")
    
    doc_scores = []
    
    for doc_id in candidate_docs:
        doc_data = collection.get(
            where={"doc_id": doc_id},
            include=["embeddings", "metadatas", "documents"],
        )
        
        doc_embeddings = torch.tensor(doc_data["embeddings"], dtype=query_embeddings.embeddings.dtype, device=device)
        
        score, top_patch_indices = compute_maxsim_with_patch(query_embeddings.embeddings[0], doc_embeddings)
        
        doc_scores.append({
            "doc_id": doc_id,
            "score": score,
            "top_patch_indices": top_patch_indices
        })
    
    top_docs = sorted(doc_scores, key=lambda x: x["score"], reverse=True)[:num_results]
    
    return top_docs, rand_query

def draw_patch_overlay(image: Image.Image, patch_indices: list, grid_size=None, total_patches=None):
    """
    Draws semi-transparent overlays on the patches corresponding to patch_indices.
    
    Args:
        image (PIL.Image): Original image.
        patch_indices (list): List of patch indices (0-indexed).
        grid_size (int, optional): Grid size used for patching. If None, will be calculated from total_patches.
        total_patches (int, optional): Total number of patches. Used to calculate grid_size if not provided.
    
    Returns:
        matplotlib.figure.Figure: Figure with overlay.
    """
    width, height = image.size
    
    # colpali has 6 special tokens at the beginning, so image patches start at index 6
    image_patch_offset = 6
    actual_image_patches = total_patches - image_patch_offset if total_patches else 1024
    
    # Calculate grid size based on actual number of image patches
    if grid_size is None:
        if total_patches is not None:
            # Calculate grid size from image patches only (excluding first 6 special tokens)
            import math
            grid_size = int(math.sqrt(actual_image_patches))
            print(f"Calculated grid_size {grid_size} from {actual_image_patches} image patches (total patches: {total_patches}, offset: {image_patch_offset})")
        else:
            # Fallback to estimated grid size
            target_patch_size = 14 
            grid_size = min(width // target_patch_size, height // target_patch_size, 32)
            grid_size = max(grid_size, 8)
            print(f"Using estimated grid_size {grid_size}")
    
    patch_w = width / grid_size
    patch_h = height / grid_size

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    
    for i, patch_index in enumerate(patch_indices):
        if patch_index < image_patch_offset:
            print(f"Skipping special token at index {patch_index}")
            continue
            
        image_patch_index = patch_index - image_patch_offset
        
        row = image_patch_index // grid_size
        col = image_patch_index % grid_size

        x0 = col * patch_w
        y0 = row * patch_h
        
        # Use different colors/alpha for different ranks
        alpha = max(0.1, 0.7 - (i * 0.02))  # Fade out lower ranked patches
        color = 'red' if i == 0 else 'orange' if i < 5 else 'yellow'
        
        rect = patches.Rectangle((x0, y0), patch_w, patch_h,
                               linewidth=2, edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(rect)
    
    ax.set_title(f"Top {len(patch_indices)} Patches Highlighted (Grid: {grid_size}x{grid_size}, Image patches: {actual_image_patches})")
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("patch.png", dpi=150, bbox_inches='tight')

    plt.close()


def main():
    """Main execution function."""
    sample_ds = load_data(sample_size=500)
    
    model, processor, collection, device = setup_model_and_collection()
    
    start_time = time.time()
    embed_and_insert(sample_ds, model, processor, collection)
    end_time = time.time()
    print(f"Embedding and inserting data took {end_time - start_time} seconds")
        
    top_docs, query_text = query_and_rank(
        rand_query=random.choice(sample_ds)['query'], 
        model=model, 
        processor=processor, 
        collection=collection, 
        device=device
        )
    
    print(f"Querying for: {query_text}")
    
    for doc in top_docs:
        print(f"Doc {doc['doc_id']} | Score: {doc['score']:.4f} | Top Patches: {doc['top_patch_indices']}")
        
    top_doc = top_docs[0]
    doc_id = top_doc["doc_id"]
    top_patches = top_doc["top_patch_indices"]
    
    image = next(item['image'] for item in sample_ds if item['docId'] == doc_id)
    
    doc_data = collection.get(
        where={"doc_id": doc_id},
    )
    actual_total_patches = len(doc_data["ids"])
    print(f"Actual total patches: {actual_total_patches}")
    draw_patch_overlay(image, top_patches, total_patches=actual_total_patches)


if __name__ == "__main__": 
    main()