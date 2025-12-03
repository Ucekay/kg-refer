import argparse
import json
import os
from typing import Dict, List, Set, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

def load_interaction_graph(train_file_path: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Load interactions from train.txt
    Format: OrgUserID RemapUserID ItemID1 ItemID2 ...
    Returns:
        user_to_items: Dict[OrgUserID, Set[ItemID]]
        item_to_users: Dict[ItemID, Set[OrgUserID]]
    """
    user_to_items = {}
    item_to_users = {}
    
    print(f"Loading interactions from {train_file_path}...")
    with open(train_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            org_user_id = parts[0]
            # parts[1] is RemapUserID, ignore for now
            item_ids = parts[2:]
            
            user_to_items[org_user_id] = set(item_ids)
            
            for item_id in item_ids:
                if item_id not in item_to_users:
                    item_to_users[item_id] = set()
                item_to_users[item_id].add(org_user_id)
                
    print(f"Loaded {len(user_to_items)} users and {len(item_to_users)} items with interactions.")
    return user_to_items, item_to_users

def load_profiles(profile_file_path: str) -> Dict[str, str]:
    """
    Load profiles from profile.json.
    Assumes JSON Lines format based on G-Refer/merge_profiles.py logic.
    """
    id_to_text = {}
    
    if not os.path.exists(profile_file_path):
        print(f"Warning: Profile file {profile_file_path} not found. Using dummy profiles.")
        return id_to_text

    print(f"Loading profiles from {profile_file_path}...")
    try:
        with open(profile_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Try to extract ID and Text
                    # Possible keys: uid, iid, user_id, item_id, review, text, summary, etc.
                    
                    obj_id = None
                    text = ""
                    
                    if "uid" in data:
                        obj_id = str(data["uid"])
                    elif "iid" in data:
                        obj_id = str(data["iid"])
                    elif "user_id" in data:
                        obj_id = str(data["user_id"])
                    elif "item_id" in data:
                        obj_id = str(data["item_id"])
                        
                    if "text" in data:
                        text = data["text"]
                    elif "review" in data:
                        text = data["review"]
                    elif "summary" in data:
                        text = data["summary"]
                    elif "business summary" in data: # G-Refer specific
                         text = data["business summary"]
                    elif "user summary" in data: # G-Refer specific
                         text = data["user summary"]
                    
                    if obj_id and text:
                        id_to_text[obj_id] = str(text)
                        
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading profile file: {e}")
        
    print(f"Loaded {len(id_to_text)} profiles.")
    return id_to_text

def get_embedding(model, text: str) -> np.ndarray:
    return model.encode(text)

def compute_similarity(embedding1, embedding2):
    # Cosine similarity
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(embedding1, embedding2) / (norm1 * norm2)

def main():
    parser = argparse.ArgumentParser(description="Similarity Search for KG-Refer")
    parser.add_argument("--target_user", type=str, default="10088", help="Target User ID (Org)")
    parser.add_argument("--target_item", type=str, default="5124", help="Target Item ID")
    parser.add_argument("--train_file", type=str, 
                        default="../../packages/kgat/datasets/yelp/train.txt", 
                        help="Path to train.txt")
    parser.add_argument("--user_profile", type=str, 
                        default="../../../G-Refer/data/yelp/user_profile.json", 
                        help="Path to user_profile.json")
    parser.add_argument("--item_profile", type=str, 
                        default="../../../G-Refer/data/yelp/item_profile.json", 
                        help="Path to item_profile.json")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model")
    parser.add_argument("--top_k", type=int, default=5, help="Top K results")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save results")

    args = parser.parse_args()
    
    # Adjust paths relative to execution if needed
    train_path = Path(args.train_file).resolve()
    user_profile_path = Path(args.user_profile).resolve()
    item_profile_path = Path(args.item_profile).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    user_to_items, item_to_users = load_interaction_graph(str(train_path))
    user_profiles = load_profiles(str(user_profile_path))
    item_profiles = load_profiles(str(item_profile_path))
    
    # Combine for easy lookup if IDs are unique enough or handle separately
    # IDs in profile.json seem to be raw IDs (e.g. strings), train.txt has mapped IDs and raw IDs.
    # train.txt: OrgUserID RemapUserID ItemID1 ItemID2 ...
    # profile.json: "uid": "...", ...
    
    # 2. Prepare Candidates
    target_user = args.target_user
    target_item = args.target_item
    
    if target_user not in user_to_items:
        print(f"Target User {target_user} not found in interactions.")
    
    if target_item not in item_to_users:
        print(f"Target Item {target_item} not found in interactions.")

    # Candidates for User Similarity: Users who bought the Target Item
    candidate_users = item_to_users.get(target_item, set())
    candidate_users.discard(target_user) # Remove self
    
    # Candidates for Item Similarity: Items bought by the Target User
    candidate_items = user_to_items.get(target_user, set())
    candidate_items.discard(target_item) # Remove self
    
    print(f"Found {len(candidate_users)} candidate users (who bought item {target_item})")
    print(f"Found {len(candidate_items)} candidate items (bought by user {target_user})")
    
    # 3. Load Model
    print(f"Loading model {args.model_name}...")
    model = SentenceTransformer(args.model_name)
    
    # 4. Compute Embeddings & Similarity
    results_user = []
    results_item = []
    
    # Target Embeddings
    target_user_text = user_profiles.get(target_user, f"User {target_user}")
    target_item_text = item_profiles.get(target_item, f"Item {target_item}")
    
    target_user_emb = model.encode(target_user_text)
    target_item_emb = model.encode(target_item_text)
    
    print("Computing user similarities...")
    for uid in tqdm(candidate_users):
        text = user_profiles.get(uid, f"User {uid}")
        emb = model.encode(text)
        sim = compute_similarity(target_user_emb, emb)
        results_user.append((uid, sim, text[:50] + "..."))
        
    print("Computing item similarities...")
    for iid in tqdm(candidate_items):
        text = item_profiles.get(iid, f"Item {iid}")
        emb = model.encode(text)
        sim = compute_similarity(target_item_emb, emb)
        results_item.append((iid, sim, text[:50] + "..."))
        
    # 5. Top K
    results_user.sort(key=lambda x: x[1], reverse=True)
    results_item.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {args.top_k} Similar Users to User {target_user} (based on interactions with Item {target_item}):")
    for uid, sim, txt in results_user[:args.top_k]:
        print(f"User: {uid}, Score: {sim:.4f}, Text: {txt}")
        
    print(f"\nTop {args.top_k} Similar Items to Item {target_item} (based on interactions by User {target_user}):")
    for iid, sim, txt in results_item[:args.top_k]:
        print(f"Item: {iid}, Score: {sim:.4f}, Text: {txt}")
        
    # 6. Save Results
    output_file = output_dir / f"similarity_results_u{target_user}_i{target_item}.json"
    
    results = {
        "target_user": target_user,
        "target_item": target_item,
        "similar_users": [
            {"id": uid, "score": float(score), "text_snippet": text} 
            for uid, score, text in results_user[:args.top_k]
        ],
        "similar_items": [
            {"id": iid, "score": float(score), "text_snippet": text}
            for iid, score, text in results_item[:args.top_k]
        ]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()

