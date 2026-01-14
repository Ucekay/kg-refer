"""Data loader module for loading user profiles, item profiles, attribute scores, paths, and similarity data."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


class DataLoader:
    """Load all necessary data for generating explainable recommendation prompts."""

    def __init__(
        self,
        user_profile_path: str | Path,
        item_profile_path: str | Path,
        attribute_scores_path: str | Path,
        path_scores_path: str | Path,
        similarity_path: str | Path,
    ):
        """Initialize the data loader with file paths.

        Args:
            user_profile_path: Path to user_profile.json
            item_profile_path: Path to item_profile.json
            attribute_scores_path: Path to user_item_attribute_scores_no_location.csv
            path_scores_path: Path to path_importance_scores_with_names.json
            similarity_path: Path to semantic_similarity_results.json
        """
        self.user_profile_path = Path(user_profile_path)
        self.item_profile_path = Path(item_profile_path)
        self.attribute_scores_path = Path(attribute_scores_path)
        self.path_scores_path = Path(path_scores_path)
        self.similarity_path = Path(similarity_path)

        # Cache for loaded data
        self._user_profiles: dict[int, str] | None = None
        self._item_profiles: dict[int, str] | None = None
        self._attribute_scores: pd.DataFrame | None = None
        self._path_scores: list[dict[str, Any]] | None = None
        self._similarity_data: list[dict[str, Any]] | None = None
        self._item_titles: dict[int, str] | None = None

    def load_user_profiles(self) -> dict[int, str]:
        """Load user profiles from JSON file.

        Returns:
            Dictionary mapping user ID to user summary text
        """
        if self._user_profiles is None:
            self._user_profiles = {}
            with open(self.user_profile_path, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    uid = data["uid"]
                    # Extract the summarization text from the nested JSON string
                    summary_data = json.loads(data["user summary"])
                    self._user_profiles[uid] = summary_data["summarization"]
        return self._user_profiles

    def load_item_profiles(self) -> dict[int, str]:
        """Load item profiles from JSON file.

        Returns:
            Dictionary mapping item ID to item summary text
        """
        if self._item_profiles is None:
            self._item_profiles = {}
            with open(self.item_profile_path, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    iid = data["iid"]
                    # Extract the summarization text from the nested JSON string
                    summary_data = json.loads(data["business summary"])
                    self._item_profiles[iid] = summary_data["summarization"]
        return self._item_profiles

    def load_attribute_scores(self) -> pd.DataFrame:
        """Load attribute scores from CSV file.

        Returns:
            DataFrame with columns: uid, iid, attribute_id, attribute_name, score
        """
        if self._attribute_scores is None:
            self._attribute_scores = pd.read_csv(self.attribute_scores_path)
        return self._attribute_scores

    def load_path_scores(self) -> list[dict[str, Any]]:
        """Load path importance scores from JSONL or JSON file.

        Returns:
            List of path dictionaries
        """
        if self._path_scores is None:
            self._path_scores = []
            with open(self.path_scores_path, encoding="utf-8") as f:
                # まずJSONL形式を試す（1行1JSONオブジェクト）
                try:
                    for line in f:
                        if line.strip():
                            self._path_scores.append(json.loads(line))
                except json.JSONDecodeError:
                    # JSONL形式でない場合は、JSON配列形式として読み込む（後方互換性）
                    f.seek(0)
                    self._path_scores = json.load(f)
        return self._path_scores

    def load_similarity_data(self) -> list[dict[str, Any]]:
        """Load semantic similarity data from JSONL or JSON file.

        Returns:
            List of similarity dictionaries
        """
        if self._similarity_data is None:
            self._similarity_data = []
            with open(self.similarity_path, encoding="utf-8") as f:
                # まずJSONL形式を試す（1行1JSONオブジェクト）
                try:
                    for line in f:
                        if line.strip():
                            self._similarity_data.append(json.loads(line))
                except json.JSONDecodeError:
                    # JSONL形式でない場合は、JSON配列形式として読み込む（後方互換性）
                    f.seek(0)
                    self._similarity_data = json.load(f)
        return self._similarity_data

    def get_user_summary(self, uid: int) -> str:
        """Get user summary for a specific user ID.

        Args:
            uid: User ID

        Returns:
            User summary text
        """
        profiles = self.load_user_profiles()
        return profiles.get(uid, f"User {uid} profile not found")

    def get_item_summary(self, iid: int) -> str:
        """Get item summary for a specific item ID.

        Args:
            iid: Item ID

        Returns:
            Item summary text
        """
        profiles = self.load_item_profiles()
        return profiles.get(iid, f"Item {iid} profile not found")

    def load_item_titles(
        self, item_list_path: str | Path | None = None
    ) -> dict[int, str]:
        """Load item titles from item_list.txt file.

        Args:
            item_list_path: Path to item_list.txt (tab-separated: id\\tname)

        Returns:
            Dictionary mapping item ID to item name
        """
        if self._item_titles is None and item_list_path is not None:
            self._item_titles = {}
            item_path = Path(item_list_path)
            if item_path.exists():
                with open(item_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and "\t" in line:
                            parts = line.split("\t")
                            iid = int(parts[0])
                            name = parts[1]
                            self._item_titles[iid] = name
        return self._item_titles if self._item_titles is not None else {}

    def get_item_title(self, iid: int, item_list_path: str | Path | None = None) -> str:
        """Get item title for a specific item ID.

        Args:
            iid: Item ID
            item_list_path: Path to item_list.txt (optional, will load if provided)

        Returns:
            Item title (business name) or placeholder if not found
        """
        titles = self.load_item_titles(item_list_path)
        if iid in titles:
            return titles[iid]
        else:
            # Print to stdout when not found
            print(f"Item title not found for iid={iid}, using placeholder")
            return f"Item {iid}"

    def get_attribute_preferences(
        self, uid: int, iid: int, min_score: float = 0.0
    ) -> list[dict[str, Any]]:
        """Get attribute preferences for a user-item pair.

        Args:
            uid: User ID
            iid: Item ID
            min_score: Minimum score threshold (default: 0.0)

        Returns:
            List of attribute preferences with score >= min_score
        """
        scores = self.load_attribute_scores()
        filtered = scores[
            (scores["uid"] == uid)
            & (scores["iid"] == iid)
            & (scores["score"] >= min_score)
        ]
        return filtered[["attribute_name", "score"]].to_dict("records")

    def get_top_paths(self, uid: int, iid: int, top_k: int = 2) -> list[dict[str, Any]]:
        """Get top-k important paths for a user-item pair.

        Args:
            uid: User ID
            iid: Item ID
            top_k: Number of top paths to retrieve (default: 2)

        Returns:
            List of top-k paths sorted by importance score
        """
        paths = self.load_path_scores()
        user_item_paths = [
            p for p in paths if p["user_id"] == uid and p["item_id"] == iid
        ]
        # Sort by importance score in descending order
        sorted_paths = sorted(
            user_item_paths, key=lambda x: x.get("importance_score", 0), reverse=True
        )
        return sorted_paths[:top_k]

    def get_similar_nodes(
        self, uid: int, iid: int, top_k: int = 2
    ) -> dict[str, list[tuple[int, float]]]:
        """Get similar users and items for a user-item pair.

        Args:
            uid: User ID
            iid: Item ID
            top_k: Number of similar nodes to retrieve for each type (default: 2)

        Returns:
            Dictionary with 'users' and 'items' keys, each containing list of (id, score) tuples
        """
        similarity_data = self.load_similarity_data()
        for entry in similarity_data:
            if entry["user_id"] == uid and entry["item_id"] == iid:
                similar_users = entry.get("similar_users", [])[:top_k]
                similar_items = entry.get("similar_items", [])[:top_k]
                return {
                    "users": [(int(u[0]), float(u[1])) for u in similar_users],
                    "items": [(int(i[0]), float(i[1])) for i in similar_items],
                }
        return {"users": [], "items": []}
