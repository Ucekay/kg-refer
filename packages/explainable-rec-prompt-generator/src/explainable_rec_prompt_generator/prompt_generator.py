"""Prompt generator module for creating explainable recommendation prompts."""

from typing import Any

from explainable_rec_prompt_generator.data_loader import DataLoader

EXPLAINABLE_REC_TEMPLATE_NO_EXAMPLES = """Given the business title, business profile, and user profile, explain why the user would enjoy this business.

CRITICAL REQUIREMENTS:
- Write EXACTLY ONE SENTENCE (no periods in the middle, only at the end)
- Start with "The user would enjoy [business name] because" or "The user would enjoy this business because"
- Maximum 50 words total
- DO NOT make meta-comments about preferences or alignment
- Focus ONLY on describing the business's actual features: food, atmosphere, service, location, menu items
- NEVER say: "preferences", "aligns with",  "their profile", "they value", "that aligns with", "aligning with"

Business title: {item_title}. Business profile: {item_summary}. User profile: {user_summary}.
### For the given user-item pair, here are several related paths connecting users and items through their interactions: {explanation_paths}
### For the user-item pair, here are some related users and items: Users: {similar_users} Items: {similar_items}
## For the user-item pair, here are the user's potential latent preferences toward the recommended item: {attribute_preferences}

### Explanation:"""


EXPLAINABLE_REC_TEMPLATE = """Given the business title, business profile, and user profile, explain why the user would enjoy this business.

CRITICAL REQUIREMENTS:
- Write EXACTLY ONE SENTENCE (no periods in the middle, only at the end)
- Start with "The user would enjoy [business name] because" or "The user would enjoy this business because"
- Maximum 50 words total
- DO NOT make meta-comments about preferences or alignment
- Focus ONLY on describing the business's actual features: food, atmosphere, service, location, menu items
- NEVER say: "preferences", "aligns with",  "their profile", "they value", "that aligns with their preferences", "aligning with their preferences", "that aligns with user's preferences", "aligning with user's preferences"

Business title: {item_title}. Business profile: {item_summary}. User profile: {user_summary}.
### For the given user-item pair, here are several related paths connecting users and items through their interactions: {explanation_paths}
### For the user-item pair, here are some related users and items: Users: {similar_users} Items: {similar_items}
### For the user-item pair, here are the user's potential latent preferences toward the recommended item: {attribute_preferences}
### Examples (follow this structure - one sentence each):
Example 1: The user would enjoy Taco Riendo because it provides a convenient location, is open late, offers highly rated food with a fresh and flavorful al pastor burrito, and creates a welcoming atmosphere with Spanish mood music, making it a delightful dining experience.
Example 2: The user would enjoy this business for its unique Taiwanese shaved ice flavors, cozy atmosphere with board games, and friendly shop owners offering discounts, making it a perfect spot to hang out and cool down with friends.
Example 3: The user would enjoy J Devoti Trattoria for its excellent food quality, especially the charcuterie board, which exceeded their expectations, making it a standout dining experience worth returning for.
### Explanation:"""


EXPLAINABLE_REC_TEMPLATE_SIMPLE = """Given the business title, business profile, and user profile, explain why the user would enjoy this business. Business title: {item_title}. Business profile: {item_summary}. User profile: {user_summary}.
### For the given user-item pair, here are several related paths connecting users and items through their interactions: {explanation_paths}
### For the user-item pair, here are some related users and items: Users: {similar_users} Items: {similar_items}
### For the user-item pair, here are the user's potential latent preferences toward the recommended item: {attribute_preferences}
### Explanation:"""


class PromptGenerator:
    """Generate explainable recommendation prompts for user-item pairs."""

    def __init__(self, data_loader: DataLoader):
        """Initialize the prompt generator with a data loader.

        Args:
            data_loader: DataLoader instance for accessing data
        """
        self.data_loader = data_loader

    def format_attribute_preferences(self, preferences: list[dict[str, Any]]) -> str:
        """Format attribute preferences for the prompt.

        Args:
            preferences: List of attribute preference dictionaries

        Returns:
            Formatted string of attribute preferences (names only, no scores)
        """
        if not preferences:
            return "No positive attribute preferences found."

        attr_names = []
        for pref in preferences:
            attr_name = pref["attribute_name"]
            attr_names.append(attr_name)

        return ", ".join(attr_names)

    def format_similar_users(self, similar_users: list[tuple[int, float]]) -> str:
        """Format similar users for the prompt.

        Args:
            similar_users: List of (user_id, similarity_score) tuples

        Returns:
            Formatted string of similar users with their profiles
        """
        if not similar_users:
            return "No similar users found."

        profiles = []
        for uid, _score in similar_users:
            user_summary = self.data_loader.get_user_summary(uid)
            profiles.append(user_summary)

        return ", ".join(profiles)

    def format_similar_items(self, similar_items: list[tuple[int, float]]) -> str:
        """Format similar items for the prompt.

        Args:
            similar_items: List of (item_id, similarity_score) tuples

        Returns:
            Formatted string of similar items with their profiles
        """
        if not similar_items:
            return "No similar items found."

        profiles = []
        for iid, _score in similar_items:
            item_summary = self.data_loader.get_item_summary(iid)
            profiles.append(item_summary)

        return ", ".join(profiles)

    def format_path(self, path: dict[str, Any]) -> str:
        """Format a single path for the prompt.

        Args:
            path: Path dictionary with node and relation information

        Returns:
            Formatted string of the path
        """
        path_node_infos = path.get("path_node_infos", [])
        path_relation_names = path.get("path_relation_names", [])

        if not path_node_infos:
            return "Empty path"

        # Build path string
        path_parts = []
        for i, node_info in enumerate(path_node_infos):
            node_type = node_info.get("node_type", "unknown")
            original_id = node_info.get("original_id")

            # Get profile for user/item nodes
            if node_type == "user":
                profile = self.data_loader.get_user_summary(original_id)
                path_parts.append(f"User (Profile: {profile})")
            elif node_type == "item":
                profile = self.data_loader.get_item_summary(original_id)
                path_parts.append(f"Item (Profile: {profile})")
            else:
                # Entity node - show entity name
                node_name = node_info.get("node_name", f"Entity_{original_id}")
                path_parts.append(node_name)

            # Add relation if not the last node
            if i < len(path_relation_names):
                relation = path_relation_names[i]
                path_parts.append(f" -> {relation} -> ")

        path_str = "".join(path_parts)
        return path_str

    def format_explanation_paths(self, paths: list[dict[str, Any]]) -> str:
        """Format explanation paths for the prompt.

        Args:
            paths: List of path dictionaries

        Returns:
            Formatted string of explanation paths
        """
        if not paths:
            return "No explanation paths found."

        lines = []
        for i, path in enumerate(paths, 1):
            formatted_path = self.format_path(path)
            lines.append(f"{i}. {formatted_path}")

        return " ".join(lines)

    def generate_prompt(
        self,
        uid: int,
        iid: int,
        min_attribute_score: float = 0.0,
        top_paths: int = 2,
        top_similar: int = 2,
        simple_mode: bool = True,
        include_examples: bool = True,
    ) -> str:
        """Generate an explainable recommendation prompt for a user-item pair.

        Args:
            uid: User ID
            iid: Item ID
            min_attribute_score: Minimum attribute score threshold (default: 0.0)
            top_paths: Number of top paths to include (default: 2)
            top_similar: Number of similar users/items to include (default: 2)
            simple_mode: If True, exclude CRITICAL REQUIREMENTS section (default: True)
            include_examples: If True, include Examples section (default: True, only used when simple_mode=False)

        Returns:
            Complete prompt string
        """
        # Get all necessary data
        item_title = self.data_loader.get_item_title(iid)
        item_summary = self.data_loader.get_item_summary(iid)
        user_summary = self.data_loader.get_user_summary(uid)

        attribute_prefs = self.data_loader.get_attribute_preferences(
            uid, iid, min_attribute_score
        )
        paths = self.data_loader.get_top_paths(uid, iid, top_paths)
        similar_nodes = self.data_loader.get_similar_nodes(uid, iid, top_similar)

        # Format each section
        formatted_attributes = self.format_attribute_preferences(attribute_prefs)
        formatted_similar_users = self.format_similar_users(similar_nodes["users"])
        formatted_similar_items = self.format_similar_items(similar_nodes["items"])
        formatted_paths = self.format_explanation_paths(paths)

        # Choose template based on mode
        if simple_mode:
            template = EXPLAINABLE_REC_TEMPLATE_SIMPLE
        elif not include_examples:
            template = EXPLAINABLE_REC_TEMPLATE_NO_EXAMPLES
        else:
            template = EXPLAINABLE_REC_TEMPLATE

        # Fill in the template
        prompt = template.format(
            item_title=item_title,
            item_summary=item_summary,
            user_summary=user_summary,
            attribute_preferences=formatted_attributes,
            similar_users=formatted_similar_users,
            similar_items=formatted_similar_items,
            explanation_paths=formatted_paths,
        )

        return prompt
