"""Path finder for exploring l-hop connections in knowledge graphs."""

from collections import deque
from typing import Dict, List, Set, Tuple


class PathFinder:
    """Finds l-hop paths between user-item pairs in the knowledge graph."""

    def __init__(
        self,
        kg_dict: Dict[int, List[Tuple[int, int]]],
        n_entities: int,
    ):
        """Initialize the path finder.

        Args:
            kg_dict: Dictionary mapping head -> [(tail, relation), ...]
            n_entities: Number of entities (items are 0 to n_entities-1)
        """
        self.kg_dict = kg_dict
        self.n_entities = n_entities

    def find_paths(
        self,
        user_id: int,
        item_id: int,
        max_hops: int,
        max_paths: int = None,
    ) -> List[List[Tuple[int, int, int]]]:
        """Find all paths from user to item within max_hops.

        Args:
            user_id: User ID (adjusted for entities offset)
            item_id: Item ID
            max_hops: Maximum number of hops
            max_paths: Maximum number of paths to return (None for all)

        Returns:
            List of paths, where each path is a list of (head, relation, tail) tuples
        """
        paths = []

        # BFS to find all paths
        # Queue contains: (current_node, path_so_far, visited_nodes)
        queue = deque([(user_id, [], {user_id})])

        while queue:
            current, path, visited = queue.popleft()

            # Check if we've reached the maximum number of paths
            if max_paths is not None and len(paths) >= max_paths:
                break

            # Check if we've reached the target
            if current == item_id and len(path) > 0:
                paths.append(path)
                if max_paths is not None and len(paths) >= max_paths:
                    break
                continue

            # Check if we've exceeded max hops
            if len(path) >= max_hops:
                continue

            # Explore neighbors
            if current in self.kg_dict:
                for tail, relation in self.kg_dict[current]:
                    # Avoid cycles
                    if tail not in visited:
                        new_path = path + [(current, relation, tail)]
                        new_visited = visited | {tail}
                        queue.append((tail, new_path, new_visited))

        return paths

    def find_paths_all_hops(
        self,
        user_id: int,
        item_id: int,
        max_hops: int,
        max_paths_per_hop: int = None,
    ) -> Dict[int, List[List[Tuple[int, int, int]]]]:
        """Find paths grouped by hop count.

        Args:
            user_id: User ID (adjusted for entities offset)
            item_id: Item ID
            max_hops: Maximum number of hops
            max_paths_per_hop: Maximum paths per hop count (None for all)

        Returns:
            Dictionary mapping hop_count -> list of paths
        """
        paths_by_hop = {i: [] for i in range(1, max_hops + 1)}

        # BFS to find all paths
        queue = deque([(user_id, [], {user_id})])

        while queue:
            current, path, visited = queue.popleft()

            # Check if we've exceeded max hops
            if len(path) >= max_hops:
                # Check if we've reached the target at max hops
                if current == item_id and len(path) > 0:
                    hop_count = len(path)
                    if (
                        max_paths_per_hop is None
                        or len(paths_by_hop[hop_count]) < max_paths_per_hop
                    ):
                        paths_by_hop[hop_count].append(path)
                continue

            # Check if we've reached the target
            if current == item_id and len(path) > 0:
                hop_count = len(path)
                if (
                    max_paths_per_hop is None
                    or len(paths_by_hop[hop_count]) < max_paths_per_hop
                ):
                    paths_by_hop[hop_count].append(path)
                # Don't continue from here - we want shortest paths
                continue

            # Explore neighbors
            if current in self.kg_dict:
                for tail, relation in self.kg_dict[current]:
                    # Avoid cycles
                    if tail not in visited:
                        new_path = path + [(current, relation, tail)]
                        new_visited = visited | {tail}
                        queue.append((tail, new_path, new_visited))

        return paths_by_hop

    def count_paths(
        self,
        user_id: int,
        item_id: int,
        max_hops: int,
    ) -> int:
        """Count the number of paths from user to item within max_hops.

        Args:
            user_id: User ID (adjusted for entities offset)
            item_id: Item ID
            max_hops: Maximum number of hops

        Returns:
            Number of paths found
        """
        count = 0

        # BFS to count paths
        queue = deque([(user_id, 0, {user_id})])

        while queue:
            current, depth, visited = queue.popleft()

            # Check if we've exceeded max hops
            if depth >= max_hops:
                continue

            # Explore neighbors
            if current in self.kg_dict:
                for tail, relation in self.kg_dict[current]:
                    # Avoid cycles
                    if tail not in visited:
                        if tail == item_id:
                            count += 1
                        else:
                            new_visited = visited | {tail}
                            queue.append((tail, depth + 1, new_visited))

        return count

    def find_shortest_paths(
        self,
        user_id: int,
        item_id: int,
        max_hops: int,
    ) -> List[List[Tuple[int, int, int]]]:
        """Find all shortest paths from user to item.

        Args:
            user_id: User ID (adjusted for entities offset)
            item_id: Item ID
            max_hops: Maximum number of hops to search

        Returns:
            List of shortest paths (all have the same length)
        """
        # BFS to find shortest path length
        queue = deque([(user_id, [], {user_id})])
        shortest_paths = []
        shortest_length = None

        while queue:
            current, path, visited = queue.popleft()

            # If we've found paths and this path is longer, stop
            if shortest_length is not None and len(path) > shortest_length:
                continue

            # Check if we've exceeded max hops
            if len(path) >= max_hops:
                continue

            # Check if we've reached the target
            if current == item_id and len(path) > 0:
                if shortest_length is None:
                    shortest_length = len(path)
                    shortest_paths.append(path)
                elif len(path) == shortest_length:
                    shortest_paths.append(path)
                continue

            # Explore neighbors
            if current in self.kg_dict:
                for tail, relation in self.kg_dict[current]:
                    # Avoid cycles
                    if tail not in visited:
                        new_path = path + [(current, relation, tail)]
                        new_visited = visited | {tail}
                        queue.append((tail, new_path, new_visited))

        return shortest_paths
