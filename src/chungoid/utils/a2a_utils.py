"""Utilities for Agent-to-Agent (A2A) communication and coordination."""

import uuid
import json
from typing import List, Optional

# Import needed for runtime type hints
from .reflection_store import ReflectionStore, Reflection

# Forward reference for type hint (still useful for complex cases / organization)
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from .reflection_store import ReflectionStore, Reflection

def generate_correlation_id() -> str:
    """Generates a new UUID4-based correlation ID."""
    return str(uuid.uuid4())

def get_reflections_by_correlation_id(
    store: ReflectionStore, correlation_id: str
) -> List[Reflection]:
    """Retrieves all reflections from the store that match the given correlation_id."""
    # Query all reflections (or a reasonable limit if the store is huge and query() is inefficient)
    # ReflectionStore.query() already filters by conversation_id or agent_id if provided.
    # We need to fetch broadly and then filter by our specific metadata field.
    # Assuming store.query() can fetch all or we handle pagination if necessary in a real scenario.
    all_reflections = store.query(limit=2000) # Arbitrary high limit, adjust if needed
    
    matched_reflections: List[Reflection] = []
    for reflection in all_reflections:
        # Ensure metadata and extra exist and extra is a string before trying loads
        extra_str = None
        if reflection.metadata and isinstance(reflection.metadata, dict):
            extra_field = reflection.metadata.get("extra")
            if isinstance(extra_field, str):
                extra_str = extra_field

        if extra_str:
            try:
                extra_data = json.loads(extra_str)
                if extra_data.get("correlation_id") == correlation_id:
                    matched_reflections.append(reflection)
            except json.JSONDecodeError:
                # Malformed JSON in 'extra', skip this reflection for correlation_id check
                continue 
    return matched_reflections 