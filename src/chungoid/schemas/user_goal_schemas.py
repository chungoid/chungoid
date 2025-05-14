from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

class UserGoalRequest(BaseModel):
    goal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    goal_description: str = Field(..., description="Clear, concise description of the desired outcome.")
    target_platform: Optional[str] = Field(None, description="Optional: Target platform or technology (e.g., 'Python CLI', 'React Web App').")
    key_constraints: Optional[Dict[str, Any]] = Field(None, description="Optional: Key constraints or specific requirements as a dictionary.")
    # We can add more fields as needed, e.g., priority, requested_by_user_id, etc. 