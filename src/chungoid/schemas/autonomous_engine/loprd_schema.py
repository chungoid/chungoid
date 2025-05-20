"""Pydantic models for the LLM-Optimized Product Requirements Document (LOPRD)."""
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, conlist
import datetime
import uuid

# --- Nested Models ---

class LOPRDMetadata(BaseModel):
    """Metadata about the LOPRD itself."""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this LOPRD document.")
    project_name: str = Field(..., description="Name of the project.")
    version: str = Field(..., pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$", description="Semantic version of this LOPRD (e.g., 1.0.0).")
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, description="Timestamp of creation.")
    last_updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, description="Timestamp of last update.")
    authors: List[str] = Field(..., description="Agent(s) or person(s) who authored/updated this document.")
    source_goal_id: str = Field(..., description="ID of the refined_user_goal.md this LOPRD is based on.")

class ProjectOverview(BaseModel):
    """High-level overview of the project."""
    executive_summary: str = Field(..., description="A brief summary of the project and its objectives.")
    target_audience: str = Field(..., description="Description of the primary users or consumers of the project output.")
    problem_statement: str = Field(..., description="The core problem this project aims to solve.")
    solution_statement: str = Field(..., description="A high-level description of the proposed solution.")

class Scope(BaseModel):
    """Defines the boundaries of the project."""
    in_scope: List[str] = Field(..., description="List of features/functionalities that are explicitly in scope.")
    out_of_scope: List[str] = Field(..., description="List of features/functionalities that are explicitly out of scope.")

class UserStory(BaseModel):
    """Represents a single user story."""
    id: str = Field(..., description="Unique identifier for the user story (e.g., US001).")
    as_a: str = Field(..., description="The type of user.", examples=["As a [type of user]"])
    i_want_to: str = Field(..., description="The action the user wants to perform.", examples=["I want [an action]"])
    so_that: str = Field(..., description="The benefit or value the user gains.", examples=["so that [a benefit/value]"])
    priority: Literal["High", "Medium", "Low"] = Field(..., description="Priority of the user story.")
    notes: Optional[str] = Field(None, description="Additional notes or context for the user story.")

class AcceptanceCriterion(BaseModel):
    """Represents an acceptance criterion for a functional requirement."""
    id: str = Field(..., description="Unique identifier for the acceptance criterion (e.g., AC001).")
    description: str = Field(..., description="A specific, testable criterion that must be met.")

class FunctionalRequirement(BaseModel):
    """Represents a detailed functional requirement."""
    id: str = Field(..., description="Unique identifier for the functional requirement (e.g., FR001).")
    description: str = Field(..., description="Detailed description of the functional requirement.")
    source_user_story_ids: Optional[List[str]] = Field(default_factory=list, description="List of User Story IDs this FR helps fulfill.")
    acceptance_criteria: conlist(AcceptanceCriterion, min_length=1) = Field(..., description="List of acceptance criteria for this functional requirement.")

class NonFunctionalRequirement(BaseModel):
    """Represents a non-functional requirement."""
    id: str = Field(..., description="Unique identifier for the NFR (e.g., NFR001).")
    category: Literal["Performance", "Security", "Usability", "Reliability", "Maintainability", "Scalability", "Accessibility", "Portability", "Other"] = Field(..., description="Category of the NFR.")
    description: str = Field(..., description="Detailed description of the non-functional requirement.")
    metric: str = Field(..., description="How this NFR will be measured or verified (e.g., 'Response time < 200ms for 99% of requests').")

class DataDictionaryGlossaryItem(BaseModel):
    """An item in the data dictionary or glossary."""
    term: str = Field(...)
    definition: str = Field(...)
    related_terms: Optional[List[str]] = Field(default_factory=list)

# --- Main LOPRD Model ---

class LOPRD(BaseModel):
    """
    LLM-Optimized Product Requirements Document (LOPRD).
    A comprehensive, structured document detailing project requirements,
    optimized for LLM consumption and generation.
    Corresponds to loprd_schema.json.
    """
    loprd_metadata: LOPRDMetadata = Field(...)
    project_overview: ProjectOverview = Field(...)
    scope: Scope = Field(...)
    user_stories: List[UserStory] = Field(default_factory=list)
    functional_requirements: List[FunctionalRequirement] = Field(default_factory=list)
    non_functional_requirements: List[NonFunctionalRequirement] = Field(default_factory=list)
    assumptions: Optional[List[str]] = Field(default_factory=list)
    constraints: Optional[List[str]] = Field(default_factory=list)
    data_dictionary_glossary: Optional[List[DataDictionaryGlossaryItem]] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "LLM-Optimized Product Requirements Document (LOPRD)",
            "description": "A comprehensive, structured document detailing project requirements, optimized for LLM consumption and generation."
        }

__all__ = [
    "LOPRDMetadata",
    "ProjectOverview",
    "Scope",
    "UserStory",
    "AcceptanceCriterion",
    "FunctionalRequirement",
    "NonFunctionalRequirement",
    "DataDictionaryGlossaryItem",
    "LOPRD"
] 