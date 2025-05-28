"""UnifiedAgent - UAEI Base Class (Phase 1)

Single interface for ALL agent execution - eliminates dual interface complexity.
According to enhanced_cycle.md Phase 1 implementation.
Enhanced with refinement capabilities for intelligent iteration cycles.
Enhanced with comprehensive JSON validation infrastructure.
"""

from __future__ import annotations

import logging
import time
import os
import asyncio
import json
import re
from abc import ABC
from typing import Any, ClassVar, List, Optional, Dict, Type, Union
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, ValidationError

from ..schemas.unified_execution_schemas import (
    ExecutionContext,
    ExecutionConfig,
    AgentExecutionResult,
    ExecutionMode,
    ExecutionMetadata,
    CompletionReason,
    CompletionAssessment,
    IterationResult,
    ToolMode,
)
from ..utils.llm_provider import LLMProvider
from ..utils.prompt_manager import PromptManager

__all__ = ["UnifiedAgent", "JsonValidationConfig", "JsonExtractionStrategy"]


class JsonExtractionStrategy(Enum):
    """Strategies for extracting JSON from LLM responses."""
    MARKDOWN_FIRST = "markdown_first"  # Try markdown code blocks first
    BRACKET_MATCHING = "bracket_matching"  # Use bracket matching algorithm
    MULTI_STRATEGY = "multi_strategy"  # Try multiple strategies in sequence
    REPAIR_ENABLED = "repair_enabled"  # Enable JSON repair for malformed JSON


class JsonValidationConfig(BaseModel):
    """Configuration for JSON validation in agents."""
    
    # Extraction configuration
    extraction_strategy: JsonExtractionStrategy = JsonExtractionStrategy.MULTI_STRATEGY
    enable_json_repair: bool = True
    max_extraction_retries: int = 3
    
    # Response format configuration
    request_json_format: bool = True
    enable_json_mode: bool = True
    use_tool_calling: bool = True
    
    # Validation configuration
    enable_schema_validation: bool = True
    strict_validation: bool = False
    allow_partial_validation: bool = True
    
    # Fallback configuration
    enable_llm_repair: bool = True
    max_repair_attempts: int = 2
    fallback_to_text: bool = True
    
    # Performance configuration
    cache_extracted_json: bool = True
    validate_async: bool = False


class UnifiedAgent(BaseModel, ABC):
    """
    Single interface for ALL agent execution - eliminates dual interface complexity.
    Replaces: invoke_async, execute_with_protocol, execute_with_protocols
    
    Phase 1: Basic unified interface with delegation to existing methods
    Phase 2: Direct implementation of agent logic
    Phase 3: Enhanced multi-iteration cycles
    Phase 4: Intelligent refinement with MCP tools and ChromaDB integration
    Phase 5: Comprehensive JSON validation infrastructure
    """
    
    # Required class metadata (enforced by validation)
    AGENT_ID: ClassVar[str]
    AGENT_VERSION: ClassVar[str] 
    PRIMARY_PROTOCOLS: ClassVar[List[str]]
    CAPABILITIES: ClassVar[List[str]]
    
    # Standard initialization
    llm_provider: LLMProvider = Field(..., description="LLM provider for AI capabilities")
    prompt_manager: PromptManager = Field(..., description="Prompt manager for templates")
    
    # Refinement capabilities (Phase 4 enhancement)
    enable_refinement: bool = Field(default=True, description="Enable intelligent refinement using MCP tools and ChromaDB")
    mcp_tools: Optional[Any] = Field(default=None, description="MCP tools registry for refinement capabilities")
    chroma_client: Optional[Any] = Field(default=None, description="ChromaDB client for storing/querying agent outputs")
    
    # JSON validation infrastructure (Phase 5 enhancement)
    json_validation_config: JsonValidationConfig = Field(default_factory=JsonValidationConfig, description="JSON validation configuration")
    json_cache: Dict[str, Any] = Field(default_factory=dict, description="Cache for extracted JSON")
    
    # Internal
    logger: Optional[logging.Logger] = Field(default=None)
    
    # Model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.logger is None:
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Initialize refinement capabilities if enabled
        if self.enable_refinement:
            self._initialize_refinement_capabilities()

    def get_id(self) -> str:
        """Get the agent's unique identifier"""
        return self.AGENT_ID

    # ========================================
    # PHASE 5: JSON VALIDATION INFRASTRUCTURE
    # ========================================

    async def _request_json_response(self, prompt: str, schema: Optional[Type[BaseModel]] = None, **llm_kwargs) -> str:
        """
        Request JSON response from LLM with proper format configuration.
        
        Args:
            prompt: The prompt to send to the LLM
            schema: Optional Pydantic schema for structured output
            **llm_kwargs: Additional arguments for LLM provider
            
        Returns:
            Raw LLM response string
        """
        # Configure JSON response format
        if self.json_validation_config.request_json_format:
            # Add JSON format request to kwargs
            if "response_format" not in llm_kwargs:
                llm_kwargs["response_format"] = {"type": "json_object"}
            
            # Enable JSON mode if supported
            if self.json_validation_config.enable_json_mode:
                llm_kwargs["json_mode"] = True
        
        # Use tool calling for structured output if available and schema provided
        if (self.json_validation_config.use_tool_calling and 
            schema and 
            hasattr(self.llm_provider, 'supports_tool_calling') and 
            self.llm_provider.supports_tool_calling()):
            
            # Create tool definition from schema
            tool_definition = {
                "type": "function",
                "function": {
                    "name": "structured_response",
                    "description": f"Provide structured response for {self.AGENT_ID}",
                    "parameters": schema.model_json_schema()
                }
            }
            
            llm_kwargs["tools"] = [tool_definition]
            llm_kwargs["tool_choice"] = {"type": "function", "function": {"name": "structured_response"}}
        
        try:
            # Make LLM request with JSON configuration
            response = await self.llm_provider.generate_async(
                prompt=prompt,
                **llm_kwargs
            )
            
            self.logger.debug(f"[JSON] Received LLM response for {self.AGENT_ID}")
            return response
            
        except Exception as e:
            self.logger.error(f"[JSON] LLM request failed for {self.AGENT_ID}: {e}")
            raise

    def _extract_json_from_response(self, response: str) -> str:
        """
        Universal JSON extraction with multiple strategies and robust error handling.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Extracted JSON string
            
        Raises:
            ValueError: If no valid JSON can be extracted
        """
        if not response or not response.strip():
            raise ValueError("Empty response provided")
        
        # Check cache first
        cache_key = hash(response)
        if (self.json_validation_config.cache_extracted_json and 
            cache_key in self.json_cache):
            return self.json_cache[cache_key]
        
        response = response.strip()
        extracted_json = None
        
        # Strategy selection based on configuration
        strategy = self.json_validation_config.extraction_strategy
        
        if strategy == JsonExtractionStrategy.MARKDOWN_FIRST:
            extracted_json = self._extract_from_markdown(response)
            if not extracted_json:
                extracted_json = self._extract_with_bracket_matching(response)
                
        elif strategy == JsonExtractionStrategy.BRACKET_MATCHING:
            extracted_json = self._extract_with_bracket_matching(response)
            
        elif strategy == JsonExtractionStrategy.MULTI_STRATEGY:
            # Try multiple strategies in order of reliability
            strategies = [
                self._extract_from_markdown,
                self._extract_from_code_blocks,
                self._extract_with_bracket_matching,
                self._extract_from_xml_tags,
                self._extract_with_regex_patterns
            ]
            
            for strategy_func in strategies:
                try:
                    extracted_json = strategy_func(response)
                    if extracted_json and self._is_valid_json_syntax(extracted_json):
                        break
                except Exception as e:
                    self.logger.debug(f"[JSON] Strategy {strategy_func.__name__} failed: {e}")
                    continue
        
        # If extraction failed and repair is enabled, try repair
        if (not extracted_json and 
            self.json_validation_config.enable_json_repair):
            extracted_json = self._repair_malformed_json(response)
        
        # If still no extraction, try repair on the original response directly
        if (not extracted_json and 
            self.json_validation_config.enable_json_repair):
            # Try to repair the entire response as potential JSON
            extracted_json = self._repair_malformed_json(response.strip())
        
        # Final validation
        if not extracted_json:
            raise ValueError(f"Could not extract valid JSON from response: {response[:200]}...")
        
        # Cache successful extraction
        if self.json_validation_config.cache_extracted_json:
            self.json_cache[cache_key] = extracted_json
        
        return extracted_json

    def _extract_from_markdown(self, response: str) -> Optional[str]:
        """Extract JSON from markdown code blocks (```json)."""
        # Look for ```json code blocks
        json_pattern = r'```json\s*\n(.*?)\n```'
        match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Fallback to generic code blocks starting with {
        generic_pattern = r'```\s*\n(\{.*?\})\s*\n```'
        match = re.search(generic_pattern, response, re.DOTALL)
        
        if match:
            potential_json = match.group(1).strip()
            if self._is_valid_json_syntax(potential_json):
                return potential_json
        
        return None

    def _extract_from_code_blocks(self, response: str) -> Optional[str]:
        """Extract JSON from generic code blocks."""
        lines = response.split('\n')
        json_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```') and not in_code_block:
                in_code_block = True
                continue
            elif line.strip() == '```' and in_code_block:
                break
            elif in_code_block:
                json_lines.append(line)
        
        if json_lines:
            potential_json = '\n'.join(json_lines).strip()
            if self._is_valid_json_syntax(potential_json):
                return potential_json
        
        return None

    def _extract_with_bracket_matching(self, response: str) -> Optional[str]:
        """Extract JSON using bracket matching algorithm."""
        # Find first opening brace
        start_idx = response.find('{')
        if start_idx == -1:
            return None
        
        # Count braces to find matching closing brace
        brace_count = 0
        for i, char in enumerate(response[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found matching closing brace
                    potential_json = response[start_idx:i+1]
                    if self._is_valid_json_syntax(potential_json):
                        return potential_json
                    # Continue looking for another JSON object
                    continue
        
        return None

    def _extract_from_xml_tags(self, response: str) -> Optional[str]:
        """Extract JSON from XML-style tags like <output>...</output>."""
        # Common XML tag patterns
        tag_patterns = [
            r'<output>(.*?)</output>',
            r'<json>(.*?)</json>',
            r'<result>(.*?)</result>',
            r'<response>(.*?)</response>'
        ]
        
        for pattern in tag_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                potential_json = match.group(1).strip()
                if self._is_valid_json_syntax(potential_json):
                    return potential_json
        
        return None

    def _extract_with_regex_patterns(self, response: str) -> Optional[str]:
        """Extract JSON using various regex patterns."""
        # Pattern for JSON objects that might be embedded in text
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested object pattern
            r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}',    # More complex nesting
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                if self._is_valid_json_syntax(match):
                    return match
        
        return None

    def _is_valid_json_syntax(self, json_str: str) -> bool:
        """Check if string is valid JSON syntax."""
        try:
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    def _repair_malformed_json(self, json_str: str) -> Optional[str]:
        """Attempt to repair malformed JSON using various strategies."""
        if not json_str:
            return None
        
        # Try json-repair library if available
        try:
            import json_repair
            repaired = json_repair.repair_json(json_str)
            if self._is_valid_json_syntax(repaired):
                self.logger.debug("[JSON] Successfully repaired JSON using json_repair")
                return repaired
        except ImportError:
            self.logger.debug("[JSON] json_repair library not available")
        except Exception as e:
            self.logger.debug(f"[JSON] json_repair failed: {e}")
        
        # Manual repair strategies
        repaired = self._manual_json_repair(json_str)
        if repaired and self._is_valid_json_syntax(repaired):
            self.logger.debug("[JSON] Successfully repaired JSON manually")
            return repaired
        
        # Advanced repair: Try to extract JSON-like content from prose
        if not self._is_valid_json_syntax(json_str):
            # Look for JSON-like patterns in the text
            json_like_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested object pattern
                r'\{.*?\}',  # Simple object pattern
            ]
            
            for pattern in json_like_patterns:
                matches = re.findall(pattern, json_str, re.DOTALL)
                for match in matches:
                    repaired = self._manual_json_repair(match)
                    if repaired and self._is_valid_json_syntax(repaired):
                        self.logger.debug("[JSON] Successfully extracted and repaired JSON from prose")
                        return repaired
        
        # Last resort: Try to create minimal valid JSON from any structured content
        if '{' in json_str and '}' in json_str:
            # Extract content between first { and last }
            start = json_str.find('{')
            end = json_str.rfind('}')
            if start < end:
                potential_json = json_str[start:end+1]
                repaired = self._manual_json_repair(potential_json)
                if repaired and self._is_valid_json_syntax(repaired):
                    self.logger.debug("[JSON] Successfully repaired JSON using last resort extraction")
                    return repaired
        
        return None

    def _manual_json_repair(self, json_str: str) -> Optional[str]:
        """Manual JSON repair strategies."""
        # Common fixes
        fixes = [
            # Fix trailing commas
            (r',(\s*[}\]])', r'\1'),
            # Fix missing quotes around keys
            (r'(\w+):', r'"\1":'),
            # Fix single quotes to double quotes
            (r"'([^']*)'", r'"\1"'),
            # Fix missing commas between objects
            (r'}\s*{', r'},{'),
            # Fix missing commas between array elements
            (r']\s*\[', r'],['),
            # Fix missing commas between key-value pairs
            (r'"\s*"([^"]+)":', r'",\n    "\1":'),
            # Fix missing commas after values
            (r'(["\d\]}])\s*"([^"]+)":', r'\1,\n    "\2":'),
        ]
        
        repaired = json_str
        for pattern, replacement in fixes:
            repaired = re.sub(pattern, replacement, repaired)
        
        return repaired

    def _validate_json_against_schema(self, json_str: str, schema: Type[BaseModel]) -> BaseModel:
        """
        Validate JSON string against Pydantic schema.
        
        Args:
            json_str: JSON string to validate
            schema: Pydantic model class for validation
            
        Returns:
            Validated Pydantic model instance
            
        Raises:
            ValidationError: If validation fails
        """
        if not self.json_validation_config.enable_schema_validation:
            # Return raw dict if validation disabled
            return json.loads(json_str)
        
        try:
            # Use Pydantic's model_validate_json for validation
            validated_model = schema.model_validate_json(json_str)
            self.logger.debug(f"[JSON] Successfully validated against {schema.__name__}")
            return validated_model
            
        except ValidationError as e:
            if self.json_validation_config.strict_validation:
                raise
            
            # Try partial validation if allowed
            if self.json_validation_config.allow_partial_validation:
                try:
                    # Parse JSON and validate with partial data
                    json_data = json.loads(json_str)
                    # Remove invalid fields and try again
                    cleaned_data = self._clean_data_for_schema(json_data, schema)
                    validated_model = schema.model_validate(cleaned_data)
                    self.logger.warning(f"[JSON] Partial validation successful for {schema.__name__}")
                    return validated_model
                except Exception as partial_error:
                    self.logger.error(f"[JSON] Partial validation failed: {partial_error}")
            
            # Re-raise original validation error
            raise e

    def _clean_data_for_schema(self, data: Dict[str, Any], schema: Type[BaseModel]) -> Dict[str, Any]:
        """Clean data to match schema requirements with type coercion and defaults."""
        if not isinstance(data, dict):
            return data
        
        # Get schema fields
        schema_fields = schema.model_fields
        cleaned_data = {}
        
        # Process existing fields with type coercion
        for key, value in data.items():
            if key in schema_fields:
                field_info = schema_fields[key]
                # Attempt type coercion for common mismatches
                try:
                    # Handle string numbers
                    if field_info.annotation == float and isinstance(value, str):
                        # Handle common non-numeric strings
                        if value.lower() in ('not_a_number', 'nan', 'null', 'none', ''):
                            cleaned_data[key] = 0.0
                        else:
                            cleaned_data[key] = float(value)
                    elif field_info.annotation == int and isinstance(value, str):
                        if value.lower() in ('not_a_number', 'nan', 'null', 'none', ''):
                            cleaned_data[key] = 0
                        else:
                            cleaned_data[key] = int(float(value))  # Handle "1.0" -> 1
                    # Handle string booleans
                    elif field_info.annotation == bool and isinstance(value, str):
                        cleaned_data[key] = value.lower() in ('true', '1', 'yes', 'on')
                    # Handle numeric to string conversion
                    elif field_info.annotation == str and isinstance(value, (int, float)):
                        cleaned_data[key] = str(value)
                    else:
                        cleaned_data[key] = value
                except (ValueError, TypeError) as e:
                    # If coercion fails, provide sensible defaults
                    self.logger.debug(f"[JSON] Type coercion failed for {key}: {e}")
                    if field_info.annotation == float:
                        cleaned_data[key] = 0.0
                    elif field_info.annotation == int:
                        cleaned_data[key] = 0
                    elif field_info.annotation == bool:
                        cleaned_data[key] = False
                    elif field_info.annotation == str:
                        cleaned_data[key] = str(value)
                    else:
                        cleaned_data[key] = value
            else:
                self.logger.debug(f"[JSON] Removing invalid field: {key}")
        
        # Add default values for missing required fields
        for field_name, field_info in schema_fields.items():
            if field_name not in cleaned_data:
                # Check if field has a default (handle different Pydantic versions)
                has_default = False
                try:
                    # Try pydantic_core first (Pydantic 2.x)
                    from pydantic_core import PydanticUndefined
                    has_default = field_info.default is not PydanticUndefined
                except ImportError:
                    try:
                        # Try pydantic (older versions)
                        from pydantic import PydanticUndefined
                        has_default = field_info.default is not PydanticUndefined
                    except ImportError:
                        # Fallback for very old versions
                        has_default = hasattr(field_info, 'default') and field_info.default is not ...
                
                if has_default:
                    cleaned_data[field_name] = field_info.default
                elif (hasattr(field_info, 'default_factory') and 
                      field_info.default_factory is not None and
                      callable(field_info.default_factory)):
                    try:
                        cleaned_data[field_name] = field_info.default_factory()
                        has_default = True  # Successfully used default factory
                    except Exception as e:
                        self.logger.debug(f"[JSON] Default factory failed for {field_name}: {e}")
                        has_default = False  # Treat as no default
                
                # If no valid default, generate one
                if not has_default and field_info.is_required():
                    # Provide sensible defaults for required fields
                    import typing
                    annotation = field_info.annotation
                    
                    # Handle typing annotations
                    if hasattr(annotation, '__origin__'):
                        if annotation.__origin__ is list:
                            cleaned_data[field_name] = []
                        elif annotation.__origin__ is dict:
                            cleaned_data[field_name] = {}
                        else:
                            cleaned_data[field_name] = f"[Missing {field_name}]"
                    elif annotation == str:
                        cleaned_data[field_name] = f"[Missing {field_name}]"
                    elif annotation == float:
                        cleaned_data[field_name] = 0.0
                    elif annotation == int:
                        cleaned_data[field_name] = 0
                    elif annotation == bool:
                        cleaned_data[field_name] = False
                    elif str(annotation).startswith('list'):
                        cleaned_data[field_name] = []
                    elif annotation == dict:
                        cleaned_data[field_name] = {}
                    else:
                        # For complex types, provide a placeholder
                        cleaned_data[field_name] = f"[Missing {field_name}]"
                    
                    self.logger.debug(f"[JSON] Added default value for missing required field: {field_name}")
        
        return cleaned_data

    async def _retry_json_extraction(self, response: str, max_retries: int = None) -> str:
        """
        Retry JSON extraction with different strategies.
        
        Args:
            response: Raw LLM response
            max_retries: Maximum number of retry attempts
            
        Returns:
            Extracted JSON string
        """
        if max_retries is None:
            max_retries = self.json_validation_config.max_extraction_retries
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return self._extract_json_from_response(response)
            except Exception as e:
                last_error = e
                self.logger.debug(f"[JSON] Extraction attempt {attempt + 1} failed: {e}")
                
                # Try LLM-based repair if enabled
                if (attempt < max_retries and 
                    self.json_validation_config.enable_llm_repair):
                    response = await self._llm_repair_json(response)
        
        # All retries failed
        raise last_error or ValueError("JSON extraction failed after all retries")

    async def _llm_repair_json(self, malformed_response: str) -> str:
        """Use LLM to repair malformed JSON response."""
        repair_prompt = f"""
        The following response contains malformed JSON. Please fix it and return only valid JSON:
        
        {malformed_response}
        
        Return only the corrected JSON, no explanations or markdown formatting.
        """
        
        try:
            repaired_response = await self.llm_provider.generate_async(
                prompt=repair_prompt,
                response_format={"type": "json_object"},
                temperature=0.1  # Low temperature for consistent repair
            )
            
            self.logger.debug("[JSON] LLM repair attempt completed")
            return repaired_response
            
        except Exception as e:
            self.logger.error(f"[JSON] LLM repair failed: {e}")
            return malformed_response  # Return original if repair fails

    async def _extract_and_validate_json(
        self, 
        response: str, 
        schema: Optional[Type[BaseModel]] = None,
        fallback_to_text: bool = None
    ) -> Union[BaseModel, Dict[str, Any], str]:
        """
        Complete JSON extraction and validation pipeline.
        
        Args:
            response: Raw LLM response
            schema: Optional Pydantic schema for validation
            fallback_to_text: Whether to fallback to text if JSON extraction fails
            
        Returns:
            Validated model, dict, or text depending on configuration and success
        """
        if fallback_to_text is None:
            fallback_to_text = self.json_validation_config.fallback_to_text
        
        try:
            # Extract JSON
            json_str = await self._retry_json_extraction(response)
            
            # Validate against schema if provided
            if schema:
                return self._validate_json_against_schema(json_str, schema)
            else:
                # Return parsed JSON dict
                return json.loads(json_str)
                
        except Exception as e:
            self.logger.error(f"[JSON] Complete extraction/validation failed: {e}")
            
            if fallback_to_text:
                self.logger.warning("[JSON] Falling back to text response")
                return response
            else:
                raise

    # ========================================
    # EXISTING METHODS (Phase 1-4)
    # ========================================

    def _initialize_refinement_capabilities(self):
        """Initialize MCP tools and ChromaDB for refinement capabilities"""
        try:
            # Initialize MCP tools registry if not provided
            if self.mcp_tools is None:
                from chungoid.mcp_tools import get_mcp_tools_registry
                self.mcp_tools = get_mcp_tools_registry()
                self.logger.info(f"[Refinement] Initialized MCP tools registry for {self.AGENT_ID}")
            
            # Initialize ChromaDB client if not provided
            if self.chroma_client is None:
                import chromadb
                self.chroma_client = chromadb.Client()
                self.logger.info(f"[Refinement] Initialized ChromaDB client for {self.AGENT_ID}")
                
        except Exception as e:
            self.logger.warning(f"[Refinement] Failed to initialize refinement capabilities: {e}")
            self.enable_refinement = False

    # ... existing code ...