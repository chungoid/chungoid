from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """
    Abstract Base Class for an LLM interaction provider.
    Defines a standard interface for sending prompts to an LLM and receiving responses.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        # TODO: Consider adding stop_sequences, system_prompt (if not part of main prompt)
        **kwargs: Any  # For provider-specific parameters
    ) -> str:
        """
        Sends a prompt to the LLM and returns the generated text response.

        Args:
            prompt: The main prompt string for the LLM.
            model_id: Optional; specific model identifier if overriding a default.
            temperature: Optional; sampling temperature.
            max_tokens: Optional; maximum number of tokens to generate.
            kwargs: Additional provider-specific arguments.

        Returns:
            The LLM's generated text response.
        """
        pass


class MockLLMProvider(LLMProvider):
    """
    A mock implementation of LLMProvider for testing and development.
    It can be configured to return predefined responses or echo prompts.
    """

    def __init__(self, predefined_responses: Optional[Dict[str, str]] = None):
        self.predefined_responses = predefined_responses if predefined_responses else {}
        # Track calls for testing purposes
        self.calls: List[Dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        self.calls.append({
            "prompt": prompt,
            "model_id": model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs
        })

        logger.info(f"MockLLMProvider received prompt (first 100 chars): {prompt[:100]}...")
        logger.info(f"MockLLMProvider args: model_id={model_id}, temp={temperature}, max_tokens={max_tokens}")

        # Check if a direct match for the prompt exists in predefined_responses
        if prompt in self.predefined_responses:
            response = self.predefined_responses[prompt]
            logger.info(f"MockLLMProvider returning predefined response for direct prompt match: {response}")
            return response

        # Check for partial matches (e.g., if prompt starts with a key)
        for key, predefined_response in self.predefined_responses.items():
            if prompt.strip().startswith(key.strip()):
                logger.info(f"MockLLMProvider returning predefined response for partial match (key: '{key}'): {predefined_response}")
                return predefined_response
        
        # Default behavior: echo back a modified prompt or a generic message
        default_response = f"Mock response for prompt: {prompt[:150]}..."
        logger.info(f"MockLLMProvider returning default echo response: {default_response}")
        return default_response

    def get_last_call_args(self) -> Optional[Dict[str, Any]]:
        return self.calls[-1] if self.calls else None

# Example of how a concrete implementation might look (conceptual)
# class OpenAILLMProvider(LLMProvider):
#     def __init__(self, api_key: str, default_model: str = "gpt-3.5-turbo"):
#         import openai # type: ignore
#         self.client = openai.AsyncOpenAI(api_key=api_key)
#         self.default_model = default_model

#     async def generate(
#         self,
#         prompt: str,
#         model_id: Optional[str] = None,
#         temperature: Optional[float] = 0.7, # Default temperature
#         max_tokens: Optional[int] = 1500,   # Default max_tokens
#         **kwargs: Any
#     ) -> str:
#         chosen_model = model_id or self.default_model
#         try:
#             logger.info(f"OpenAILLMProvider calling model: {chosen_model} with prompt (first 100 chars): {prompt[:100]}...")
#             # Note: OpenAI API might prefer a messages format for chat models
#             # This is a simplified example for completion-style interaction.
#             # For chat models, you'd construct a messages list:
#             # messages = [{"role": "user", "content": prompt}]
#             # response = await self.client.chat.completions.create(
#             #     model=chosen_model,
#             #     messages=messages,
#             #     temperature=temperature,
#             #     max_tokens=max_tokens,
#             #     **kwargs
#             # )
#             # return response.choices[0].message.content or ""
            
#             # Using a more generic completion for this example, assuming `prompt` is a full prompt
#             # This might vary significantly based on the chosen model and its capabilities.
#             # The actual implementation needs to align with the specific OpenAI client library and model type.
            
#             # Placeholder for actual OpenAI call which might be different
#             # For instance, for newer models, it's often chat completions.
#             # This is illustrative.
#             if "chat.completions" in chosen_model: # A guess
#                 response = await self.client.chat.completions.create(
#                     model=chosen_model,
#                     messages=[{"role": "system", "content": "You are a helpful assistant."}, # Or use a passed system_prompt
#                               {"role": "user", "content": prompt}],
#                     temperature=temperature,
#                     max_tokens=max_tokens,
#                     **kwargs
#                 )
#                 return response.choices[0].message.content or ""
#             else: # Fallback for older completion models, if any
#                 response = await self.client.completions.create(
#                     model=chosen_model,
#                     prompt=prompt,
#                     temperature=temperature,
#                     max_tokens=max_tokens,
#                     **kwargs
#                 )
#                 return response.choices[0].text.strip()

#         except Exception as e:
#             logger.error(f"Error calling OpenAI API: {e}")
#             # Consider specific error handling or re-raising
#             raise # Re-raise the exception for the caller to handle

#         return "Error: Could not get response from OpenAI." # Fallback 