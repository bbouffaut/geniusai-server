"""
ChatGPT/OpenAI Provider for metadata generation using OpenAI API
"""
import json
from typing import Dict, Any
from llm_provider_base import LLMProviderBase, MetadataGenerationRequest, MetadataGenerationResponse, QualityScoreRequest, QualityScoreResponse
from config import logger


class ChatGPTProvider(LLMProviderBase):
    """
    Provider for OpenAI ChatGPT API.
    Supports GPT-4o, GPT-4-turbo, and other vision-capable models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.timeout = config.get('timeout', 120)
        self.client = None
        self._client_api_key = None
        
        if self.api_key:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                timeout=self.timeout
            )
            self._client_api_key = self.api_key
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
            self._client_api_key = None

    def _ensure_client(self) -> bool:
        if self.api_key and (self.client is None or self._client_api_key != self.api_key):
            self._initialize_client()
        return self.is_available()
    
    def is_available(self) -> bool:
        """Check if OpenAI API is configured"""
        return self.client is not None and bool(self.api_key)
    
    def generate_metadata(self, request: MetadataGenerationRequest) -> MetadataGenerationResponse:
        """
        Generate metadata using OpenAI API.
        
        Args:
            request: MetadataGenerationRequest with image and options
            
        Returns:
            MetadataGenerationResponse with generated metadata
        """
        if request.api_key:
            self.api_key = request.api_key
        if not self._ensure_client():
            if not request.api_key:
                return MetadataGenerationResponse(
                    uuid=request.uuid,
                    success=False,
                    error="OpenAI API not configured"
                )
            return MetadataGenerationResponse(
                uuid=request.uuid,
                success=False,
                error="OpenAI API initialization failed with provided API key"
            )
        
        try:
            # Convert image to base64 data URI
            image_b64 = self._image_to_base64(request.image_data)
            data_uri = f"data:image/jpeg;base64,{image_b64}"
            
            # Prepare prompts
            system_prompt = self._prepare_system_prompt(request)
            user_prompt = self._prepare_user_prompt(request)
            
            # Prepare response format
            response_format = self._prepare_openai_response_format(request)
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri
                            }
                        }
                    ]
                }
            ]
            
            completion_params = {
                "model": request.model,
                "messages": messages,
                "response_format": response_format,
            }
            
            # GPT-5 models require reasoning_effort
            if request.model.startswith("gpt-5"):
                completion_params["reasoning_effort"] = "low"
            
            # Make API call
            self._log_llm_payload("OpenAI metadata request", completion_params)
            response = self.client.chat.completions.create(**completion_params)
            
            # Check finish reason
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                error_msg = f"OpenAI generation failed: {choice.finish_reason}"
                logger.error(error_msg)
                return MetadataGenerationResponse(
                    uuid=request.uuid,
                    success=False,
                    error=error_msg,
                    input_tokens=response.usage.prompt_tokens if response.usage else 0,
                    output_tokens=response.usage.completion_tokens if response.usage else 0
                )
            
            # Extract message content
            content = choice.message.content
            logger.debug(f"OpenAI raw response: {content}")
            
            # Parse JSON
            parsed_data = json.loads(content)
            
            # Extract metadata
            keywords = parsed_data.get("keywords", [])
            
            caption = parsed_data.get("caption") if request.generate_caption else None
            title = parsed_data.get("title") if request.generate_title else None
            alt_text = parsed_data.get("alt_text") if request.generate_alt_text else None
            
            # Token usage
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            
            return MetadataGenerationResponse(
                uuid=request.uuid,
                success=True,
                keywords=keywords,
                caption=caption,
                title=title,
                alt_text=alt_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from OpenAI response: {e}")
            return MetadataGenerationResponse(
                uuid=request.uuid,
                success=False,
                error=f"JSON parsing error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error generating metadata with OpenAI: {e}", exc_info=True)
            return MetadataGenerationResponse(
                uuid=request.uuid,
                success=False,
                error=str(e)
            )
    
    def generate_quality_scores(self, request: QualityScoreRequest) -> QualityScoreResponse:
        """
        Generate quality scores using ChatGPT API.
        
        Args:
            request: QualityScoreRequest with image
            
        Returns:
            QualityScoreResponse with quality scores and critique
        """
        
        if request.api_key:
            self.api_key = request.api_key
        if not self._ensure_client():
            if not request.api_key:
                return QualityScoreResponse(
                    uuid=request.uuid,
                    success=False,
                    error="OpenAI API not configured"
                )
            return QualityScoreResponse(
                uuid=request.uuid,
                success=False,
                error="OpenAI API initialization failed with provided API key"
            )

        try:
            # Convert image to base64 data URI
            image_b64 = self._image_to_base64(request.image_data)
            data_uri = f"data:image/jpeg;base64,{image_b64}"
            
            # Prepare quality scoring prompts using base class methods
            system_prompt = self._prepare_quality_system_prompt(request)
            user_prompt = self._prepare_quality_user_prompt(request)
            
            # Prepare OpenAI response schema
            quality_schema = {
                "type": "object",
                "properties": {
                    "overall_score": {"type": "number"},
                    "composition_score": {"type": "number"},
                    "lighting_score": {"type": "number"},
                    "motiv_score": {"type": "number"},
                    "colors_score": {"type": "number"},
                    "emotion_score": {"type": "number"},
                    "critique": {"type": "string"}
                },
                "required": ["overall_score", "composition_score", "lighting_score", "motiv_score", "colors_score", "emotion_score", "critique"],
                "additionalProperties": False
            }
            
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "quality_scores",
                    "schema": quality_schema,
                    "strict": True
                }
            }
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri}
                        }
                    ]
                }
            ]
            
            completion_params = {
                "model": request.model,
                "messages": messages,
                "response_format": response_format,
            }
            
            if request.model.startswith("gpt-5"):
                completion_params["reasoning_effort"] = "low"
            
            # Make API call
            self._log_llm_payload("OpenAI quality scoring request", completion_params)
            response = self.client.chat.completions.create(**completion_params)
            
            # Check finish reason
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                error_msg = f"Generation failed: {choice.finish_reason}"
                logger.error(error_msg)
                return QualityScoreResponse(
                    uuid=request.uuid,
                    success=False,
                    error=error_msg,
                    input_tokens=response.usage.prompt_tokens if response.usage else 0,
                    output_tokens=response.usage.completion_tokens if response.usage else 0
                )
            
            # Parse response
            content = choice.message.content
            logger.debug(f"OpenAI quality response: {content}")
            
            parsed_data = json.loads(content)
            
            # Extract scores
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            
            return QualityScoreResponse(
                uuid=request.uuid,
                success=True,
                overall_score=float(parsed_data.get("overall_score", 0)),
                composition_score=float(parsed_data.get("composition_score", 0)),
                lighting_score=float(parsed_data.get("lighting_score", 0)),
                motiv_score=float(parsed_data.get("motiv_score", 0)),
                colors_score=float(parsed_data.get("colors_score", 0)),
                emotion_score=float(parsed_data.get("emotion_score", 0)),
                critique=parsed_data.get("critique", ""),
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from OpenAI quality response: {e}")
            return QualityScoreResponse(uuid=request.uuid, success=False, error=f"JSON parsing error: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating quality scores with OpenAI: {e}", exc_info=True)
            return QualityScoreResponse(uuid=request.uuid, success=False, error=str(e))
    
    def _prepare_openai_response_format(self, request: MetadataGenerationRequest) -> Dict[str, Any]:
        """Prepare OpenAI-style response format with JSON schema"""
        schema = self._prepare_response_structure(request)
        schema["additionalProperties"] = False
        
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "metadata_response",
                "schema": schema,
                "strict": True
            }
        }
    
    def list_available_models(self) -> list:
        """
        List available OpenAI models.
        
        Args:
            only_multimodal: If True, return only vision-capable models
            
        Returns:
            List of model identifiers
        """
        if not self.api_key:
            logger.info("OpenAI API key not configured; returning no ChatGPT models")
            return []

        if not self._ensure_client():
            logger.warning("OpenAI API key is present, but client initialization failed; returning no ChatGPT models")
            return []

        try:
            self._log_llm_payload(
                "OpenAI list models request",
                {"operation": "client.models.list"},
            )
            response = self.client.models.list()
            dynamic_models = []
            for model in getattr(response, "data", response):
                model_id = self._extract_model_id(model)
                if self._is_vision_chat_model(model_id):
                    dynamic_models.append(model_id)

            dynamic_models = sorted(set(dynamic_models))
            if dynamic_models:
                logger.info(f"Returning {len(dynamic_models)} ChatGPT models from OpenAI API")
                return dynamic_models

            logger.warning("OpenAI API returned no matching vision chat models")
        except Exception as e:
            logger.error(f"Error listing OpenAI models from API: {e}", exc_info=True)

        return []

    def _extract_model_id(self, model: Any) -> str:
        if isinstance(model, dict):
            return model.get("id", "")
        return getattr(model, "id", "")

    def _is_vision_chat_model(self, model_id: str) -> bool:
        if not model_id:
            return False

        lower_model_id = model_id.lower()
        excluded_fragments = (
            "audio",
            "dall-e",
            "embedding",
            "image",
            "moderation",
            "realtime",
            "search",
            "transcribe",
            "tts",
            "whisper",
        )
        if any(fragment in lower_model_id for fragment in excluded_fragments):
            return False

        # The OpenAI models endpoint exposes availability, not modality flags, so
        # keep a conservative allow-list of chat model families this provider uses.
        supported_prefixes = (
            "gpt-4.1",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4-vision",
            "gpt-5",
        )
        return any(
            lower_model_id == prefix or lower_model_id.startswith(f"{prefix}-")
            for prefix in supported_prefixes
        )
