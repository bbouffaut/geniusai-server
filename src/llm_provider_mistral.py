"""
Mistral AI Provider for metadata generation using the Mistral Chat Completions API
"""
import json
from typing import Any, Dict, Optional

import requests

from llm_provider_base import (
    LLMProviderBase,
    MetadataGenerationRequest,
    MetadataGenerationResponse,
    QualityScoreRequest,
    QualityScoreResponse,
)
from config import logger, MISTRAL_BASE_URL


class MistralProvider(LLMProviderBase):
    """
    Provider for Mistral AI API.
    Supports Mistral and Ministral vision-capable chat completion models.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", MISTRAL_BASE_URL).rstrip("/")
        self.timeout = config.get("timeout", 120)

    def is_available(self) -> bool:
        """Check if Mistral API is configured"""
        return bool(self.api_key)

    def generate_metadata(self, request: MetadataGenerationRequest) -> MetadataGenerationResponse:
        """
        Generate metadata using Mistral AI API.

        Args:
            request: MetadataGenerationRequest with image and options

        Returns:
            MetadataGenerationResponse with generated metadata
        """
        api_key = self._get_api_key(request.api_key)
        if not api_key:
            return MetadataGenerationResponse(
                uuid=request.uuid,
                success=False,
                error="Mistral API not configured",
            )

        try:
            image_b64 = self._image_to_base64(request.image_data)
            data_uri = f"data:image/jpeg;base64,{image_b64}"

            system_prompt = self._prepare_system_prompt(request)
            user_prompt = self._prepare_user_prompt(request)
            response_format = self._prepare_mistral_response_format(request)

            payload = {
                "model": request.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": data_uri},
                        ],
                    },
                ],
                "response_format": response_format,
            }
            self._add_max_tokens(payload, request.max_tokens)

            self._log_llm_payload("Mistral metadata request", payload)
            response_data = self._post_chat_completion(api_key, payload)

            choice = self._first_choice(response_data)
            finish_reason = choice.get("finish_reason")
            if finish_reason and finish_reason != "stop":
                error_msg = f"Mistral generation failed: {finish_reason}"
                logger.error(error_msg)
                usage = response_data.get("usage") or {}
                return MetadataGenerationResponse(
                    uuid=request.uuid,
                    success=False,
                    error=error_msg,
                    input_tokens=self._input_tokens(usage),
                    output_tokens=self._output_tokens(usage),
                )

            content = self._extract_message_content(choice)
            logger.debug(f"Mistral raw response: {content}")
            parsed_data = json.loads(self._clean_json_response(content))

            keywords = parsed_data.get("keywords", [])
            caption = parsed_data.get("caption") if request.generate_caption else None
            title = parsed_data.get("title") if request.generate_title else None
            alt_text = parsed_data.get("alt_text") if request.generate_alt_text else None

            usage = response_data.get("usage") or {}
            return MetadataGenerationResponse(
                uuid=request.uuid,
                success=True,
                keywords=keywords,
                caption=caption,
                title=title,
                alt_text=alt_text,
                input_tokens=self._input_tokens(usage),
                output_tokens=self._output_tokens(usage),
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Mistral response: {e}")
            return MetadataGenerationResponse(
                uuid=request.uuid,
                success=False,
                error=f"JSON parsing error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error generating metadata with Mistral: {e}", exc_info=True)
            return MetadataGenerationResponse(uuid=request.uuid, success=False, error=str(e))

    def generate_quality_scores(self, request: QualityScoreRequest) -> QualityScoreResponse:
        """
        Generate quality scores using Mistral AI API.

        Args:
            request: QualityScoreRequest with image

        Returns:
            QualityScoreResponse with quality scores and critique
        """
        api_key = self._get_api_key(request.api_key)
        if not api_key:
            return QualityScoreResponse(
                uuid=request.uuid,
                success=False,
                error="Mistral API not configured",
            )

        try:
            image_b64 = self._image_to_base64(request.image_data)
            data_uri = f"data:image/jpeg;base64,{image_b64}"

            system_prompt = self._prepare_quality_system_prompt(request)
            user_prompt = self._prepare_quality_user_prompt(request)
            quality_schema = {
                "type": "object",
                "properties": {
                    "overall_score": {"type": "number"},
                    "composition_score": {"type": "number"},
                    "lighting_score": {"type": "number"},
                    "motiv_score": {"type": "number"},
                    "colors_score": {"type": "number"},
                    "emotion_score": {"type": "number"},
                    "critique": {"type": "string"},
                },
                "required": [
                    "overall_score",
                    "composition_score",
                    "lighting_score",
                    "motiv_score",
                    "colors_score",
                    "emotion_score",
                    "critique",
                ],
                "additionalProperties": False,
            }

            payload = {
                "model": request.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": data_uri},
                        ],
                    },
                ],
                "response_format": self._json_schema_response_format(
                    "quality_scores", quality_schema
                ),
            }
            self._add_max_tokens(payload, request.max_tokens)

            self._log_llm_payload("Mistral quality scoring request", payload)
            response_data = self._post_chat_completion(api_key, payload)

            choice = self._first_choice(response_data)
            finish_reason = choice.get("finish_reason")
            if finish_reason and finish_reason != "stop":
                error_msg = f"Mistral quality generation failed: {finish_reason}"
                logger.error(error_msg)
                usage = response_data.get("usage") or {}
                return QualityScoreResponse(
                    uuid=request.uuid,
                    success=False,
                    error=error_msg,
                    input_tokens=self._input_tokens(usage),
                    output_tokens=self._output_tokens(usage),
                )

            content = self._extract_message_content(choice)
            logger.debug(f"Mistral quality response: {content}")
            parsed_data = json.loads(self._clean_json_response(content))

            usage = response_data.get("usage") or {}
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
                input_tokens=self._input_tokens(usage),
                output_tokens=self._output_tokens(usage),
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Mistral quality response: {e}")
            return QualityScoreResponse(
                uuid=request.uuid,
                success=False,
                error=f"JSON parsing error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error generating quality scores with Mistral: {e}", exc_info=True)
            return QualityScoreResponse(uuid=request.uuid, success=False, error=str(e))

    def list_available_models(self) -> list:
        """
        List Mistral vision-capable models.

        Returns:
            List of model identifiers
        """
        api_key = self._get_api_key(None)
        if not api_key:
            logger.info("Mistral API key not configured; returning no Mistral models")
            return []

        try:
            models_url = f"{self.base_url}/models"
            self._log_llm_payload(
                "Mistral list models request",
                {
                    "method": "GET",
                    "url": models_url,
                    "headers": {
                        "Authorization": f"Bearer <redacted; length={len(api_key)}>",
                    },
                },
            )
            response = requests.get(
                models_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=self.timeout,
            )

            if not response.ok:
                raise RuntimeError(self._format_http_error(response))

            response_data = response.json()
            dynamic_models = []
            for model in self._extract_model_cards(response_data):
                model_id = model.get("id") or model.get("name")
                if self._is_vision_chat_model(model_id, model):
                    dynamic_models.append(model_id)

            dynamic_models = sorted(set(dynamic_models))
            if dynamic_models:
                logger.info(f"Returning {len(dynamic_models)} Mistral models from API")
                return dynamic_models

            logger.warning("Mistral API returned no matching vision chat models")
        except Exception as e:
            logger.error(f"Error listing Mistral models from API: {e}", exc_info=True)

        return []

    def _get_api_key(self, request_api_key: Optional[str]) -> Optional[str]:
        if request_api_key:
            self.api_key = request_api_key
        return request_api_key or self.api_key

    def _post_chat_completion(self, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )

        if not response.ok:
            raise RuntimeError(self._format_http_error(response))

        return response.json()

    def _extract_model_cards(self, response_data: Any) -> list:
        if isinstance(response_data, list):
            return [item for item in response_data if isinstance(item, dict)]
        if isinstance(response_data, dict):
            data = response_data.get("data") or []
            return [item for item in data if isinstance(item, dict)]
        return []

    def _is_vision_chat_model(self, model_id: Optional[str], model: Dict[str, Any]) -> bool:
        if not model_id or model.get("archived"):
            return False

        capabilities = model.get("capabilities") or {}
        if capabilities:
            return bool(capabilities.get("vision")) and bool(capabilities.get("completion_chat", True))

        fallback_model_ids = {
            "mistral-large-2512",
            "mistral-medium-2508",
            "mistral-small-2506",
            "ministral-14b-2512",
            "ministral-8b-2512",
            "ministral-3b-2512",
        }
        return model_id in fallback_model_ids

    def _prepare_mistral_response_format(
        self, request: MetadataGenerationRequest
    ) -> Dict[str, Any]:
        schema = self._prepare_response_structure(request)
        schema["additionalProperties"] = False
        return self._json_schema_response_format("metadata_response", schema)

    def _json_schema_response_format(self, name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
                "strict": True,
            },
        }

    def _add_max_tokens(self, payload: Dict[str, Any], max_tokens: Optional[int]) -> None:
        if max_tokens is None or max_tokens == "":
            return
        payload["max_tokens"] = int(max_tokens)

    def _first_choice(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        choices = response_data.get("choices") or []
        if not choices:
            raise ValueError("Mistral returned no choices")
        return choices[0]

    def _extract_message_content(self, choice: Dict[str, Any]) -> str:
        message = choice.get("message") or {}
        content = message.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_chunks = []
            for chunk in content:
                if isinstance(chunk, dict):
                    if isinstance(chunk.get("text"), str):
                        text_chunks.append(chunk["text"])
                    elif isinstance(chunk.get("content"), str):
                        text_chunks.append(chunk["content"])
                elif isinstance(chunk, str):
                    text_chunks.append(chunk)
            return "\n".join(text_chunks)

        if isinstance(content, dict):
            return json.dumps(content)

        raise ValueError("Mistral returned no usable message content")

    def _clean_json_response(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _input_tokens(self, usage: Dict[str, Any]) -> int:
        return int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)

    def _output_tokens(self, usage: Dict[str, Any]) -> int:
        return int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

    def _format_http_error(self, response: requests.Response) -> str:
        try:
            body = response.json()
            message = body.get("message")
            if not message and isinstance(body.get("error"), dict):
                message = body["error"].get("message")
            if not message:
                message = json.dumps(body)
        except Exception:
            message = response.text[:500]

        return f"Mistral API request failed with status {response.status_code}: {message}"
