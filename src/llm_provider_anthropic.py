"""
Anthropic Provider for metadata generation using the Anthropic Messages API.
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
from config import ANTHROPIC_API_VERSION, ANTHROPIC_BASE_URL, logger


class AnthropicProvider(LLMProviderBase):
    """
    Provider for Anthropic API.
    Supports Claude vision-capable message models.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", ANTHROPIC_BASE_URL).rstrip("/")
        self.api_version = config.get("api_version", ANTHROPIC_API_VERSION)
        self.timeout = config.get("timeout", 120)
        self.default_max_tokens = int(config.get("max_tokens") or 1024)

    def is_available(self) -> bool:
        """Check if Anthropic API is configured."""
        return bool(self.api_key)

    def generate_metadata(self, request: MetadataGenerationRequest) -> MetadataGenerationResponse:
        """
        Generate metadata using Anthropic API.

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
                error="Anthropic API not configured",
            )

        try:
            schema = self._prepare_response_structure(request)
            schema["additionalProperties"] = False
            payload = self._prepare_message_payload(
                image_data=request.image_data,
                model=request.model,
                system_prompt=self._prepare_system_prompt(request),
                user_prompt=self._prepare_user_prompt(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tool_name="record_metadata",
                tool_description="Record the requested photo metadata.",
                tool_schema=schema,
            )

            self._log_llm_payload("Anthropic metadata request", payload)
            response_data = self._post_message(api_key, payload)

            stop_reason = response_data.get("stop_reason")
            if stop_reason and stop_reason not in {"end_turn", "tool_use"}:
                error_msg = f"Anthropic generation failed: {stop_reason}"
                logger.error(error_msg)
                usage = response_data.get("usage") or {}
                return MetadataGenerationResponse(
                    uuid=request.uuid,
                    success=False,
                    error=error_msg,
                    input_tokens=self._input_tokens(usage),
                    output_tokens=self._output_tokens(usage),
                )

            parsed_data = self._extract_tool_input(response_data, "record_metadata")
            if parsed_data is None:
                content = self._extract_text_content(response_data)
                logger.debug(f"Anthropic raw response: {content}")
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
            logger.error(f"Failed to parse JSON from Anthropic response: {e}")
            return MetadataGenerationResponse(
                uuid=request.uuid,
                success=False,
                error=f"JSON parsing error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error generating metadata with Anthropic: {e}", exc_info=True)
            return MetadataGenerationResponse(uuid=request.uuid, success=False, error=str(e))

    def generate_quality_scores(self, request: QualityScoreRequest) -> QualityScoreResponse:
        """
        Generate quality scores using Anthropic API.

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
                error="Anthropic API not configured",
            )

        try:
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
            payload = self._prepare_message_payload(
                image_data=request.image_data,
                model=request.model,
                system_prompt=self._prepare_quality_system_prompt(request),
                user_prompt=self._prepare_quality_user_prompt(request),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tool_name="record_quality_scores",
                tool_description="Record the requested photo quality scores and critique.",
                tool_schema=quality_schema,
            )

            self._log_llm_payload("Anthropic quality scoring request", payload)
            response_data = self._post_message(api_key, payload)

            stop_reason = response_data.get("stop_reason")
            if stop_reason and stop_reason not in {"end_turn", "tool_use"}:
                error_msg = f"Anthropic quality generation failed: {stop_reason}"
                logger.error(error_msg)
                usage = response_data.get("usage") or {}
                return QualityScoreResponse(
                    uuid=request.uuid,
                    success=False,
                    error=error_msg,
                    input_tokens=self._input_tokens(usage),
                    output_tokens=self._output_tokens(usage),
                )

            parsed_data = self._extract_tool_input(response_data, "record_quality_scores")
            if parsed_data is None:
                content = self._extract_text_content(response_data)
                logger.debug(f"Anthropic quality response: {content}")
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
            logger.error(f"Failed to parse JSON from Anthropic quality response: {e}")
            return QualityScoreResponse(
                uuid=request.uuid,
                success=False,
                error=f"JSON parsing error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Error generating quality scores with Anthropic: {e}", exc_info=True)
            return QualityScoreResponse(uuid=request.uuid, success=False, error=str(e))

    def list_available_models(self) -> list:
        """
        List Anthropic vision-capable models.

        Returns:
            List of model identifiers
        """
        api_key = self._get_api_key(None)
        if not api_key:
            logger.info("Anthropic API key not configured; returning no Anthropic models")
            return []

        try:
            dynamic_models = []
            for model in self._get_model_cards(api_key):
                model_id = model.get("id") or model.get("name")
                if self._is_vision_messages_model(model_id, model):
                    dynamic_models.append(model_id)

            dynamic_models = sorted(set(dynamic_models))
            if dynamic_models:
                logger.info(f"Returning {len(dynamic_models)} Anthropic models from API")
                return dynamic_models

            logger.warning("Anthropic API returned no matching vision message models")
        except Exception as e:
            logger.error(f"Error listing Anthropic models from API: {e}", exc_info=True)

        return []

    def _get_api_key(self, request_api_key: Optional[str]) -> Optional[str]:
        if request_api_key:
            self.api_key = request_api_key
        return request_api_key or self.api_key

    def _prepare_message_payload(
        self,
        image_data: bytes,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: Optional[int],
        tool_name: str,
        tool_description: str,
        tool_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        image_b64 = self._image_to_base64(image_data)
        payload = {
            "model": model,
            "max_tokens": self._max_tokens(max_tokens),
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
            "tools": [
                {
                    "name": tool_name,
                    "description": tool_description,
                    "input_schema": tool_schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": tool_name},
            "temperature": temperature,
        }

        return payload

    def _post_message(self, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/messages",
            headers=self._headers(api_key, include_content_type=True),
            json=payload,
            timeout=self.timeout,
        )

        if not response.ok:
            raise RuntimeError(self._format_http_error(response))

        return response.json()

    def _get_model_cards(self, api_key: str) -> list:
        models = []
        params = {"limit": 1000}

        while True:
            models_url = f"{self.base_url}/models"
            debug_headers = self._headers(api_key)
            debug_headers["x-api-key"] = f"<redacted; length={len(api_key)}>"
            self._log_llm_payload(
                "Anthropic list models request",
                {
                    "method": "GET",
                    "url": models_url,
                    "headers": debug_headers,
                    "params": params,
                },
            )
            response = requests.get(
                models_url,
                headers=self._headers(api_key),
                params=params,
                timeout=self.timeout,
            )

            if not response.ok:
                raise RuntimeError(self._format_http_error(response))

            response_data = response.json()
            models.extend(self._extract_model_cards(response_data))

            if not isinstance(response_data, dict) or not response_data.get("has_more"):
                break

            last_id = response_data.get("last_id")
            if not last_id:
                break
            params = {"limit": 1000, "after_id": last_id}

        return models

    def _headers(self, api_key: str, include_content_type: bool = False) -> Dict[str, str]:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.api_version,
        }
        if include_content_type:
            headers["Content-Type"] = "application/json"
        return headers

    def _extract_model_cards(self, response_data: Any) -> list:
        if isinstance(response_data, list):
            return [item for item in response_data if isinstance(item, dict)]
        if isinstance(response_data, dict):
            data = response_data.get("data") or []
            return [item for item in data if isinstance(item, dict)]
        return []

    def _is_vision_messages_model(self, model_id: Optional[str], model: Dict[str, Any]) -> bool:
        if not model_id:
            return False

        model_type = model.get("type")
        if model_type and model_type != "model":
            return False

        lower_model_id = model_id.lower()
        if not lower_model_id.startswith("claude-"):
            return False

        excluded_prefixes = (
            "claude-2",
            "claude-instant",
        )
        return not lower_model_id.startswith(excluded_prefixes)

    def _extract_tool_input(self, response_data: Dict[str, Any], tool_name: str) -> Optional[Dict[str, Any]]:
        for content_block in response_data.get("content") or []:
            if not isinstance(content_block, dict):
                continue

            if content_block.get("type") != "tool_use" or content_block.get("name") != tool_name:
                continue

            tool_input = content_block.get("input")
            if isinstance(tool_input, dict):
                return tool_input
            if isinstance(tool_input, str):
                parsed_input = json.loads(tool_input)
                if isinstance(parsed_input, dict):
                    return parsed_input

        return None

    def _extract_text_content(self, response_data: Dict[str, Any]) -> str:
        text_chunks = []
        for content_block in response_data.get("content") or []:
            if isinstance(content_block, dict):
                text = content_block.get("text")
                if isinstance(text, str):
                    text_chunks.append(text)
            elif isinstance(content_block, str):
                text_chunks.append(content_block)

        if text_chunks:
            return "\n".join(text_chunks)

        raise ValueError("Anthropic returned no usable message content")

    def _clean_json_response(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _max_tokens(self, max_tokens: Optional[int]) -> int:
        if max_tokens is None or max_tokens == "":
            return self.default_max_tokens
        return int(max_tokens)

    def _input_tokens(self, usage: Dict[str, Any]) -> int:
        return int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)

    def _output_tokens(self, usage: Dict[str, Any]) -> int:
        return int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)

    def _format_http_error(self, response: requests.Response) -> str:
        try:
            body = response.json()
            message = None
            if isinstance(body.get("error"), dict):
                message = body["error"].get("message")
            if not message:
                message = body.get("message")
            if not message:
                message = json.dumps(body)
        except Exception:
            message = response.text[:500]

        return f"Anthropic API request failed with status {response.status_code}: {message}"
