"""
Service layer for metadata generation and quality scoring across different LLM providers.
Handles provider selection, initialization, and orchestration.
Uses lazy loading - providers are only initialized when needed.
"""
from typing import Dict, List, Optional, Any
import os
from llm_provider_base import (
    LLMProviderBase, 
    MetadataGenerationRequest, 
    MetadataGenerationResponse,
    QualityScoreRequest,
    QualityScoreResponse
)
from llm_provider_ollama import OllamaProvider
from llm_provider_lmstudio import LMStudioProvider
from llm_provider_chatgpt import ChatGPTProvider
from llm_provider_gemini import GeminiProvider
from llm_provider_mistral import MistralProvider
from llm_provider_anthropic import AnthropicProvider
from config import logger, DEFAULT_METADATA_PROVIDER, DEFAULT_METADATA_LANGUAGE, DEFAULT_KEYWORD_CATEGORIES
from PIL import Image, ExifTags
import io
from datetime import datetime

class AnalysisService:
    """
    Central service for managing metadata generation and quality scoring across multiple LLM providers.
    Handles provider initialization, selection, and fallback logic.
    Uses lazy loading - providers are created but models loaded only on first use.
    """
    
    def __init__(self, lazy_load=True):
        """
        Initialize the metadata service with all available providers.
        
        Args:
            lazy_load: If True, only check availability. Models are loaded on first use.
        """
        self.providers: Dict[str, LLMProviderBase] = {}
        self.lazy_load = lazy_load
        self._initialize_providers()
    
    def _initialize_providers(self):
        """
        Initialize all configured providers with lazy loading.
        Providers are created but heavy models are only loaded on first use.
        
        Note: Ollama and LM Studio availability is NOT checked here - they will be
        dynamically checked when listing models, allowing them to be started after server startup.
        """
        logger.info("Checking available LLM providers (lazy loading enabled)...")
        
        # Qwen (local) - Models loaded on first use
        # try:
        #     qwen = QwenProvider({})
        #     if qwen.is_available():
        #         self.providers['qwen'] = qwen
        #         logger.info("✓ Qwen provider available (will load on first use)")
        #     else:
        #         logger.warning("✗ Qwen provider not available (model files not found)")
        # except Exception as e:
        #     logger.error(f"✗ Failed to initialize Qwen provider: {e}")
        
        # Ollama (local) - Always register, availability checked dynamically
        try:
            ollama = OllamaProvider({})
            self.providers['ollama'] = ollama
            if ollama.is_available():
                logger.info("✓ Ollama provider available")
            else:
                logger.info("○ Ollama provider registered (server not running, can be started later)")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Ollama provider: {e}")
        
        # LM Studio (local) - Always register, availability checked dynamically
        try:
            lmstudio = LMStudioProvider({})
            self.providers['lmstudio'] = lmstudio
            if lmstudio.is_available():
                logger.info("✓ LM Studio provider initialized")
            else:
                logger.info("○ LM Studio provider registered (server not running, can be started later)")
        except Exception as e:
            logger.error(f"✗ Failed to initialize LM Studio provider: {e}")
        
        # ChatGPT (cloud) - Always add to providers, API key can be provided later
        try:
            chatgpt = ChatGPTProvider({})
            self.providers['chatgpt'] = chatgpt
            if chatgpt.is_available():
                logger.info("✓ ChatGPT provider initialized")
            else:
                logger.info("○ ChatGPT provider registered (API key not configured, can be provided later)")
        except Exception as e:
            logger.error(f"✗ Failed to initialize ChatGPT provider: {e}")
        
        # Gemini (cloud) - Always add to providers, API key can be provided later
        try:
            gemini = GeminiProvider({})
            self.providers['gemini'] = gemini
            if gemini.is_available():
                logger.info("✓ Gemini provider initialized")
            else:
                logger.info("○ Gemini provider registered (API key not configured, can be provided later)")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Gemini provider: {e}")

        # Mistral (cloud) - Always add to providers, API key can be provided later
        try:
            mistral = MistralProvider({})
            self.providers['mistral'] = mistral
            if mistral.is_available():
                logger.info("✓ Mistral provider initialized")
            else:
                logger.info("○ Mistral provider registered (API key not configured, can be provided later)")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Mistral provider: {e}")

        # Anthropic (cloud) - Always add to providers, API key can be provided later
        try:
            anthropic = AnthropicProvider({})
            self.providers['anthropic'] = anthropic
            if anthropic.is_available():
                logger.info("✓ Anthropic provider initialized")
            else:
                logger.info("○ Anthropic provider registered (API key not configured, can be provided later)")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Anthropic provider: {e}")
        
        if not self.providers:
            logger.error("⚠️  No LLM providers available! Metadata generation will not work.")
        else:
            logger.info(f"Metadata service ready with {len(self.providers)} provider(s): {', '.join(self.providers.keys())}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())

    def analyze_batch(self, image_triplets: list[tuple[bytes, str, str]], options: dict, _image_model=None, _image_processor=None,
                     uuids_needing_embeddings=None, uuids_needing_metadata=None, uuids_needing_quality=None):
        """
        Analyzes a batch of images, generating metadata and quality scores.
        Metadata text embeddings are generated later by the indexing service after
        all metadata fields have been assembled.
        Only generates data for UUIDs in the corresponding needing_* lists.
        """
        uuids = [triplet[1] for triplet in image_triplets]
        image_data = [triplet[0] for triplet in image_triplets]

        # If no specific UUIDs lists provided, generate for all (backward compatibility)
        if uuids_needing_embeddings is None:
            uuids_needing_embeddings = uuids if options.get('compute_embeddings', True) else []
        if uuids_needing_metadata is None:
            uuids_needing_metadata = uuids if options.get('compute_metadata', False) else []
        if uuids_needing_quality is None:
            uuids_needing_quality = uuids if options.get('compute_quality', True) else []

        embeddings = None
        if len(uuids_needing_embeddings) > 0:
            logger.debug("Metadata embeddings are generated by service_index after metadata assembly.")

        datetimes = self._read_datetime_from_exifs(uuids, image_data)

        metadata_results = None
        if len(uuids_needing_metadata) > 0:
            logger.info(f"Generating metadata for {len(uuids_needing_metadata)} images out of {len(uuids)} total")
            logger.info(f"UUIDs needing metadata: {uuids_needing_metadata}")
            # Filter to only process images that need metadata
            filtered_triplets = [(image_data[i], uuids[i], '') for i, uuid in enumerate(uuids) if uuid in uuids_needing_metadata]
            logger.info(f"Filtered to {len(filtered_triplets)} triplets for metadata generation")
            partial_results = self._generate_metadata_batch([t[1] for t in filtered_triplets], 
                                                           [t[0] for t in filtered_triplets], 
                                                           options)
            # Reconstruct full results array with None for images that didn't need metadata
            metadata_results = []
            partial_idx = 0
            for uuid in uuids:
                if uuid in uuids_needing_metadata:
                    metadata_results.append(partial_results[partial_idx] if partial_results else None)
                    partial_idx += 1
                else:
                    metadata_results.append(None)

        ratings = None
        if len(uuids_needing_quality) > 0:
            logger.info(f"Generating quality scores for {len(uuids_needing_quality)} images out of {len(uuids)} total")
            logger.info(f"UUIDs needing quality: {uuids_needing_quality}")
            # Filter to only process images that need quality scores
            filtered_triplets = [(image_data[i], uuids[i], '') for i, uuid in enumerate(uuids) if uuid in uuids_needing_quality]
            logger.info(f"Filtered to {len(filtered_triplets)} triplets for quality generation")
            partial_ratings = self._get_quality_scores_batch([t[1] for t in filtered_triplets], 
                                                            [t[0] for t in filtered_triplets], 
                                                            options)
            # Reconstruct full ratings array
            ratings = []
            partial_idx = 0
            for uuid in uuids:
                if uuid in uuids_needing_quality:
                    ratings.append(partial_ratings[partial_idx] if partial_ratings else None)
                    partial_idx += 1
                else:
                    ratings.append(None)
        
        return embeddings, datetimes, metadata_results, ratings

    def _read_datetime_from_exifs(self, uuids: List[str], image_data: List[bytes]) -> Dict[str, Optional[float]]:
        """
        Analyzes image data to extract capture time for a batch of images.
        """
        results = {}
        for i, data in enumerate(image_data):
            uuid = uuids[i]
            capture_time = None
            try:
                image_for_exif = Image.open(io.BytesIO(data))
                exif_data = image_for_exif.getexif()
                if exif_data:                    
                    date_time_original = exif_data.get(36867) # DateTimeOriginal
                    if date_time_original:
                        dt_object = datetime.strptime(date_time_original, '%Y:%m:%d %H:%M:%S')
                        capture_time = float(dt_object.timestamp())
            except Exception as e:
                logger.warning(f"Could not read EXIF data for {uuid}: {e}")
            
            results[uuid] = capture_time
        return results

    def _generate_metadata_batch(self, uuids: List[str], image_data: List[bytes], options: dict) -> List[Optional[MetadataGenerationResponse]]:
        """
        Generates metadata for all images in the batch.
        """
        results = []
        for i, uuid in enumerate(uuids):
            response = self.generate_metadata_single(uuid, image_data[i], options)
            results.append(response)
        return results

    def _get_quality_scores_batch(self, uuids: List[str], image_data: List[bytes], options: dict) -> List[Optional[QualityScoreResponse]]:
        """
        Generates quality scores for all images in the batch.
        """
        results = []
        for i, uuid in enumerate(uuids):
            response = self.generate_quality_scores_single(uuid, image_data[i], options)
            results.append(response)
        return results

    def generate_metadata_single(
        self,
        uuid: str,
        image_data: bytes,
        options: dict
    ) -> MetadataGenerationResponse:
        """
        Generate metadata for a single image.
        """
        provider = options.get('provider') or DEFAULT_METADATA_PROVIDER
        
        if provider not in self.providers:
            if not self.providers:
                return MetadataGenerationResponse(uuid=uuid, success=False, error="No LLM providers available")
            provider = list(self.providers.keys())[0]
            logger.warning(f"Requested provider '{provider}' not available, using fallback: {provider}")
        
        selected_provider = self.providers[provider]
        logger.info(f"Generating metadata for {uuid} using {provider}")
        
        request = MetadataGenerationRequest(
            image_data=image_data,
            uuid=uuid,
            provider=provider,
            model=options['model'],
            api_key=options.get('api_key'),
            generate_keywords=options['generate_keywords'],
            generate_caption=options['generate_caption'],
            generate_title=options['generate_title'],
            generate_alt_text=options['generate_alt_text'],
            language=options['language'],
            max_tokens=options.get('max_tokens'),
            user_prompt=options.get('user_prompt'),
            submit_gps=options['submit_gps'],
            submit_keywords=options['submit_keywords'],
            submit_folder_names=options['submit_folder_names'],
            existing_keywords=options.get('existing_keywords'),
            gps_coordinates=options.get('gps_coordinates'),
            folder_names=options.get('folder_names'),
            user_context=options.get('user_context'),
            keyword_categories=options.get('keyword_categories'),
            system_prompt=options.get('prompt'),
            date_time=options.get('photo_date') if options.get('submit_date_time') else None,
        )
        
        try:
            response = selected_provider.generate_metadata(request)
            if not response.success:
                logger.error(f"✗ Failed to generate metadata for {uuid}: {response.error}")
            return response
        except Exception as e:
            logger.error(f"Unexpected error during metadata generation for {uuid}: {e}", exc_info=True)
            return MetadataGenerationResponse(uuid=uuid, success=False, error=str(e))
    
    def generate_quality_scores_single(
        self,
        uuid: str,
        image_data: bytes,
        options: dict
    ) -> QualityScoreResponse:
        """
        Generate quality scores for a single image.
        """
        provider = options.get('provider') or DEFAULT_METADATA_PROVIDER
        
        if provider not in self.providers:
            if not self.providers:
                return QualityScoreResponse(uuid=uuid, success=False, error="No LLM providers available")
            provider = list(self.providers.keys())[0]
            logger.warning(f"Requested provider '{provider}' not available, using fallback: {provider}")
        
        selected_provider = self.providers[provider]
        logger.info(f"Generating quality scores for {uuid} using {provider}")
        
        request = QualityScoreRequest(
            image_data=image_data,
            uuid=uuid,
            provider=provider,
            model=options['model'],
            api_key=options.get('api_key'),
            language=options['language'],
            max_tokens=options.get('max_tokens'),
            system_prompt=options.get('system_prompt'),
            user_prompt=options.get('user_prompt')
        )
        
        try:
            response = selected_provider.generate_quality_scores(request)
            if not response.success:
                logger.error(f"✗ Failed to generate quality scores for {uuid}: {response.error}")
            return response
        except Exception as e:
            logger.error(f"Unexpected error during quality scoring for {uuid}: {e}", exc_info=True)
            return QualityScoreResponse(uuid=uuid, success=False, error=str(e))

    def get_available_models(
        self,
        openai_apikey: Optional[str] = None,
        gemini_apikey: Optional[str] = None,
        mistral_apikey: Optional[str] = None,
        anthropic_apikey: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Return all available multimodal (vision-capable) models from all providers.
        """
        result: Dict[str, List[str]] = {}
        cloud_provider_api_keys = {
            'chatgpt': openai_apikey,
            'gemini': gemini_apikey,
            'mistral': mistral_apikey,
            'anthropic': anthropic_apikey,
        }

        for provider_name, provider_instance in self.providers.items():
            try:
                if provider_name in cloud_provider_api_keys:
                    provider_instance.api_key = cloud_provider_api_keys[provider_name]
                
                if provider_name in ['ollama', 'lmstudio'] and not provider_instance.is_available():
                    result[provider_name] = []
                    continue
                
                models = provider_instance.list_available_models()
                result[provider_name] = models if models else []
            except Exception as e:
                logger.error(f"Error listing models for provider {provider_name}: {e}", exc_info=True)
                result[provider_name] = []
        return result

# Global service instance
_analysis_service: Optional[AnalysisService] = None

def get_analysis_service() -> AnalysisService:
    """
    Get or create the global analysis service instance.
    """
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService()
    return _analysis_service
