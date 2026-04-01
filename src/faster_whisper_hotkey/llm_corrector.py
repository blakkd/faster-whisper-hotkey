import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class LLMCorrector:
    """
    Handles transcription correction via an OpenAI-compatible API endpoint.
    """

    def __init__(self, endpoint: str, model_name: str):
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.model_name = model_name

    def correct(self, text: str) -> str:
        """
        Send transcribed text to the LLM for correction and return the corrected version.
        Falls back to original text on error.
        """
        if not text.strip():
            return text

        prompt = (
            "Please correct any errors in the following speech-to-text transcription. "
            "Repair broken segments, ensure proper grammar and clear formulation, "
            "and preserve or infer the original meaning. Your answer should contain "
            "only the corrected text, nothing else.\n\n"
            f"Transcription:\n{text}"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a transcription proofreader."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.6,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                self.endpoint, json=payload, headers=headers, timeout=120
            )
            response.raise_for_status()

            result = response.json()
            content = (
                result.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            if content:
                logger.info(f'LLM corrected: "{text}" -> "{content.strip()}"')
                return content.strip()

        except Exception as e:
            logger.warning(f"LLM correction failed: {e}. Using original text.")

        return text
