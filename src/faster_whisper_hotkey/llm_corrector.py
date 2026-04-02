import logging

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

        user_prompt = (
            "Please review the following speech-to-text transcription and correct it as necessary. If errors exist, repair fragmented segments and ensure proper grammar and clarity. Convert numbers and symbols written as words into their standard numerical and symbolic forms. Inferring the context may help guide your revision. If the transcription is already correct, return it unchanged. Your answer should contain only the revised text, without any additional commentary.\n\n"
            f"Transcription:\n{text}"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": user_prompt},
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
                corrected = content.strip()
                # Strip surrounding quotes (single or double), even if mismatched
                if len(corrected) >= 1:
                    if corrected.startswith(('"', "'")):
                        corrected = corrected[1:]
                    if len(corrected) >= 1 and corrected.endswith(('"', "'")):
                        corrected = corrected[:-1]
                if corrected != text:
                    logger.info(f'LLM corrected text: "{corrected}"')
                else:
                    logger.info("LLM correction: no changes needed")
                return corrected

        except Exception as e:
            logger.warning(f"LLM correction failed: {e}. Using original text.")

        return text
