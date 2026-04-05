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
            "Task: Professional cleanup and normalization of a speech-to-text (STT) transcription.\n\n"
            "Requirements:\n"
            "1. Structural Repair: Resolve transcription artifacts such as disjointed phrasing, "
            "incorrectly placed pauses, and run-on sentences. Reconstruct fragmented segments "
            "into grammatically correct, fluid sentences while preserving the original meaning.\n"
            "2. Holistic Normalization: Convert all spelled-out numbers, symbols, and technical "
            "notations into their standard written forms. This includes, but is not limited to:\n"
            "   - Quantitative/Financial: 'five percent' → '5%', 'twelve dollars and fifty cents' → '$12.50'.\n"
            "   - Temporal: 'October fifth twenty twenty four' → 'October 5, 2024', 'two thirty PM' → '2:30 PM'.\n"
            "   - Technical/Scientific: 'H two O' → 'H2O', 'carbon dioxide' → 'CO2', 'ten kilograms' → '10kg'.\n"
            "   - Symbols: 'press at blue sky web dot x, y, z.' → 'press@blueskyweb.xyz', 'hash tag' → '#'.\n"
            "3. Contextual Correction: Use the surrounding subject matter to correct homophones "
            "or misheard technical terms (e.g., correcting 'cell' to 'sell' or vice versa based on context).\n"
            "4. Fidelity: If the transcription is already correct, return it unchanged.\n\n"
            "Constraint: Output ONLY the revised text. Do not include any introductory remarks, "
            "explanations, or meta-commentary.\n\n"
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
                    return corrected

                logger.info("LLM correction: no correction needed")
                return corrected

        except Exception as e:
            logger.warning(f"LLM correction failed: {e}. Using original text.")

        return text
