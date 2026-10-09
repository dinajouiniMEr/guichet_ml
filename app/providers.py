
from app.config import HF_TOKEN, MODEL_ID
from app.schemas import Prediction


class MockProvider:
    name = "mock"

    def detect_language(self, text: str) -> list[Prediction]:
        # Détection simplifiée de l'arabe
        if any("\u0600" <= c <= "\u06ff" for c in text):
            return [Prediction(label="ar", score=0.97)]

        # Quelques mots caractéristiques du français
        mots_fr = {
            "je", "mon", "ma", "compte",
            "bonjour", "pas", "plus"
        }

        mots = set(text.lower().replace(".", "").split())

        if mots & mots_fr:
            return [
                Prediction(label="fr", score=0.95),
                Prediction(label="it", score=0.03)
            ]

        # Par défaut, la règle classe le texte en anglais
        return [
            Prediction(label="en", score=0.93),
            Prediction(label="de", score=0.04)
        ]


class HfProvider:
    name = "huggingface"

    def __init__(self):
        from huggingface_hub import InferenceClient

        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN absent")

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=HF_TOKEN
        )

    def detect_language(self, text: str) -> list[Prediction]:
        brut = self.client.text_classification(
            text,
            model=MODEL_ID,
            top_k=3
        )

        return [
            Prediction(label=item.label, score=item.score)
            for item in brut
        ]


def get_provider(mode: str):
    if mode == "real":
        try:
            return HfProvider()
        except Exception:
            return MockProvider()

    return MockProvider()