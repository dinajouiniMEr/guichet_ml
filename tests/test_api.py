
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_mode_mock():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["mode"] == "mock"


def test_detection_francais():
    response = client.post(
        "/detect-language",
        json={
            "text": "Je ne peux plus acceder a mon compte."
        }
    )

    assert response.status_code == 200
    assert response.json()["language"] == "fr"


def test_detection_arabe():
    response = client.post(
        "/detect-language",
        json={
            "text": "مرحبا، أريد الدخول إلى حسابي"
        }
    )

    assert response.status_code == 200
    assert response.json()["language"] == "ar"


def test_texte_trop_court_rejete():
    response = client.post(
        "/detect-language",
        json={"text": "ab"}
    )

    assert response.status_code == 422



def test_champ_text_absent():
    response = client.post(
        "/detect-language",
        json={}
    )

    assert response.status_code == 422


def test_fournisseur_en_panne(monkeypatch):
    def fournisseur_indisponible(text):
        raise RuntimeError("Panne simulée")

    monkeypatch.setattr(
        "app.main.provider.detect_language",
        fournisseur_indisponible
    )

    response = client.post(
        "/detect-language",
        json={"text": "Bonjour tout le monde"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Service d'inference indisponible"
    )