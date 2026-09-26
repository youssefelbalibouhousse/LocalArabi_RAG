"""Tests d'INTÉGRATION de l'API : on traverse toute la pile HTTP.

La base de données est en mémoire (voir conftest.py) et le RAG est SIMULÉ
(monkeypatch), donc aucun appel réseau vers Ollama n'est effectué.
Les tests restent rapides et reproductibles.
"""

from app import config, rag

# --- Santé ---------------------------------------------------------------

def test_health_repond_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# --- Configuration publique ----------------------------------------------

def test_config_publique_expose_allow_registration(client):
    response = client.get("/config")

    assert response.status_code == 200
    assert "allow_registration" in response.json()
    assert response.json()["languages"] == ["ar", "fr"]


def test_config_publique_ne_fuite_aucun_secret(client):
    """Garde-fou : cette route est publique, aucun secret ne doit y apparaître."""
    texte = str(client.get("/config").json()).lower()

    assert "secret" not in texte
    assert "password" not in texte


# --- Inscription ---------------------------------------------------------

def test_register_cree_un_compte(client):
    response = client.post(
        "/register",
        json={"username": "alice", "password": "secret123"},
    )

    assert response.status_code == 200
    corps = response.json()
    assert corps["username"] == "alice"
    # Le mot de passe (ni son empreinte) ne doit JAMAIS sortir de l'API.
    assert "password" not in corps
    assert "hashed_password" not in corps


def test_register_refuse_un_nom_deja_pris(client):
    client.post("/register", json={"username": "alice", "password": "secret123"})

    response = client.post(
        "/register",
        json={"username": "alice", "password": "autre-mot-de-passe"},
    )

    assert response.status_code == 400


def test_register_valide_le_corps_de_la_requete(client):
    """Champ manquant → erreur de validation 422 (gérée par Pydantic)."""
    response = client.post("/register", json={"username": "alice"})

    assert response.status_code == 422


def test_register_refuse_si_inscription_desactivee(client, monkeypatch):
    """Pilote : l'inscription publique peut être fermée aux inconnus."""
    monkeypatch.setattr(config, "ALLOW_REGISTRATION", False)

    response = client.post(
        "/register",
        json={"username": "intrus", "password": "secret123"},
    )

    assert response.status_code == 403


def test_register_autorise_si_inscription_activee(client, monkeypatch):
    monkeypatch.setattr(config, "ALLOW_REGISTRATION", True)

    response = client.post(
        "/register",
        json={"username": "alice", "password": "secret123"},
    )

    assert response.status_code == 200


# --- Connexion -----------------------------------------------------------

def test_login_renvoie_un_jeton(client):
    client.post("/register", json={"username": "alice", "password": "secret123"})

    response = client.post(
        "/login",
        data={"username": "alice", "password": "secret123"},
    )

    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert response.json()["access_token"]


def test_login_refuse_un_mauvais_mot_de_passe(client):
    client.post("/register", json={"username": "alice", "password": "secret123"})

    response = client.post(
        "/login",
        data={"username": "alice", "password": "mauvais"},
    )

    assert response.status_code == 401


def test_login_refuse_un_utilisateur_inconnu(client):
    response = client.post(
        "/login",
        data={"username": "inconnu", "password": "secret123"},
    )

    assert response.status_code == 401


# --- Profil protégé ------------------------------------------------------

def test_me_renvoie_l_utilisateur_connecte(client, auth_headers):
    response = client.get("/me", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["username"] == "tester"


def test_me_refuse_sans_jeton(client):
    assert client.get("/me").status_code == 401


def test_me_refuse_un_jeton_invalide(client):
    response = client.get("/me", headers={"Authorization": "Bearer jeton-bidon"})

    assert response.status_code == 401


# --- /ask (route protégée) -----------------------------------------------

def test_ask_refuse_sans_jeton(client):
    response = client.get("/ask", params={"question": "une question"})

    assert response.status_code == 401


def test_ask_repond_avec_sources_et_extraits(client, auth_headers, monkeypatch):
    """Le RAG est simulé : on vérifie la logique de l'API, pas celle du modèle."""
    monkeypatch.setattr(rag, "get_collection", lambda: "collection-simulee")
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda collection, question, n_results=None: (
            ["Le texte de la source."],
            [{"source": "a.pdf", "page": 3, "line_start": 1, "line_end": 5}],
        ),
    )
    monkeypatch.setattr(
        rag,
        "generate",
        lambda question, context, language="ar": "Réponse simulée.",
    )

    response = client.get(
        "/ask",
        params={"question": "une question", "language": "fr"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    corps = response.json()
    assert corps["language"] == "fr"
    assert "Réponse simulée." in corps["answer"]
    assert "a.pdf — page 3 (lignes 1-5)" in corps["answer"]
    assert "Le texte de la source." in corps["answer"]
    assert corps["sources"][0]["page"] == 3
    assert corps["context_used"] == ["Le texte de la source."]


def test_ask_relance_le_modele_si_la_langue_n_est_pas_respectee(
    client, auth_headers, monkeypatch
):
    """Bout en bout : le modèle répond en arabe alors que « fr » est demandé.

    Le contexte est en arabe (comme dans la réalité), ce qui pousse le modèle à
    répondre en arabe. L'API doit détecter la langue et faire réécrire.
    """
    monkeypatch.setattr(rag, "get_collection", lambda: "collection-simulee")
    monkeypatch.setattr(
        rag,
        "retrieve",
        lambda collection, question, n_results=None: (
            ["نص المصدر المستخرج من الوثيقة."],
            [{"source": "a.pdf", "page": 1, "line_start": 1, "line_end": 2}],
        ),
    )

    reponses = ["هذه إجابة بالعربية عن السؤال.", "Voici la réponse en français."]
    monkeypatch.setattr(rag, "generate", lambda *args, **kwargs: reponses.pop(0))

    response = client.get(
        "/ask",
        params={"question": "une question", "language": "fr"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    corps = response.json()
    assert corps["language"] == "fr"
    assert "Voici la réponse en français." in corps["answer"]
    assert "هذه إجابة" not in corps["answer"]


def test_ask_repond_en_arabe_par_defaut(client, auth_headers, monkeypatch):
    monkeypatch.setattr(rag, "get_collection", lambda: "collection-simulee")
    monkeypatch.setattr(rag, "retrieve", lambda collection, question, n_results=None: ([], []))
    monkeypatch.setattr(rag, "generate", lambda *args, **kwargs: "inutilisé")

    response = client.get("/ask", params={"question": "سؤال"}, headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["language"] == "ar"
    assert "لا توجد معلومات كافية" in response.json()["answer"]


def test_ask_repond_sans_appeler_le_llm_si_aucun_document(client, auth_headers, monkeypatch):
    """Sans contexte pertinent, le LLM ne doit PAS être sollicité."""
    monkeypatch.setattr(rag, "get_collection", lambda: "collection-simulee")
    monkeypatch.setattr(rag, "retrieve", lambda collection, question, n_results=None: ([], []))

    def generate_interdit(*args, **kwargs):
        raise AssertionError("generate() ne doit pas être appelé sans contexte")

    monkeypatch.setattr(rag, "generate", generate_interdit)

    response = client.get(
        "/ask",
        params={"question": "question sans réponse", "language": "fr"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert "pas assez d'informations" in response.json()["answer"]
    assert response.json()["sources"] == []
