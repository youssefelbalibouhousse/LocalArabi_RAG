"""Limitation de débit (rate limiting) des routes sensibles.

Protège trois choses bien distinctes :
  - la SÉCURITÉ : force brute sur /login, spam de comptes sur /register ;
  - le BUDGET   : chaque appel à /ask consomme du GPU (loué) ou des tokens (API) ;
  - l'ÉQUITÉ    : un client ne doit pas monopoliser le service.

Algorithme : FENÊTRE GLISSANTE (sliding window).
On mémorise l'horodatage de chaque requête d'un client ; une requête est refusée
si le nombre d'horodatages encore dans la fenêtre atteint la limite. Contrairement
à la « fenêtre fixe », cela évite le pic en frontière (10 requêtes en 2 secondes
au passage de la minute, alors que la limite est de 5 par minute).

Limite assumée : l'état est conservé EN MÉMOIRE, donc par instance de l'API.
Avec plusieurs répliques, chaque réplique aurait son propre compteur ; il
faudrait un stockage partagé (Redis) pour un compteur global. Pour un pilote
mono-instance, c'est suffisant.
"""

from __future__ import annotations

import time
from collections import deque
from math import ceil
from typing import ClassVar

from fastapi import HTTPException, Request, status

from app import config


def client_key(request: Request) -> str:
    """Identifie le client pour le comptage.

    Derrière un reverse proxy (Caddy en production), l'adresse de la connexion
    est celle du proxy : la véritable IP du client se trouve dans l'en-tête
    X-Forwarded-For. On ne l'utilise donc que s'il est présent.

    ⚠️ Cet en-tête est falsifiable par le client. Il n'est digne de confiance que
    si l'API n'est PAS joignable directement depuis Internet — c'est le cas de
    notre architecture de production, où seul Caddy est exposé
    (voir compose.prod.yaml, qui retire le port publié de l'API).
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "inconnu"


class RateLimit:
    """Dépendance FastAPI limitant le nombre de requêtes par client.

    Utilisation :

        ask_limit = RateLimit(*config.ASK_RATE_LIMIT)

        @app.get("/ask", dependencies=[Depends(ask_limit)])
        def ask(...): ...
    """

    # Registre de toutes les instances : permet aux tests de remettre les
    # compteurs à zéro entre deux cas de test.
    # ClassVar indique qu'il s'agit d'un attribut de CLASSE (partagé), et non
    # d'un attribut d'instance — sans quoi chaque instance aurait sa propre liste.
    _instances: ClassVar[list[RateLimit]] = []

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        RateLimit._instances.append(self)

    def reset(self) -> None:
        """Vide l'historique de ce limiteur."""
        self._hits.clear()

    @classmethod
    def reset_all(cls) -> None:
        """Vide l'historique de tous les limiteurs (utilisé par les tests)."""
        for instance in cls._instances:
            instance.reset()

    def __call__(self, request: Request) -> None:
        """Point d'entrée utilisé par FastAPI (dépendance)."""
        if not config.RATE_LIMIT_ENABLED:
            return
        self.check(client_key(request))

    def check(self, key: str) -> None:
        """Applique l'algorithme pour un identifiant de client donné.

        Lève une HTTPException 429 si la limite est atteinte. Séparer cette
        méthode de `__call__` rend l'algorithme testable sans construire
        d'objet HTTP.
        """
        # time.monotonic() est insensible aux changements d'heure du système.
        now = time.monotonic()
        hits = self._hits.setdefault(key, deque())

        # 1. On oublie les requêtes sorties de la fenêtre.
        while hits and now - hits[0] >= self.window_seconds:
            hits.popleft()

        # 2. Fenêtre pleine : on refuse en indiquant quand réessayer.
        if len(hits) >= self.max_requests:
            retry_after = max(1, ceil(self.window_seconds - (now - hits[0])))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Trop de requêtes. Réessayez dans {retry_after} seconde(s).",
                headers={"Retry-After": str(retry_after)},
            )

        # 3. Sinon, on enregistre la requête dans la fenêtre.
        hits.append(now)
