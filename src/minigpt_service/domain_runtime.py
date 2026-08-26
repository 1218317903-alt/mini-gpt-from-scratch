"""Request-scoped private-domain Adapter runtime and HTTP control plane."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, cast

from fastapi import FastAPI, HTTPException

from agent.domain_adapter import (
    AdapterActivationBackend,
    DomainAdapterController,
    DomainAdapterRegistry,
)
from minigpt_service.schemas import (
    ChatRequest,
    DomainAuthorizationRequest,
    DomainAuthorizationResponse,
    DomainCatalogResponse,
)

if TYPE_CHECKING:
    from minigpt_service.chat_model import ChatModel

_ResultT = TypeVar("_ResultT")


class DomainRuntime:
    """Bind authorization, Adapter lifetime, and domain routes in one component."""

    def __init__(
        self,
        registry: DomainAdapterRegistry | None,
        chat_model: ChatModel | None,
    ) -> None:
        self.registry = registry
        self.controller: DomainAdapterController | None = None
        if (
            registry is not None
            and chat_model is not None
            and hasattr(chat_model, "activate_adapter")
            and hasattr(chat_model, "deactivate_adapter")
        ):
            self.controller = DomainAdapterController(
                registry,
                cast(AdapterActivationBackend, chat_model),
            )

    def run(
        self,
        request: ChatRequest,
        operation: Callable[[], _ResultT],
    ) -> _ResultT:
        """Run default or authorized inference without cross-request Adapter leaks."""
        if self.controller is None:
            if request.domain_id is not None:
                raise HTTPException(
                    status_code=503,
                    detail="Request-scoped domain adapters are not configured.",
                )
            return operation()
        if request.domain_id is None:
            return self.controller.run_default(operation)
        assert request.domain_access_token is not None
        try:
            return self.controller.run_authorized(
                request.domain_id,
                request.domain_access_token,
                operation,
            )
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    def register_routes(self, app: FastAPI) -> None:
        """Register catalog and authorization routes on an application."""

        @app.get(
            "/domains",
            response_model=DomainCatalogResponse,
            summary="List enabled private-domain adapters without paths or secrets",
        )
        def domain_catalog() -> DomainCatalogResponse:
            if self.registry is None:
                return DomainCatalogResponse(
                    enabled=False,
                    domains=[],
                    active_domain=None,
                )
            return DomainCatalogResponse(
                enabled=True,
                domains=self.registry.catalog(enabled_only=True),
                active_domain=(
                    self.controller.active_domain
                    if self.controller is not None
                    else None
                ),
            )

        @app.post(
            "/domains/{domain_id}/authorize",
            response_model=DomainAuthorizationResponse,
            summary="Validate access before request-scoped Adapter inference",
        )
        def authorize_domain(
            domain_id: str,
            request: DomainAuthorizationRequest,
        ) -> DomainAuthorizationResponse:
            if self.registry is None:
                raise HTTPException(
                    status_code=503,
                    detail="Domain registry is disabled.",
                )
            try:
                selection = self.registry.authorize(domain_id, request.access_token)
            except PermissionError as error:
                raise HTTPException(status_code=403, detail=str(error)) from error
            except FileNotFoundError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
            return DomainAuthorizationResponse(
                domain_id=selection.domain_id,
                authorized=True,
                base_model=selection.base_model,
                adult_content_enabled=selection.adult_content_enabled,
                activated=False,
            )

        @app.post(
            "/domains/{domain_id}/activate",
            response_model=DomainAuthorizationResponse,
            summary="Deprecated persistent Adapter activation endpoint",
            deprecated=True,
        )
        def activate_domain(
            domain_id: str,
            request: DomainAuthorizationRequest,
        ) -> DomainAuthorizationResponse:
            del domain_id, request
            raise HTTPException(
                status_code=409,
                detail=(
                    "Persistent Adapter activation is disabled because it can leak "
                    "across concurrent requests. Pass domain_id and "
                    "domain_access_token in /chat or /agent/chat instead."
                ),
            )

        @app.post(
            "/domains/deactivate",
            response_model=DomainCatalogResponse,
            summary="Deprecated persistent Adapter deactivation endpoint",
            deprecated=True,
        )
        def deactivate_domain() -> DomainCatalogResponse:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Persistent Adapter activation is disabled; request-scoped "
                    "adapters are restored automatically."
                ),
            )
