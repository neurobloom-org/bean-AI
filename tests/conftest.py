from __future__ import annotations
import sys
from typing import Any
from unittest.mock import MagicMock


class _BaseAgent:
    model_fields: dict = {}
    model_config: dict = {}

    def __init__(self, *, name: str = "", **kwargs: Any) -> None:
        self.name = name

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)


class _Event:
    def __init__(self, *, author: str = "", actions: Any = None, **kwargs: Any):
        self.author = author
        self.actions = actions


class _EventActions:
    def __init__(self, *, state_delta: dict | None = None, **kwargs: Any):
        self.state_delta = state_delta or {}


class _LlmAgent(_BaseAgent):
    pass


def _install_adk_mocks() -> None:
    to_remove = [k for k in sys.modules if "google.adk" in k]
    for k in to_remove:
        del sys.modules[k]

    _adk_agents = MagicMock()
    _adk_agents.BaseAgent = _BaseAgent
    _adk_agents.LlmAgent = _LlmAgent

    _adk_invocation = MagicMock()
    _adk_invocation.InvocationContext = MagicMock()

    _adk_events = MagicMock()
    _adk_events.Event = _Event
    _adk_events.EventActions = _EventActions

    _adk_tools = MagicMock()
    _adk_tools.FunctionTool = MagicMock

    _adk_runners = MagicMock()
    _adk_runners.InMemoryRunner = MagicMock()

    _adk = MagicMock()
    _adk.agents = _adk_agents
    _adk.events = _adk_events
    _adk.tools = _adk_tools
    _adk.runners = _adk_runners

    sys.modules["google.adk"] = _adk
    sys.modules["google.adk.agents"] = _adk_agents
    sys.modules["google.adk.agents.invocation_context"] = _adk_invocation
    sys.modules["google.adk.events"] = _adk_events
    sys.modules["google.adk.tools"] = _adk_tools
    sys.modules["google.adk.runners"] = _adk_runners


_install_adk_mocks()


def pytest_collection_finish(session):
    _install_adk_mocks()
