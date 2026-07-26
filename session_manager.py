"""
Lightweight in-memory session manager.

Keeps, per session_id:
  - a rolling window of the last N (role, content) turns (chat memory)
  - the active document_id (last PDF uploaded in this session), if any
  - a "pending_question" slot used by the ask_human tool: when the agent
    needs clarification, it stores the question here and returns it to the
    client immediately (stateless HTTP). The next request in the same
    session is treated as the human's answer to that question.

NOTE: This is intentionally process-local (a dict guarded by a lock). Swap
for Redis in production / multi-worker deployments -- the interface below
is the only thing that would need to change.
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SessionState:
    session_id: str
    history: deque = field(default_factory=deque)
    active_document_id: str | None = None
    pending_question: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionManager:
    def __init__(self):
        self._settings = get_settings()
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = SessionState(
                    session_id=session_id,
                    history=deque(maxlen=self._settings.max_memory_turns),
                )
                self._sessions[session_id] = state
                logger.info("session_created", extra={"session_id": session_id})
            return state

    def get_history(self, session_id: str) -> list[dict]:
        state = self._get_or_create(session_id)
        return list(state.history)

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        state = self._get_or_create(session_id)
        with self._lock:
            state.history.append({"role": role, "content": content})
            state.updated_at = time.time()

    def set_active_document(self, session_id: str, document_id: str) -> None:
        state = self._get_or_create(session_id)
        with self._lock:
            state.active_document_id = document_id

    def get_active_document(self, session_id: str) -> str | None:
        return self._get_or_create(session_id).active_document_id

    def set_pending_question(self, session_id: str, question: str) -> None:
        state = self._get_or_create(session_id)
        with self._lock:
            state.pending_question = question
        logger.info("pending_question_set", extra={"session_id": session_id})

    def pop_pending_question(self, session_id: str) -> str | None:
        state = self._get_or_create(session_id)
        with self._lock:
            q = state.pending_question
            state.pending_question = None
        return q

    def has_pending_question(self, session_id: str) -> bool:
        return self._get_or_create(session_id).pending_question is not None


_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
