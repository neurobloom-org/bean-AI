"""
shared/exceptions.py
----------------------
Custom exceptions for Bean AI system.
All other modules import exceptions from here.
"""


class BeanBaseException(Exception):
    """Base exception for all Bean AI errors."""
    pass


class TranscriptionError(BeanBaseException):
    """Raised when speech-to-text fails."""
    pass


class EmbeddingError(BeanBaseException):
    """Raised when embedding generation fails."""
    pass


class RAGError(BeanBaseException):
    """Raised when RAG retrieval fails."""
    pass


class LLMError(BeanBaseException):
    """Raised when LLM generation fails."""
    pass


class SafetyError(BeanBaseException):
    """Raised when a safety/crisis issue is detected."""
    pass


class DatabaseError(BeanBaseException):
    """Raised when database operations fail."""
    pass


class AuthenticationError(BeanBaseException):
    """Raised when authentication fails."""
    pass
