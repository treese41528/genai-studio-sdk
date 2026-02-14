"""
GenAI Studio: A Python wrapper for Purdue's GenAI Studio API.

This module provides both a programmatic API and command-line interface for
interacting with Purdue's GenAI Studio, which runs Open WebUI backed by
LiteLLM and Ollama. It handles the quirks of this specific deployment while
presenting a clean, OpenAI-like interface.

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

Purdue's GenAI Studio Stack:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Open WebUI (Frontend)                                                  │
    │  https://genai.rcac.purdue.edu                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  LiteLLM (API Router / Proxy)                                           │
    │  - Routes requests to appropriate backend                               │
    │  - Handles API key authentication                                       │
    │  - Injects user identity into requests (causes issues with embeddings)  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Ollama (Model Backend)                                                 │
    │  - Runs the actual LLM models                                           │
    │  - Provides chat completions and embeddings                             │
    │  - Does NOT support 'user' or 'encoding_format' parameters              │
    └─────────────────────────────────────────────────────────────────────────┘

API Endpoint Structure (differs from standard OpenAI):
    - /api/models          → List models (NOT /v1/models which returns HTML)
    - /api/chat/completions → Chat completions (OpenAI-compatible)
    - /api/embeddings      → Embeddings (requires workaround, see below)

Key Workaround - Embeddings:
    The server automatically injects parameters that Ollama doesn't support:
    - 'user': Your email from API key authentication
    - 'encoding_format': 'base64' (OpenAI SDK default)
    
    We suppress these with: extra_body={"user": None, "encoding_format": None}
    Without this workaround, embeddings return a 400 error.

================================================================================
INSTALLATION
================================================================================

    pip install openai httpx numpy

================================================================================
ENVIRONMENT SETUP
================================================================================

    # Set your API key (get from GenAI Studio → Settings → Account → API Keys)
    export GENAI_STUDIO_API_KEY="your-api-key-here"

================================================================================
QUICK START - LIBRARY USAGE
================================================================================

    from genai_studio import GenAIStudio, Conversation
    
    # Initialize client
    ai = GenAIStudio()
    ai.select_model("gemma3:12b")
    
    # Simple chat
    response = ai.chat("What is statistics?")
    print(response)
    
    # Chat with system prompt
    response = ai.chat(
        "Explain p-values",
        system="You are a statistics tutor. Be concise."
    )
    
    # Streaming chat (tokens appear as generated)
    for chunk in ai.chat_stream("Tell me about regression"):
        print(chunk, end="", flush=True)
    
    # Multi-turn conversation with automatic history
    conv = Conversation(system="You are a helpful tutor.")
    conv.add_user("What is correlation?")
    response = ai.chat_conversation(conv)  # History auto-updated
    conv.add_user("How do I interpret it?")
    response = ai.chat_conversation(conv)  # Includes full context
    
    # Embeddings
    ai.select_model("llama3.2:latest")
    embedding = ai.embed("statistical significance")
    print(f"Dimension: {len(embedding)}")
    
    # Semantic similarity
    similarity = ai.similarity("mean", "average")
    print(f"Similarity: {similarity:.4f}")

    # ── RAG (Knowledge Base) ────────────────────────────────────────────
    
    # Upload a file and create a knowledge base
    file_info = ai.upload_file("lecture_notes.pdf")
    kb = ai.create_knowledge_base("STAT 350 Notes")
    ai.add_file_to_knowledge_base(kb.id, file_info.id)
    
    # Wait for indexing, then query with RAG context
    import time; time.sleep(10)
    response = ai.chat(
        "What does chapter 3 say about hypothesis testing?",
        collections=[kb.id]
    )
    print(response)
    
    # List and manage resources
    print(ai.list_knowledge_bases())
    print(ai.list_files())
    
    # Cleanup when done
    ai.delete_knowledge_base(kb.id)
================================================================================
QUICK START - CLI USAGE
================================================================================

    # List models
    python genai_studio.py models
    python genai_studio.py models --filter llama
    
    # Chat
    python genai_studio.py chat -m gemma3:12b "What is AI?"
    python genai_studio.py chat -m gemma3:12b -i              # Interactive
    python genai_studio.py chat -m gemma3:12b -i --stream     # With streaming
    
    # Embeddings
    python genai_studio.py embed -m llama3.2:latest "hello" "world" --similarity
    
    # Health check
    python genai_studio.py health
    
    # RAG workflow
    python genai_studio.py rag upload notes.pdf                Upload file
    python genai_studio.py rag create-kb "My Notes"            Create knowledge base
    python genai_studio.py rag link <kb_id> <file_id>          Link file to KB
    python genai_studio.py rag query -m gemma3:12b <kb_id> "What does it say about X?"
    python genai_studio.py rag list-kb                         List knowledge bases
    python genai_studio.py rag list-files                      List uploaded files
    python genai_studio.py rag delete-kb <kb_id>               Delete knowledge base
    python genai_studio.py rag test                            Run full lifecycle test
    
    # Chat with RAG context (alternative to rag query)
    python genai_studio.py chat -m gemma3:12b -k <kb_id> "question?"

Version: 1.2.0

================================================================================
MODULE INFORMATION
================================================================================

Author: Timothy Reese
Date: January 2026
Version: 1.2.0
"""

from __future__ import annotations

import os
import sys
import time
import json
import threading
from typing import Iterator, Callable, Any, Union, List, Dict, Optional
from dataclasses import dataclass, field

import httpx
from openai import OpenAI


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
#
# These defaults are tuned for Purdue's GenAI Studio deployment.
# The timeout values are intentionally generous to handle:
#   - Cold starts when models aren't loaded
#   - Large models (70b+) that take longer to respond
#   - Network latency to RCAC servers
#
# ════════════════════════════════════════════════════════════════════════════

__version__ = "1.2.0"

# Base URL for Purdue's GenAI Studio instance
# This runs Open WebUI with LiteLLM backend
DEFAULT_BASE_URL = "https://genai.rcac.purdue.edu"

# Total request timeout in seconds
# Increased from typical 60s to handle cold starts and large models
DEFAULT_TIMEOUT = 120

# Connection timeout in seconds
# How long to wait for initial TCP connection
DEFAULT_CONNECT_TIMEOUT = 30

# Environment variable name for API key
# Users should set this rather than hardcoding keys
ENV_API_KEY = "GENAI_STUDIO_API_KEY"


# ════════════════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ════════════════════════════════════════════════════════════════════════════
#
# We define custom exceptions to provide clear, actionable error messages.
# Each exception includes guidance on how to resolve the issue.
#
# Exception Hierarchy:
#   GenAIStudioError (base)
#   ├── AuthenticationError  - API key issues
#   ├── ModelNotFoundError   - Invalid model ID
#   ├── ConnectionError      - Network/server issues
#   └── TimeoutError         - Request took too long
#
# ════════════════════════════════════════════════════════════════════════════




class GenAIStudioError(Exception):
    """
    Base exception for all GenAI Studio errors.
    
    All custom exceptions inherit from this class, allowing users to catch
    all GenAI Studio errors with a single except clause:
    
        try:
            response = ai.chat("Hello")
        except GenAIStudioError as e:
            print(f"GenAI Studio error: {e}")
    """
    pass


class AuthenticationError(GenAIStudioError):
    """
    Raised when API key is missing or invalid.
    
    Common causes:
    - GENAI_STUDIO_API_KEY environment variable not set
    - API key is expired or revoked
    - API key was copy/pasted incorrectly
    
    Resolution:
    1. Go to GenAI Studio → Settings → Account → API Keys
    2. Generate a new key or copy existing key
    3. Set environment variable: export GENAI_STUDIO_API_KEY="your-key"
    """
    pass


class ModelNotFoundError(GenAIStudioError):
    """
    Raised when requested model is not available.
    
    Common causes:
    - Typo in model name
    - Model was removed from the server
    - Model name format changed (e.g., "gemma3:12b" vs "gemma-3-12b")
    
    Resolution:
    1. Run: ai.models to see available models
    2. Or CLI: python genai_studio.py models
    3. Use exact model ID from the list
    """
    pass


class ConnectionError(GenAIStudioError):
    """
    Raised when unable to connect to the server.
    
    Common causes:
    - Not connected to Purdue network or VPN
    - Server is down for maintenance
    - Firewall blocking the connection
    - DNS resolution failure
    
    Resolution:
    1. Check your network connection
    2. Connect to Purdue VPN if off-campus
    3. Try accessing https://genai.rcac.purdue.edu in browser
    4. Check RCAC status page for outages
    """
    pass


class TimeoutError(GenAIStudioError):
    """
    Raised when a request times out.
    
    Common causes:
    - Server is handling many requests (cold start)
    - Large model taking time to load into memory
    - Network congestion
    - Very long prompt or response
    
    Resolution:
    1. Increase timeout: GenAIStudio(timeout=180)
    2. CLI: python genai_studio.py --timeout 180 ...
    3. Try a smaller model (e.g., 7b instead of 70b)
    4. Wait and retry - server may be warming up
    """
    pass

class RAGError(GenAIStudioError):
    """
    Raised when a RAG operation (file upload, KB management) fails.
    
    Common causes:
    - File format not supported by the server
    - File too large for upload limits
    - Knowledge base ID not found or already deleted
    - File not yet finished processing/indexing
    - Server-side embedding or chunking error
    - Attempting to link a file that doesn't exist
    
    Resolution:
    1. Verify file exists and format is supported (PDF, TXT, CSV, DOCX, MD)
    2. Check that file/KB IDs are correct: ai.list_files(), ai.list_knowledge_bases()
    3. Wait longer after upload before linking (server needs processing time)
    4. Wait longer after linking before querying (indexing takes 5-30s)
    5. Check server health: ai.health_check()
    6. For persistent failures, try deleting and recreating the knowledge base
    
    Typical RAG workflow:
        file = ai.upload_file("notes.pdf")       # Step 1: Upload
        time.sleep(2)                             # Step 2: Let upload settle
        kb = ai.create_knowledge_base("My KB")    # Step 3: Create collection
        ai.add_file_to_knowledge_base(kb.id, file.id)  # Step 4: Link
        time.sleep(10)                            # Step 5: Wait for indexing
        ai.chat("question", collections=[kb.id])  # Step 6: Query
    """
    pass

# ════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════
#
# These dataclasses provide structured containers for API data.
# They offer several advantages over raw dictionaries:
#   - Type hints for IDE autocompletion
#   - Validation through type checking
#   - Convenient methods for common operations
#   - Clear documentation of available fields
#
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FileInfo:
    """
    Metadata for an uploaded file on GenAI Studio.
    
    Returned by upload_file() and list_files(). Contains the server-
    assigned file ID needed to link files to knowledge bases, along
    with the original filename and any server-provided metadata.
    
    Attributes:
    ----------
    id : str
        Server-assigned file ID (UUID). Use this to:
        - Link the file to a knowledge base: ai.add_file_to_knowledge_base(kb_id, file.id)
        - Delete the file: ai.delete_file(file.id)
        
    filename : str
        Original filename as uploaded (e.g., "lecture_notes.pdf").
        Preserved from the local file path during upload.
        
    meta : dict
        Additional metadata returned by the server. Contents vary
        by server version but may include:
        - size: File size in bytes
        - content_type: MIME type
        - created_at: Upload timestamp
        - hash: Content hash for deduplication
        
    raw_response : dict
        The complete, unprocessed JSON response from the upload or
        list API. Access this for any fields not exposed in this
        dataclass.
    
    Example:
    -------
    >>> # Upload and inspect
    >>> info = ai.upload_file("lecture_notes.pdf")
    >>> print(f"File ID: {info.id}")
    >>> print(f"Filename: {info.filename}")
    File ID: 029a1ad8-95eb-4964-b9fb-64cf980a6e02
    Filename: lecture_notes.pdf
    
    >>> # Link to a knowledge base
    >>> ai.add_file_to_knowledge_base(kb.id, info.id)
    
    >>> # List all uploaded files
    >>> for f in ai.list_files():
    ...     print(f"{f.id}  {f.filename}")
    
    >>> # Check raw server metadata
    >>> print(info.raw_response.keys())
    """
    id: str
    filename: str
    meta: dict = field(default_factory=dict)
    raw_response: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        """
        String representation for debugging.
        
        Example:
        -------
        >>> print(info)
        FileInfo(id='029a1ad8-...', filename='lecture_notes.pdf')
        """
        return f"FileInfo(id='{self.id}', filename='{self.filename}')"


@dataclass
class KnowledgeBase:
    """
    Metadata for a knowledge base (RAG collection) on GenAI Studio.
    
    A knowledge base is a named collection of files that have been
    indexed for retrieval-augmented generation. Once files are linked
    and indexed, pass the KB id to chat methods via the `collections`
    parameter to ground model responses in your documents.
    
    Returned by create_knowledge_base(), list_knowledge_bases(), and
    get_knowledge_base().
    
    Attributes:
    ----------
    id : str
        Server-assigned collection ID (UUID). This is the primary
        identifier used throughout the RAG workflow:
        - Link files: ai.add_file_to_knowledge_base(kb.id, file_id)
        - Query: ai.chat("question", collections=[kb.id])
        - Delete: ai.delete_knowledge_base(kb.id)
        
    name : str
        Human-readable name for the knowledge base (e.g., "STAT 350
        Course Notes"). Set during creation; useful for display in
        list_knowledge_bases() output.
        
    description : str
        Optional description of the knowledge base contents.
        Defaults to empty string if not provided during creation.
        
    raw_response : dict
        The complete, unprocessed JSON response from the API.
        May contain additional fields like:
        - created_at: Creation timestamp
        - updated_at: Last modification timestamp
        - files: List of linked file IDs
        - data: Server-internal metadata
    
    Example:
    -------
    >>> # Create and use
    >>> kb = ai.create_knowledge_base("STAT 350 Materials", "All course PDFs")
    >>> print(f"Collection ID: {kb.id}")
    >>> print(f"Name: {kb.name}")
    Collection ID: 207ff2b1-330c-493b-9b05-01f18aa975a3
    Name: STAT 350 Materials
    
    >>> # Link files and query
    >>> ai.add_file_to_knowledge_base(kb.id, file_info.id)
    >>> time.sleep(10)  # Wait for indexing
    >>> response = ai.chat("Summarize chapter 1", collections=[kb.id])
    
    >>> # List all knowledge bases
    >>> for kb in ai.list_knowledge_bases():
    ...     print(f"{kb.id}  {kb.name}  {kb.description}")
    
    >>> # Cleanup
    >>> ai.delete_knowledge_base(kb.id)
    """
    id: str
    name: str
    description: str = ""
    raw_response: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        """
        String representation for debugging.
        
        Example:
        -------
        >>> print(kb)
        KnowledgeBase(id='207ff2b1-...', name='STAT 350 Materials')
        """
        return f"KnowledgeBase(id='{self.id}', name='{self.name}')"


@dataclass
class ChatMessage:
    """
    A single message in a conversation.
    
    Represents one turn in a chat conversation, following the OpenAI
    message format with role and content fields.
    
    Attributes:
    ----------
    role : str
        The role of the message author. Must be one of:
        - 'system': Sets the assistant's behavior/persona
        - 'user': Message from the human user
        - 'assistant': Response from the AI model
        
    content : str
        The text content of the message.
    
    Class Methods:
    -------------
    system(content) : Create a system message
    user(content) : Create a user message  
    assistant(content) : Create an assistant message
    
    Example:
    -------
    >>> # Using constructor
    >>> msg = ChatMessage(role="user", content="Hello!")
    
    >>> # Using factory methods (preferred)
    >>> system_msg = ChatMessage.system("You are a helpful tutor.")
    >>> user_msg = ChatMessage.user("What is variance?")
    >>> assistant_msg = ChatMessage.assistant("Variance measures...")
    
    >>> # Convert to dict for API calls
    >>> msg.to_dict()
    {'role': 'user', 'content': 'Hello!'}
    """
    role: str  # 'system', 'user', or 'assistant'
    content: str
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary format for OpenAI API.
        
        Returns:
        -------
        dict
            Dictionary with 'role' and 'content' keys.
        """
        return {"role": self.role, "content": self.content}
    
    @classmethod
    def system(cls, content: str) -> ChatMessage:
        """
        Create a system message.
        
        System messages set the assistant's behavior, personality, or
        constraints. They're processed before user messages and influence
        all subsequent responses.
        
        Parameters:
        ----------
        content : str
            The system prompt text.
        
        Returns:
        -------
        ChatMessage
            A message with role='system'.
        
        Example:
        -------
        >>> msg = ChatMessage.system("You are a statistics tutor. Be concise.")
        """
        return cls(role="system", content=content)
    
    @classmethod
    def user(cls, content: str) -> ChatMessage:
        """
        Create a user message.
        
        User messages represent input from the human user.
        
        Parameters:
        ----------
        content : str
            The user's message text.
        
        Returns:
        -------
        ChatMessage
            A message with role='user'.
        
        Example:
        -------
        >>> msg = ChatMessage.user("What is a confidence interval?")
        """
        return cls(role="user", content=content)
    
    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        """
        Create an assistant message.
        
        Assistant messages represent previous responses from the AI.
        Include these in conversation history to maintain context.
        
        Parameters:
        ----------
        content : str
            The assistant's response text.
        
        Returns:
        -------
        ChatMessage
            A message with role='assistant'.
        
        Example:
        -------
        >>> msg = ChatMessage.assistant("A confidence interval is...")
        """
        return cls(role="assistant", content=content)


@dataclass
class ChatResponse:
    """
    Response from a chat completion request.
    
    Contains the assistant's response along with metadata about
    the completion, including token usage statistics.
    
    Attributes:
    ----------
    content : str
        The assistant's response text. This is the main output
        you'll typically want to display or process.
        
    model : str
        The model ID that generated this response.
        May differ slightly from requested model (e.g., version suffix).
        
    finish_reason : str or None
        Why the model stopped generating. Common values:
        - 'stop': Natural end of response
        - 'length': Hit max_tokens limit
        - 'content_filter': Content was filtered
        
    prompt_tokens : int or None
        Number of tokens in the input prompt.
        Useful for cost estimation and context length tracking.
        
    completion_tokens : int or None
        Number of tokens in the generated response.
        
    total_tokens : int or None
        Sum of prompt_tokens and completion_tokens.
        
    raw_response : Any
        The complete, unprocessed response object from the OpenAI client.
        Access this for any fields not exposed in this dataclass.
    
    Properties:
    ----------
    usage : dict
        Dictionary containing all token counts.
    
    Example:
    -------
    >>> response = ai.chat_complete("What is variance?")
    >>> print(response.content)
    "Variance measures how spread out..."
    >>> print(f"Used {response.total_tokens} tokens")
    >>> if response.finish_reason == 'length':
    ...     print("Response was truncated")
    """
    content: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw_response: Any = None
    
    @property
    def usage(self) -> dict:
        """
        Get token usage as a dictionary.
        
        Returns:
        -------
        dict
            Dictionary with prompt_tokens, completion_tokens, total_tokens.
        
        Example:
        -------
        >>> response.usage
        {'prompt_tokens': 15, 'completion_tokens': 42, 'total_tokens': 57}
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }





@dataclass
class EmbeddingResponse:
    """
    Response from an embedding request.
    
    Contains the embedding vectors along with metadata. Supports
    indexing and iteration for convenient access to embeddings.
    
    Attributes:
    ----------
    embeddings : list[list[float]]
        List of embedding vectors. Each vector is a list of floats
        representing the semantic meaning of the corresponding input text.
        Vectors from llama3.2 are typically 3072-dimensional.
        
    texts : list[str]
        The input texts that were embedded, in the same order
        as the embeddings list.
        
    model : str
        The model ID used to generate embeddings.
        
    dimension : int
        The dimensionality of each embedding vector.
        Useful for validating compatibility with vector databases.
        
    prompt_tokens : int or None
        Number of tokens processed across all input texts.
        
    raw_response : Any
        The complete response object from the OpenAI client.
    
    Methods:
    -------
    __getitem__(index) : Access embedding by index
    __len__() : Number of embeddings
    __iter__() : Iterate over embeddings
    
    Example:
    -------
    >>> response = ai.embed_complete(["cat", "dog", "banana"])
    >>> print(f"Dimension: {response.dimension}")
    Dimension: 3072
    
    >>> # Access by index
    >>> cat_embedding = response[0]
    >>> dog_embedding = response[1]
    
    >>> # Iterate
    >>> for emb in response:
    ...     print(f"Vector length: {len(emb)}")
    
    >>> # Get text-embedding pairs
    >>> for text, emb in zip(response.texts, response.embeddings):
    ...     print(f"'{text}' → {len(emb)} dims")
    """
    embeddings: list[list[float]]
    texts: list[str]
    model: str
    dimension: int
    prompt_tokens: int | None = None
    raw_response: Any = None
    
    def __getitem__(self, index: int) -> list[float]:
        """
        Access embedding by index.
        
        Parameters:
        ----------
        index : int
            Index of the embedding to retrieve.
        
        Returns:
        -------
        list[float]
            The embedding vector at the specified index.
        
        Example:
        -------
        >>> first_embedding = response[0]
        >>> last_embedding = response[-1]
        """
        return self.embeddings[index]
    
    def __len__(self) -> int:
        """
        Get the number of embeddings.
        
        Returns:
        -------
        int
            Number of embedding vectors.
        
        Example:
        -------
        >>> len(response)
        3
        """
        return len(self.embeddings)
    
    def __iter__(self):
        """
        Iterate over embeddings.
        
        Yields:
        ------
        list[float]
            Each embedding vector in order.
        
        Example:
        -------
        >>> for embedding in response:
        ...     process(embedding)
        """
        return iter(self.embeddings)


@dataclass
class Conversation:
    """
    Manages a multi-turn conversation with automatic history tracking.
    
    This class simplifies multi-turn conversations by maintaining the
    message history and providing convenient methods for adding messages.
    When used with chat_conversation(), the assistant's responses are
    automatically appended to the history.
    
    Attributes:
    ----------
    system : str or None
        Optional system prompt that sets the assistant's behavior.
        This is prepended to messages when converting to API format.
        
    messages : list[ChatMessage]
        List of conversation messages (user and assistant turns).
        Does NOT include the system message.
    
    Methods:
    -------
    add_user(content) : Add a user message (chainable)
    add_assistant(content) : Add an assistant message (chainable)
    clear() : Clear message history (keeps system prompt)
    to_messages() : Convert to OpenAI API format
    
    Example:
    -------
    >>> # Create conversation with system prompt
    >>> conv = Conversation(system="You are a statistics tutor.")
    
    >>> # Add user message and get response
    >>> conv.add_user("What is correlation?")
    >>> response = ai.chat_conversation(conv)
    >>> # conv now contains both user message and assistant response
    
    >>> # Continue the conversation
    >>> conv.add_user("How do I interpret the value?")
    >>> response = ai.chat_conversation(conv)
    >>> # Full history is maintained
    
    >>> # Check history length
    >>> print(f"Conversation has {len(conv)} messages")
    
    >>> # Clear and start over (keeps system prompt)
    >>> conv.clear()
    
    >>> # Method chaining
    >>> conv.add_user("Question 1").add_assistant("Answer 1").add_user("Follow up")
    """
    system: str | None = None
    messages: list[ChatMessage] = field(default_factory=list)
    
    def __post_init__(self):
        """
        Post-initialization processing.
        
        Note: System prompt is stored separately and not added to messages.
        This keeps messages as a clean list of just user/assistant turns.
        """
        # System prompt handled separately in to_messages()
        pass
    
    def add_user(self, content: str) -> Conversation:
        """
        Add a user message to the conversation.
        
        Parameters:
        ----------
        content : str
            The user's message text.
        
        Returns:
        -------
        Conversation
            Returns self for method chaining.
        
        Example:
        -------
        >>> conv.add_user("What is the Central Limit Theorem?")
        
        >>> # Method chaining
        >>> conv.add_user("First question").add_user("Second question")
        """
        self.messages.append(ChatMessage.user(content))
        return self  # Enable method chaining
    
    def add_assistant(self, content: str) -> Conversation:
        """
        Add an assistant message to the conversation.
        
        Typically called automatically by chat_conversation() when
        auto_update=True (the default). Call manually when using
        streaming or when you need to inject assistant messages.
        
        Parameters:
        ----------
        content : str
            The assistant's response text.
        
        Returns:
        -------
        Conversation
            Returns self for method chaining.
        
        Example:
        -------
        >>> # Manual addition (usually automatic)
        >>> conv.add_assistant("The CLT states that...")
        
        >>> # Useful for injecting example responses
        >>> conv.add_user("What is 2+2?")
        >>> conv.add_assistant("4")
        >>> conv.add_user("What is 3+3?")
        >>> # Model will follow the established pattern
        """
        self.messages.append(ChatMessage.assistant(content))
        return self  # Enable method chaining
    
    def clear(self) -> Conversation:
        """
        Clear conversation history while keeping the system prompt.
        
        Use this to start a fresh conversation without creating
        a new Conversation object.
        
        Returns:
        -------
        Conversation
            Returns self for method chaining.
        
        Example:
        -------
        >>> conv.clear()
        >>> # System prompt is preserved, messages are cleared
        >>> conv.add_user("New conversation starts here")
        """
        self.messages = []
        return self  # Enable method chaining
    
    def to_messages(self) -> list[dict]:
        """
        Convert conversation to OpenAI API message format.
        
        Returns a list of message dictionaries suitable for passing
        to the chat completions API. System prompt (if set) is
        prepended to the message list.
        
        Returns:
        -------
        list[dict]
            List of message dicts with 'role' and 'content' keys.
        
        Example:
        -------
        >>> conv = Conversation(system="Be helpful")
        >>> conv.add_user("Hello")
        >>> conv.to_messages()
        [
            {'role': 'system', 'content': 'Be helpful'},
            {'role': 'user', 'content': 'Hello'}
        ]
        """
        result = []
        
        # Add system message first if present
        if self.system:
            result.append({"role": "system", "content": self.system})
        
        # Add all conversation messages
        result.extend(m.to_dict() for m in self.messages)
        
        return result
    
    def __len__(self) -> int:
        """
        Get the number of messages (excluding system prompt).
        
        Returns:
        -------
        int
            Number of user and assistant messages.
        """
        return len(self.messages)


# ════════════════════════════════════════════════════════════════════════════
# PROGRESS INDICATOR
# ════════════════════════════════════════════════════════════════════════════
#
# The Spinner class provides visual feedback during long-running operations.
# This is especially important for:
#   - Model list fetching (can take 5-30s on cold start)
#   - Chat completions with large models
#   - Embedding generation for many texts
#
# The spinner runs in a background thread and automatically cleans up
# the terminal line when stopped.
#
# ════════════════════════════════════════════════════════════════════════════

class Spinner:
    """
    A terminal spinner for visual feedback during long operations.
    
    Displays an animated spinner character with a message while
    a long-running operation is in progress. Automatically detects
    if output is a TTY (terminal) and disables itself when piped
    or redirected.
    
    The spinner runs in a background thread, so it won't block
    the main operation.
    
    Attributes:
    ----------
    FRAMES : list[str]
        Animation frames using Unicode braille characters.
        These provide smooth animation across most terminals.
    
    Parameters:
    ----------
    message : str
        Text to display next to the spinner.
    stream : file
        Output stream (default: sys.stderr).
    enabled : bool
        Whether to show the spinner (auto-disabled for non-TTY).
    
    Usage:
    -----
    # As context manager (recommended)
    with Spinner("Loading models..."):
        models = fetch_models()
    
    # Manual control
    spinner = Spinner("Processing...")
    spinner.start()
    try:
        do_long_operation()
    finally:
        spinner.stop()
    
    Example:
    -------
    >>> with Spinner("Fetching models"):
    ...     time.sleep(2)  # Spinner animates during this time
    >>> # Spinner automatically clears when done
    """
    
    # Unicode braille characters for smooth animation
    # These are widely supported and look good in most terminals
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(
        self,
        message: str = "",
        stream=None,
        enabled: bool = True
    ):
        """
        Initialize the spinner.
        
        Parameters:
        ----------
        message : str, optional
            Text to display next to the spinner.
            
        stream : file, optional
            Output stream for the spinner. Defaults to sys.stderr
            so it doesn't interfere with stdout content.
            
        enabled : bool, optional
            Whether to enable the spinner. Automatically disabled
            if the stream is not a TTY (e.g., when piped).
        """
        self.message = message
        self.stream = stream or sys.stderr
        
        # Only enable if explicitly enabled AND output is a terminal
        # This prevents garbled output when piped to files or other commands
        self.enabled = enabled and self.stream.isatty()
        
        # Threading primitives for clean start/stop
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
    
    def _spin(self):
        """
        Animation loop (runs in background thread).
        
        Cycles through spinner frames at 10 FPS until stop_event is set.
        Uses carriage return (\\r) to overwrite the same line.
        """
        frame_idx = 0
        while not self._stop_event.is_set():
            # Get current frame character
            frame = self.FRAMES[frame_idx % len(self.FRAMES)]
            
            # Write spinner and message, overwriting previous line
            self.stream.write(f"\r{frame} {self.message}")
            self.stream.flush()
            
            frame_idx += 1
            
            # Wait 100ms or until stop is requested
            # Using wait() allows immediate response to stop
            self._stop_event.wait(0.1)
        
        # Clear the spinner line when done
        # Write spaces to overwrite, then return to line start
        self.stream.write("\r" + " " * (len(self.message) + 3) + "\r")
        self.stream.flush()
    
    def start(self):
        """
        Start the spinner animation.
        
        Spawns a daemon thread to run the animation. The daemon flag
        ensures the thread won't prevent program exit.
        """
        if self.enabled:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
    
    def stop(self):
        """
        Stop the spinner animation.
        
        Signals the animation thread to stop and waits for it to
        finish cleaning up the terminal line.
        """
        if self._thread:
            self._stop_event.set()
            self._thread.join()  # Wait for thread to finish
            self._thread = None
    
    def __enter__(self):
        """Context manager entry - start spinner."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Context manager exit - stop spinner."""
        self.stop()


# ════════════════════════════════════════════════════════════════════════════
# MAIN CLIENT CLASS
# ════════════════════════════════════════════════════════════════════════════
#
# GenAIStudio is the primary interface for interacting with the API.
# It wraps the OpenAI Python SDK and adds:
#   - Automatic handling of GenAI Studio's API quirks
#   - Model validation and caching
#   - Convenience methods for common operations
#   - Streaming support
#   - Conversation management
#   - Callbacks for UI integration
#
# ════════════════════════════════════════════════════════════════════════════

class GenAIStudio:
    """
    Client for Purdue GenAI Studio API.
    
    Provides a high-level interface for:
    - Listing and selecting models
    - Chat completions (single-turn, multi-turn, streaming)
    - Text embeddings with semantic similarity
    - Health checking and connection management
    
    Architecture Notes:
    ------------------
    Purdue's GenAI Studio uses Open WebUI with LiteLLM as the backend router.
    The API structure differs from standard OpenAI in several ways:
    
    1. Models endpoint: /api/models (NOT /v1/models which returns HTML)
       We use httpx directly for this since OpenAI client expects different format.
    
    2. Chat endpoint: /api/chat/completions
       This is OpenAI-compatible, so we use the OpenAI client.
    
    3. Embeddings endpoint: /api/embeddings
       Requires workaround: extra_body={"user": None, "encoding_format": None}
       The server injects 'user' (your email) and 'encoding_format' params,
       but Ollama (the backend) doesn't support these, causing 400 errors.
    
    Thread Safety:
    -------------
    The client is generally thread-safe for concurrent requests, but
    model selection (select_model) should be done before spawning threads
    or use the model= parameter in each call.
    
    Attributes:
    ----------
    api_key : str
        Your GenAI Studio API key.
        
    base_url : str
        Base URL for the API (default: https://genai.rcac.purdue.edu).
        
    timeout : int
        Total request timeout in seconds.
        
    connect_timeout : int
        Connection establishment timeout in seconds.
        
    validate_model : bool
        Whether to validate model exists when selecting.
        
    client : OpenAI
        The underlying OpenAI client instance.
        
    on_request_start : Callable or None
        Callback invoked when a request begins.
        
    on_request_end : Callable or None
        Callback invoked when a request completes.
    
    Example:
    -------
    >>> # Basic usage
    >>> ai = GenAIStudio()
    >>> ai.select_model("gemma3:12b")
    >>> response = ai.chat("Explain statistics")
    >>> print(response)
    
    >>> # With custom timeout for slow connections
    >>> ai = GenAIStudio(timeout=180)
    
    >>> # Skip model validation for faster startup
    >>> ai = GenAIStudio(validate_model=False)
    >>> ai.select_model("gemma3:12b")  # Won't fetch model list
    
    >>> # With UI callbacks
    >>> ai = GenAIStudio()
    >>> ai.on_request_start = lambda op: show_loading()
    >>> ai.on_request_end = lambda op: hide_loading()
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
        validate_model: bool = True
    ):
        """
        Initialize the GenAI Studio client.
        
        Parameters:
        ----------
        api_key : str, optional
            Your GenAI Studio API key. If not provided, reads from
            the GENAI_STUDIO_API_KEY environment variable.
            
            Get your key from:
            GenAI Studio → Settings → Account → API Keys
            
        base_url : str, optional
            Base URL for the GenAI Studio instance.
            Default: "https://genai.rcac.purdue.edu"
            
            Change this if you're connecting to a different
            Open WebUI / LiteLLM deployment.
            
        timeout : int, optional
            Total request timeout in seconds. This covers the entire
            request lifecycle including connection, sending, and
            receiving the response.
            Default: 120 (generous for cold starts)
            
            Increase this if you experience timeouts with large models
            or slow network connections.
            
        connect_timeout : int, optional
            Connection establishment timeout in seconds. This is how
            long to wait for the initial TCP connection to be established.
            Default: 30
            
            If you're getting connection timeouts specifically, increase
            this value.
            
        validate_model : bool, optional
            If True, validate that the model exists when calling
            select_model(). This requires fetching the model list
            from the server.
            Default: True
            
            Set to False for faster startup when you know the model
            name is correct. This skips the model list API call.
        
        Raises:
        ------
        AuthenticationError
            If no API key is provided and GENAI_STUDIO_API_KEY
            environment variable is not set.
        
        Example:
        -------
        >>> # Using environment variable (recommended)
        >>> # First: export GENAI_STUDIO_API_KEY="your-key"
        >>> ai = GenAIStudio()
        
        >>> # Explicit API key (not recommended for shared code)
        >>> ai = GenAIStudio(api_key="sk-...")
        
        >>> # Custom timeout for slow connections
        >>> ai = GenAIStudio(timeout=180, connect_timeout=60)
        
        >>> # Fast startup without model validation
        >>> ai = GenAIStudio(validate_model=False)
        """
        # ════════════════════════════════════════════════════════════════
        # API KEY RESOLUTION
        # ════════════════════════════════════════════════════════════════
        # Priority: explicit parameter > environment variable
        # Using environment variables is preferred for security
        
        self.api_key = api_key or os.environ.get(ENV_API_KEY)
        
        if not self.api_key:
            raise AuthenticationError(
                f"API key required. Either:\n"
                f"  1. Set {ENV_API_KEY} environment variable\n"
                f"  2. Pass api_key parameter to GenAIStudio()\n\n"
                f"Get your key from: GenAI Studio → Settings → Account → API Keys"
            )
        
        # ════════════════════════════════════════════════════════════════
        # CONFIGURATION
        # ════════════════════════════════════════════════════════════════
        
        # Strip trailing slash to prevent double-slash in URL construction
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.validate_model = validate_model
        
        # ════════════════════════════════════════════════════════════════
        # HTTP CLIENT SETUP
        # ════════════════════════════════════════════════════════════════
        # Configure httpx timeout object for direct HTTP requests
        # (used for model listing since OpenAI client can't parse the response)
        
        self._http_timeout = httpx.Timeout(
            timeout=timeout,      # Total timeout
            connect=connect_timeout  # Connection timeout specifically
        )
        
        # ════════════════════════════════════════════════════════════════
        # OPENAI CLIENT SETUP
        # ════════════════════════════════════════════════════════════════
        # We use the official OpenAI Python SDK, pointing it to GenAI Studio.
        # The /api prefix is required (not /v1) for Open WebUI compatibility.
        #
        # Why OpenAI SDK?
        # - Handles request formatting, retries, and error parsing
        # - Provides streaming support
        # - Well-tested and maintained
        # - Familiar API for users
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.base_url}/api",
            timeout=timeout
        )
        
        # ════════════════════════════════════════════════════════════════
        # INTERNAL STATE
        # ════════════════════════════════════════════════════════════════
        
        # Currently selected model (None until select_model() is called)
        self._model: str | None = None
        
        # Cached model list (populated on first access to .models property)
        # Caching avoids repeated API calls when checking model list
        self._available_models: list[str] | None = None
        
        # ════════════════════════════════════════════════════════════════
        # UI INTEGRATION CALLBACKS
        # ════════════════════════════════════════════════════════════════
        # These callbacks allow UI code to show loading indicators,
        # update progress, or log requests.
        #
        # The callback receives the operation name as a string:
        # - "chat" for chat completions
        # - "chat_stream" for streaming chat
        # - "chat_messages" for multi-turn chat
        # - "embed" for embeddings
        
        self.on_request_start: Callable[[str], None] | None = None
        self.on_request_end: Callable[[str], None] | None = None
    
    # ════════════════════════════════════════════════════════════════════
    # INTERNAL HTTP HELPERS
    # ════════════════════════════════════════════════════════════════════
    
    def _http_headers(self, json_content: bool = True) -> dict:
        """Build headers for direct HTTP requests (non-OpenAI endpoints)."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers
    
    def _http_get(self, path: str) -> httpx.Response:
        with httpx.Client(timeout=self._http_timeout) as http:
            resp = http.get(f"{self.base_url}{path}",
                           headers=self._http_headers(json_content=False))
        resp.raise_for_status()
        return resp
    
    def _http_post(self, path: str, **kwargs) -> httpx.Response:
        headers = kwargs.pop("headers", None)
        if headers is None:
            if "files" not in kwargs:
                headers = self._http_headers(json_content=True)
            else:
                headers = self._http_headers(json_content=False)
        with httpx.Client(timeout=self._http_timeout) as http:
            resp = http.post(f"{self.base_url}{path}", headers=headers, **kwargs)
        resp.raise_for_status()
        return resp
    
    def _http_delete(self, path: str) -> httpx.Response:
        with httpx.Client(timeout=self._http_timeout) as http:
            resp = http.delete(f"{self.base_url}{path}",
                              headers=self._http_headers(json_content=False))
        resp.raise_for_status()
        return resp
    
    # ════════════════════════════════════════════════════════════════════
    # RAG HELPER
    # ════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _build_rag_extra_body(collections=None, extra_body=None):
        """Merge collection IDs into extra_body for RAG chat requests."""
        if not collections and not extra_body:
            return None
        result = dict(extra_body) if extra_body else {}
        if collections:
            result["files"] = [{"type": "collection", "id": cid} for cid in collections]
        return result or None
    
    # ════════════════════════════════════════════════════════════════════
    # RAG: FILE MANAGEMENT
    # ════════════════════════════════════════════════════════════════════
    
    def upload_file(self, filepath: str) -> FileInfo:
        """Upload a file to GenAI Studio. Returns FileInfo with server-assigned ID."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "rb") as f:
                resp = self._http_post("/api/v1/files/", files={"file": (filename, f)})
            data = resp.json()
            return FileInfo(id=data["id"], filename=data.get("filename", filename),
                           meta=data.get("meta", {}), raw_response=data)
        except httpx.HTTPStatusError as e:
            raise RAGError(f"Upload failed ({e.response.status_code}): {e.response.text}")
    
    def list_files(self) -> list[FileInfo]:
        """List all uploaded files."""
        try:
            resp = self._http_get("/api/v1/files/")
            data = resp.json()
            items = data if isinstance(data, list) else data.get("data", data.get("files", []))
            return [FileInfo(id=i["id"], filename=i.get("filename", i.get("name", "unknown")),
                            meta=i.get("meta", {}), raw_response=i) for i in items]
        except httpx.HTTPStatusError as e:
            raise RAGError(f"List files failed ({e.response.status_code}): {e.response.text}")
    
    def delete_file(self, file_id: str) -> bool:
        """Delete an uploaded file."""
        try:
            self._http_delete(f"/api/v1/files/{file_id}")
            return True
        except httpx.HTTPStatusError as e:
            raise RAGError(f"Delete file failed ({e.response.status_code}): {e.response.text}")
    
    # ════════════════════════════════════════════════════════════════════
    # RAG: KNOWLEDGE BASE MANAGEMENT
    # ════════════════════════════════════════════════════════════════════
    
    def create_knowledge_base(self, name: str, description: str = "") -> KnowledgeBase:
        """Create a new knowledge base (RAG collection)."""
        try:
            resp = self._http_post("/api/v1/knowledge/create",
                                   json={"name": name, "description": description, "data": {}})
            data = resp.json()
            return KnowledgeBase(id=data["id"], name=data.get("name", name),
                                description=data.get("description", description), raw_response=data)
        except httpx.HTTPStatusError as e:
            raise RAGError(f"Create KB failed ({e.response.status_code}): {e.response.text}")
    
    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        """List all knowledge bases."""
        try:
            resp = self._http_get("/api/v1/knowledge/")
            data = resp.json()
            items = data if isinstance(data, list) else data.get("data", data.get("knowledge", []))
            return [KnowledgeBase(id=i["id"], name=i.get("name", "untitled"),
                                  description=i.get("description", ""), raw_response=i) for i in items]
        except httpx.HTTPStatusError as e:
            raise RAGError(f"List KBs failed ({e.response.status_code}): {e.response.text}")
    
    def get_knowledge_base(self, kb_id: str) -> KnowledgeBase:
        """Get details for a specific knowledge base."""
        try:
            resp = self._http_get(f"/api/v1/knowledge/{kb_id}")
            data = resp.json()
            return KnowledgeBase(id=data["id"], name=data.get("name", "untitled"),
                                description=data.get("description", ""), raw_response=data)
        except httpx.HTTPStatusError as e:
            raise RAGError(f"Get KB failed ({e.response.status_code}): {e.response.text}")
    
    def add_file_to_knowledge_base(self, kb_id: str, file_id: str) -> bool:
        """Link an uploaded file to a knowledge base for indexing."""
        try:
            self._http_post(f"/api/v1/knowledge/{kb_id}/file/add", json={"file_id": file_id})
            return True
        except httpx.HTTPStatusError as e:
            raise RAGError(f"Link failed ({e.response.status_code}): {e.response.text}")
    
    def remove_file_from_knowledge_base(self, kb_id: str, file_id: str) -> bool:
        """Unlink a file from a knowledge base."""
        try:
            self._http_post(f"/api/v1/knowledge/{kb_id}/file/remove", json={"file_id": file_id})
            return True
        except httpx.HTTPStatusError as e:
            raise RAGError(f"Unlink failed ({e.response.status_code}): {e.response.text}")
    
    def delete_knowledge_base(self, kb_id: str) -> bool:
        """Delete a knowledge base (files are NOT deleted)."""
        try:
            self._http_delete(f"/api/v1/knowledge/{kb_id}/delete")
            return True
        except httpx.HTTPStatusError as e:
            raise RAGError(f"Delete KB failed ({e.response.status_code}): {e.response.text}")
    # ════════════════════════════════════════════════════════════════════
    # MODEL MANAGEMENT
    # ════════════════════════════════════════════════════════════════════
    #
    # Methods for listing, selecting, and validating models.
    #
    # The model list is cached after first fetch to avoid repeated API calls.
    # Use refresh_models() to force a fresh fetch.
    #
    # ════════════════════════════════════════════════════════════════════
    
    @property
    def models(self) -> list[str]:
        """
        List available model IDs (cached).
        
        Returns a sorted list of model identifier strings that can be
        used with select_model(), chat(), and embed() methods.
        
        The list is cached after the first API call. Use refresh_models()
        to force a fresh fetch from the server.
        
        Returns:
        -------
        list[str]
            Sorted list of available model IDs.
        
        Raises:
        ------
        ConnectionError
            If unable to connect to the server.
            Check network connection and VPN status.
            
        TimeoutError
            If the request times out.
            Server may be starting up; try again or increase timeout.
            
        AuthenticationError
            If the API key is invalid.
        
        Note:
        ----
        We use httpx directly here instead of the OpenAI client because
        Open WebUI's /api/models endpoint returns a non-standard format
        that the OpenAI client's models.list() cannot parse.
        
        The endpoint returns: {"data": [{"id": "model-name", ...}, ...]}
        OpenAI SDK expects: {"data": [{"id": "model-name"}, ...]} with
        different nested structure.
        
        Example:
        -------
        >>> ai = GenAIStudio()
        >>> print(ai.models)
        ['deepseek-r1:1.5b', 'gemma3:12b', 'llama3.2:latest', ...]
        
        >>> # Filter for specific models
        >>> llama_models = [m for m in ai.models if 'llama' in m]
        >>> print(llama_models)
        ['llama3.1:70b-instruct-q4_K_M', 'llama3.1:latest', 'llama3.2:latest']
        
        >>> # Check if a model is available
        >>> if "gemma3:12b" in ai.models:
        ...     ai.select_model("gemma3:12b")
        """
        if self._available_models is None:
            self._fetch_models()
        return self._available_models
    
    def _fetch_models(self) -> None:
        """
        Fetch model list from the server.
        
        Internal method that makes the HTTP request to get available models.
        Called automatically when accessing the models property.
        
        Raises:
        ------
        TimeoutError
            If connection or read times out.
        ConnectionError
            If unable to establish connection.
        AuthenticationError
            If API key is invalid (401 response).
        GenAIStudioError
            For other HTTP errors.
        """
        try:
            # Use httpx directly because OpenAI client can't parse
            # Open WebUI's non-standard model list format
            with httpx.Client(timeout=self._http_timeout) as http:
                response = http.get(
                    f"{self.base_url}/api/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
            
            # Raise exception for HTTP errors (4xx, 5xx)
            response.raise_for_status()
            
            # Parse response and extract model IDs
            # Response format: {"data": [{"id": "model-name", ...}, ...]}
            data = response.json()
            self._available_models = sorted([m["id"] for m in data["data"]])
            
        except httpx.ConnectTimeout:
            # Connection establishment timed out
            raise TimeoutError(
                f"Connection timed out after {self.connect_timeout}s.\n"
                f"The server may be starting up (cold start). Try:\n"
                f"  - Waiting a moment and retrying\n"
                f"  - Increasing timeout: GenAIStudio(timeout=180)\n"
                f"  - CLI: python genai_studio.py --timeout 180 ..."
            )
        except httpx.ReadTimeout:
            # Connection established but response took too long
            raise TimeoutError(
                f"Read timed out after {self.timeout}s.\n"
                f"The server is responding slowly. Try:\n"
                f"  - Increasing timeout: GenAIStudio(timeout=180)\n"
                f"  - Checking if the server is under heavy load"
            )
        except httpx.ConnectError as e:
            # Could not establish connection at all
            raise ConnectionError(
                f"Could not connect to {self.base_url}\n"
                f"Possible causes:\n"
                f"  - Not connected to Purdue network or VPN\n"
                f"  - Server is down for maintenance\n"
                f"  - Firewall blocking the connection\n"
                f"Details: {e}"
            )
        except httpx.HTTPStatusError as e:
            # HTTP error response (4xx, 5xx)
            if e.response.status_code == 401:
                raise AuthenticationError(
                    "Invalid API key.\n"
                    "Please check your GENAI_STUDIO_API_KEY or generate a new one at:\n"
                    "GenAI Studio → Settings → Account → API Keys"
                )
            raise GenAIStudioError(f"HTTP error {e.response.status_code}: {e}")
    
    def refresh_models(self) -> list[str]:
        """
        Force refresh the cached model list.
        
        Use this if models have been added or removed on the server
        since the client was initialized, or if you want to verify
        current availability.
        
        Returns:
        -------
        list[str]
            Fresh list of available model IDs.
        
        Example:
        -------
        >>> ai = GenAIStudio()
        >>> ai.models  # Initial fetch
        ['model-a', 'model-b']
        
        >>> # ... admin adds new models on server ...
        
        >>> ai.refresh_models()  # Get updated list
        ['model-a', 'model-b', 'model-c']
        """
        self._available_models = None  # Clear cache
        return self.models  # Triggers fresh fetch
    
    @property
    def model(self) -> str | None:
        """
        Get the currently selected model ID.
        
        Returns:
        -------
        str or None
            The currently selected model ID, or None if no model is selected.
        
        Example:
        -------
        >>> ai = GenAIStudio()
        >>> ai.model  # None - no model selected yet
        None
        
        >>> ai.select_model("gemma3:12b")
        >>> ai.model
        'gemma3:12b'
        """
        return self._model
    
    @model.setter
    def model(self, model_id: str) -> None:
        """
        Set the active model with optional validation.
        
        Parameters:
        ----------
        model_id : str
            The model ID to select.
        
        Raises:
        ------
        ModelNotFoundError
            If validate_model=True and the model_id is not in the
            available models list.
        
        Example:
        -------
        >>> ai = GenAIStudio()
        >>> ai.model = "gemma3:12b"  # Valid
        
        >>> ai.model = "nonexistent"  # Raises ModelNotFoundError
        """
        # Only validate if validation is enabled
        # This allows fast startup when you know the model name is correct
        if self.validate_model and model_id not in self.models:
            raise ModelNotFoundError(
                f"Model '{model_id}' not found.\n"
                f"Available models ({len(self.models)}): "
                f"{', '.join(self.models[:5])}..."
            )
        self._model = model_id
    
    def select_model(self, model_id: str) -> GenAIStudio:
        """
        Select a model (fluent interface).
        
        This method allows method chaining for concise code.
        It's equivalent to setting ai.model = model_id.
        
        Parameters:
        ----------
        model_id : str
            The model ID to select. Must be in the available models
            list (unless validate_model=False).
        
        Returns:
        -------
        GenAIStudio
            Returns self for method chaining.
        
        Raises:
        ------
        ModelNotFoundError
            If validate_model=True and the model is not found.
        
        Example:
        -------
        >>> ai = GenAIStudio()
        
        >>> # Method chaining
        >>> response = ai.select_model("gemma3:12b").chat("Hello!")
        
        >>> # Equivalent to:
        >>> ai.select_model("gemma3:12b")
        >>> response = ai.chat("Hello!")
        
        >>> # Check available models first
        >>> if "gemma3:12b" in ai.models:
        ...     ai.select_model("gemma3:12b")
        """
        self.model = model_id  # Uses the setter with validation
        return self  # Enable method chaining
    
    def _resolve_model(self, model: str | None) -> str:
        """
        Resolve model from parameter or instance default.
        
        Internal helper that determines which model to use for a request.
        Priority: explicit parameter > selected model.
        
        Parameters:
        ----------
        model : str or None
            Explicitly provided model, or None to use default.
        
        Returns:
        -------
        str
            The resolved model ID.
        
        Raises:
        ------
        ValueError
            If no model is selected and none is provided.
        """
        model = model or self._model
        if not model:
            raise ValueError(
                "No model selected. Either:\n"
                "  1. Call ai.select_model('model-name') first\n"
                "  2. Pass model='model-name' to this method"
            )
        return model
    
    # ════════════════════════════════════════════════════════════════════
    # CHAT COMPLETIONS
    # ════════════════════════════════════════════════════════════════════
    #
    # Methods for generating chat responses from the model.
    #
    # Three main patterns:
    # 1. Single-turn: chat() / chat_complete() for one-off questions
    # 2. Multi-turn: chat_messages() / chat_conversation() for conversations
    # 3. Streaming: chat_stream() for real-time token output
    #
    # All methods support these common **kwargs parameters:
    # - temperature (float): Randomness, 0.0-2.0, default ~0.7
    # - max_tokens (int): Maximum response length
    # - top_p (float): Nucleus sampling, 0.0-1.0
    # - stop (str or list): Stop sequences
    # - presence_penalty (float): Penalize repeated topics, -2.0 to 2.0
    # - frequency_penalty (float): Penalize repeated tokens, -2.0 to 2.0
    #
    # ════════════════════════════════════════════════════════════════════
    
    def chat(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        collections: list[str] | None = None,
        **kwargs
    ) -> str:
        """
        Send a chat completion request and get the response text.
        
        This is the simplest method for single-turn conversations.
        For multi-turn conversations, use chat_conversation() or chat_messages().
        For streaming output, use chat_stream().
        For full response metadata, use chat_complete().
        
        Parameters:
        ----------
        prompt : str
            The user's message or question.
            
        model : str, optional
            Model ID to use. If not provided, uses the currently
            selected model (set via select_model()).
            
        system : str, optional
            System prompt to set the assistant's behavior or persona.
            This is prepended to the conversation and influences all responses.
            
            Examples:
            - "You are a statistics tutor. Be concise and use examples."
            - "Respond only in bullet points."
            - "You are a helpful assistant that explains things simply."
            
        **kwargs : dict
            Additional parameters passed to the OpenAI API:
            
            temperature : float, optional
                Controls randomness in responses.
                - 0.0: Deterministic, always picks most likely token
                - 0.7: Balanced creativity (default)
                - 1.0: More creative/varied
                - 2.0: Maximum randomness
                
            max_tokens : int, optional
                Maximum number of tokens in the response.
                If not set, the model decides when to stop.
                Useful for limiting response length.
                
            top_p : float, optional
                Nucleus sampling parameter (0.0 to 1.0).
                Alternative to temperature for controlling randomness.
                Usually use one or the other, not both.
                
            stop : str or list[str], optional
                Stop sequence(s). Generation stops when these are produced.
                Example: stop=["\\n\\n", "END"]
                
            presence_penalty : float, optional
                Penalize tokens based on whether they've appeared at all.
                Range: -2.0 to 2.0. Positive values encourage new topics.
                
            frequency_penalty : float, optional
                Penalize tokens based on how often they've appeared.
                Range: -2.0 to 2.0. Positive values reduce repetition.
        
        Returns:
        -------
        str
            The assistant's response text.
        
        Raises:
        ------
        ValueError
            If no model is selected and none is provided.
        GenAIStudioError
            If the API request fails.
        
        Example:
        -------
        >>> ai = GenAIStudio()
        >>> ai.select_model("gemma3:12b")
        
        >>> # Simple query
        >>> response = ai.chat("What is a confidence interval?")
        >>> print(response)
        
        >>> # With system prompt
        >>> response = ai.chat(
        ...     "Explain regression",
        ...     system="You are a statistics professor. Use examples."
        ... )
        
        >>> # With parameters
        >>> response = ai.chat(
        ...     "Write a haiku about data",
        ...     temperature=0.9,
        ...     max_tokens=50
        ... )
        
        >>> # Override model for single call
        >>> response = ai.chat("Hello", model="llama3.2:latest")
        """
        extra_body = self._build_rag_extra_body(
            collections=collections,
            extra_body=kwargs.pop("extra_body", None)
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        # Use chat_complete internally, extract just the content string
        response = self.chat_complete(prompt, model=model, system=system, **kwargs)
        return response.content
    
    def chat_complete(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        collections: list[str] | None = None,  # ← ADD
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat completion request with full response details.
        
        Similar to chat(), but returns a ChatResponse object with
        additional metadata like token usage and finish reason.
        
        Parameters:
        ----------
        prompt : str
            The user's message or question.
            
        model : str, optional
            Model ID to use. Falls back to selected model.
            
        system : str, optional
            System prompt to set assistant behavior.
            
        **kwargs : dict
            Additional parameters (temperature, max_tokens, etc.)
            See chat() docstring for full list.
        
        Returns:
        -------
        ChatResponse
            Full response with:
            - content: The response text
            - model: Model that generated the response
            - finish_reason: Why generation stopped
            - prompt_tokens: Input token count
            - completion_tokens: Output token count
            - total_tokens: Total tokens used
            - raw_response: Original API response object
        
        Example:
        -------
        >>> response = ai.chat_complete("What is variance?")
        >>> print(response.content)
        "Variance measures how spread out..."
        
        >>> print(f"Used {response.total_tokens} tokens")
        Used 47 tokens
        
        >>> if response.finish_reason == 'length':
        ...     print("Response was truncated - consider increasing max_tokens")
        
        >>> # Access raw response for any additional fields
        >>> print(response.raw_response.id)
        """
        # ── Model Resolution ────────────────────────────────────────────
        # Priority: explicit parameter > selected model
        model = self._resolve_model(model)
        
        # ── Build Messages Array ────────────────────────────────────────
        # OpenAI chat format: list of {"role": ..., "content": ...} dicts
        messages = []
        
        # System message sets the assistant's behavior (optional but recommended)
        if system:
            messages.append({"role": "system", "content": system})
        
        # User message is the actual prompt
        messages.append({"role": "user", "content": prompt})
        
        extra_body = self._build_rag_extra_body(
            collections=collections,
            extra_body=kwargs.pop("extra_body", None)
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        
        # ── Invoke Request Callbacks ────────────────────────────────────
        # These allow UI code to show loading indicators
        if self.on_request_start:
            self.on_request_start("chat")
        
        # ── API Call ────────────────────────────────────────────────────
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs  # Pass through temperature, max_tokens, etc.
            )
        finally:
            # Always call end callback, even on error
            if self.on_request_end:
                self.on_request_end("chat")
        
        # ── Build Response Object ───────────────────────────────────────
        # Extract relevant fields into our ChatResponse dataclass
        return ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            # Usage may not always be present (e.g., some backends don't report it)
            prompt_tokens=getattr(response.usage, 'prompt_tokens', None),
            completion_tokens=getattr(response.usage, 'completion_tokens', None),
            total_tokens=getattr(response.usage, 'total_tokens', None),
            raw_response=response  # Keep original for advanced use
        )
    
    def chat_stream(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        collections: list[str] | None = None,  # ← ADD
        **kwargs
    ) -> Iterator[str]:
        """
        Stream a chat completion, yielding tokens as they arrive.
        
        Streaming provides a better user experience for long responses
        by showing text as it's generated rather than waiting for the
        complete response.
        
        Parameters:
        ----------
        prompt : str
            The user's message or question.
            
        model : str, optional
            Model ID to use. Falls back to selected model.
            
        system : str, optional
            System prompt to set assistant behavior.
            
        **kwargs : dict
            Additional parameters (temperature, max_tokens, etc.)
            See chat() docstring for full list.
        
        Yields:
        ------
        str
            Individual tokens or chunks as they're generated.
            Chunks may be single tokens or small groups of tokens.
        
        Note:
        ----
        Token usage statistics are not available when streaming.
        If you need usage data, use chat_complete() instead.
        
        Example:
        -------
        >>> # Print tokens as they arrive
        >>> for chunk in ai.chat_stream("Tell me a story"):
        ...     print(chunk, end="", flush=True)
        >>> print()  # Final newline
        
        >>> # Collect into string if needed
        >>> chunks = list(ai.chat_stream("Explain AI"))
        >>> full_response = "".join(chunks)
        
        >>> # With system prompt
        >>> for chunk in ai.chat_stream(
        ...     "Explain quantum computing",
        ...     system="Explain like I'm five"
        ... ):
        ...     print(chunk, end="", flush=True)
        """
        # ── Model Resolution ────────────────────────────────────────────
        model = self._resolve_model(model)
        
        # ── Build Messages Array ────────────────────────────────────────
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        extra_body = self._build_rag_extra_body(
            collections=collections,
            extra_body=kwargs.pop("extra_body", None)
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        
        # ── Invoke Request Callbacks ────────────────────────────────────
        if self.on_request_start:
            self.on_request_start("chat_stream")
        
        # ── Streaming API Call ──────────────────────────────────────────
        try:
            # The stream=True parameter returns an iterator instead of
            # waiting for the complete response
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,  # Enable streaming
                **kwargs
            )
            
            # Yield each chunk as it arrives
            for chunk in stream:
                # Check if this chunk has content
                # Some chunks may be empty (e.g., role-only initial chunk)
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            if self.on_request_end:
                self.on_request_end("chat_stream")
    
    def chat_messages(
        self,
        messages: list[dict] | list[ChatMessage],
        model: str | None = None,
        stream: bool = False,
        collections: list[str] | None = None,  # ← ADD
        **kwargs
    ) -> ChatResponse | Iterator[str]:
        """
        Send a multi-turn conversation.
        
        Use this when you need full control over the message history,
        such as when implementing your own conversation management
        or injecting specific message patterns.
        
        For simpler conversation tracking, consider using chat_conversation()
        with a Conversation object instead.
        
        Parameters:
        ----------
        messages : list[dict] or list[ChatMessage]
            Full conversation history. Can be either:
            - List of dicts with 'role' and 'content' keys
            - List of ChatMessage objects
            
            Roles should be: 'system', 'user', 'assistant'
            
        model : str, optional
            Model ID to use. Falls back to selected model.
            
        stream : bool, optional
            If True, return a streaming iterator instead of waiting
            for the complete response.
            Default: False
            
        **kwargs : dict
            Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
        -------
        ChatResponse or Iterator[str]
            - If stream=False: ChatResponse with full response
            - If stream=True: Iterator yielding tokens
        
        Example:
        -------
        >>> # Using dicts
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful tutor."},
        ...     {"role": "user", "content": "What is correlation?"},
        ...     {"role": "assistant", "content": "Correlation measures..."},
        ...     {"role": "user", "content": "How do I interpret it?"}
        ... ]
        >>> response = ai.chat_messages(messages)
        >>> print(response.content)
        
        >>> # Using ChatMessage objects
        >>> messages = [
        ...     ChatMessage.system("Be concise"),
        ...     ChatMessage.user("What is mean?"),
        ... ]
        >>> response = ai.chat_messages(messages)
        
        >>> # With streaming
        >>> for chunk in ai.chat_messages(messages, stream=True):
        ...     print(chunk, end="", flush=True)
        """
        model = self._resolve_model(model)
        
        # ── Convert ChatMessage objects to dicts if needed ──────────────
        # The API expects dicts, but we accept both for convenience
        msg_list = [
            m.to_dict() if isinstance(m, ChatMessage) else m
            for m in messages
        ]
        extra_body = self._build_rag_extra_body(
            collections=collections,
            extra_body=kwargs.pop("extra_body", None)
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        
        # ── Handle streaming vs non-streaming ───────────────────────────
        if stream:
            return self._stream_messages(model, msg_list, **kwargs)
        
        # ── Non-streaming request ───────────────────────────────────────
        if self.on_request_start:
            self.on_request_start("chat_messages")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=msg_list,
                **kwargs
            )
        finally:
            if self.on_request_end:
                self.on_request_end("chat_messages")
        
        return ChatResponse(
            content=response.choices[0].message.content,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            prompt_tokens=getattr(response.usage, 'prompt_tokens', None),
            completion_tokens=getattr(response.usage, 'completion_tokens', None),
            total_tokens=getattr(response.usage, 'total_tokens', None),
            raw_response=response
        )
    
    def _stream_messages(
        self,
        model: str,
        messages: list[dict],
        **kwargs
    ) -> Iterator[str]:
        """
        Internal streaming implementation for messages.
        
        Separated from chat_messages to keep the return type handling clean.
        """
        if self.on_request_start:
            self.on_request_start("chat_stream")
        
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            if self.on_request_end:
                self.on_request_end("chat_stream")
    
    def chat_conversation(
        self,
        conversation: Conversation,
        model: str | None = None,
        stream: bool = False,
        auto_update: bool = True,
        collections: list[str] | None = None,  # ← ADD
        **kwargs
    ) -> ChatResponse | Iterator[str]:
        """
        Chat using a Conversation object that tracks history automatically.
        
        This is the recommended method for multi-turn conversations.
        The Conversation object maintains message history, and when
        auto_update=True (default), the assistant's response is automatically
        appended to the conversation.
        
        Parameters:
        ----------
        conversation : Conversation
            A Conversation object containing the system prompt and
            message history.
            
        model : str, optional
            Model ID to use. Falls back to selected model.
            
        stream : bool, optional
            If True, return a streaming iterator.
            Note: When streaming, you must manually add the assistant
            response to the conversation (auto_update doesn't work).
            Default: False
            
        auto_update : bool, optional
            If True and stream=False, automatically append the assistant's
            response to the conversation history.
            Default: True
            
        **kwargs : dict
            Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
        -------
        ChatResponse or Iterator[str]
            - If stream=False: ChatResponse with full response
            - If stream=True: Iterator yielding tokens
        
        Example:
        -------
        >>> # Create conversation with system prompt
        >>> conv = Conversation(system="You are a statistics tutor.")
        
        >>> # First turn
        >>> conv.add_user("What is variance?")
        >>> response = ai.chat_conversation(conv)
        >>> print(response.content)
        >>> # conv now has both user message and assistant response
        
        >>> # Second turn - full context is included automatically
        >>> conv.add_user("How is it calculated?")
        >>> response = ai.chat_conversation(conv)
        >>> # conv now has 4 messages: user, assistant, user, assistant
        
        >>> # With streaming (must manually track response)
        >>> conv.add_user("Give me an example")
        >>> full_response = []
        >>> for chunk in ai.chat_conversation(conv, stream=True, auto_update=False):
        ...     print(chunk, end="", flush=True)
        ...     full_response.append(chunk)
        >>> conv.add_assistant("".join(full_response))  # Manual update
        
        >>> # Clear and start over
        >>> conv.clear()
        >>> conv.add_user("New topic...")
        """
        model = self._resolve_model(model)
        
        # Convert Conversation to message list
        messages = conversation.to_messages()
        
        extra_body = self._build_rag_extra_body(
            collections=collections,
            extra_body=kwargs.pop("extra_body", None)
        )
        if extra_body:
            kwargs["extra_body"] = extra_body
        
        # ── Handle streaming ────────────────────────────────────────────
        if stream:
            # For streaming, caller must manually update conversation
            # because we can't know the full response until stream completes
            return self._stream_messages(model, messages, **kwargs)
        
        # ── Non-streaming request ───────────────────────────────────────
        response = self.chat_messages(messages, model=model, **kwargs)
        
        # Auto-update conversation with assistant response
        if auto_update:
            conversation.add_assistant(response.content)
        
        return response
    
    # ════════════════════════════════════════════════════════════════════
    # EMBEDDINGS
    # ════════════════════════════════════════════════════════════════════
    #
    # Methods for generating text embeddings (vector representations).
    #
    # Embeddings capture semantic meaning and can be used for:
    # - Semantic search (find similar documents)
    # - Clustering (group similar texts)
    # - Classification (categorize texts)
    # - Recommendations (find related content)
    # - Duplicate detection
    #
    # IMPORTANT WORKAROUND:
    # GenAI Studio's LiteLLM backend injects 'user' and 'encoding_format'
    # parameters that Ollama doesn't support. We suppress these with:
    #   extra_body={"user": None, "encoding_format": None}
    #
    # ════════════════════════════════════════════════════════════════════
    
    def embed(
        self,
        text: str | list[str],
        model: str | None = None
    ) -> list[float] | list[list[float]]:
        """
        Generate embedding vector(s) for text.
        
        Embeddings are numerical representations of text that capture
        semantic meaning. Similar texts will have similar embeddings
        (high cosine similarity).
        
        Parameters:
        ----------
        text : str or list[str]
            Text to embed. Can be:
            - Single string: Returns single embedding vector
            - List of strings: Returns list of embedding vectors
            
        model : str, optional
            Model ID to use. Falls back to selected model.
            Note: Not all models produce equally good embeddings.
            Llama models typically produce 3072-dimensional embeddings.
        
        Returns:
        -------
        list[float] or list[list[float]]
            - If text is a string: Returns single embedding (list of floats)
            - If text is a list: Returns list of embeddings
        
        Note:
        ----
        This method uses extra_body={"user": None, "encoding_format": None}
        to work around a LiteLLM/Ollama incompatibility on GenAI Studio.
        
        The server automatically injects:
        - user: "your-email@purdue.edu" (from API key authentication)
        - encoding_format: "base64" (from OpenAI SDK default)
        
        But Ollama (the backend) doesn't support these parameters,
        causing a 400 error. Setting them to None suppresses them.
        
        Example:
        -------
        >>> ai = GenAIStudio()
        >>> ai.select_model("llama3.2:latest")
        
        >>> # Single text
        >>> embedding = ai.embed("statistical significance")
        >>> print(f"Dimensions: {len(embedding)}")
        Dimensions: 3072
        
        >>> # Batch processing (more efficient for multiple texts)
        >>> texts = ["mean", "median", "mode"]
        >>> embeddings = ai.embed(texts)
        >>> print(f"Got {len(embeddings)} embeddings")
        Got 3 embeddings
        
        >>> # Semantic similarity
        >>> e1 = ai.embed("king")
        >>> e2 = ai.embed("queen")
        >>> e3 = ai.embed("banana")
        >>> 
        >>> # king and queen should be more similar than king and banana
        >>> sim_king_queen = ai.cosine_similarity(e1, e2)
        >>> sim_king_banana = ai.cosine_similarity(e1, e3)
        >>> print(f"king-queen: {sim_king_queen:.3f}")   # ~0.85
        >>> print(f"king-banana: {sim_king_banana:.3f}") # ~0.45
        """
        # Use embed_complete internally, extract just the vectors
        response = self.embed_complete(text, model=model)
        
        # Return single embedding or list based on input type
        if isinstance(text, str):
            return response.embeddings[0]
        return response.embeddings
    
    def embed_complete(
        self,
        text: str | list[str],
        model: str | None = None
    ) -> EmbeddingResponse:
        """
        Generate embeddings with full response details.
        
        Similar to embed(), but returns an EmbeddingResponse object
        with additional metadata like dimension and token usage.
        
        Parameters:
        ----------
        text : str or list[str]
            Text(s) to embed.
            
        model : str, optional
            Model ID to use. Falls back to selected model.
        
        Returns:
        -------
        EmbeddingResponse
            Full response with:
            - embeddings: List of embedding vectors
            - texts: The input texts
            - model: Model used
            - dimension: Vector dimensionality
            - prompt_tokens: Token count
            - raw_response: Original API response
        
        Example:
        -------
        >>> response = ai.embed_complete(["cat", "dog", "car"])
        >>> 
        >>> print(f"Model: {response.model}")
        >>> print(f"Dimension: {response.dimension}")
        >>> print(f"Tokens used: {response.prompt_tokens}")
        >>> 
        >>> # Access embeddings by index
        >>> cat_emb = response[0]
        >>> dog_emb = response[1]
        >>> 
        >>> # Iterate with text-embedding pairs
        >>> for text, emb in zip(response.texts, response.embeddings):
        ...     print(f"'{text}' → {len(emb)} dimensions")
        """
        model = self._resolve_model(model)
        
        # ── Normalize input to list ─────────────────────────────────────
        # API expects a list, but we accept single string for convenience
        texts = [text] if isinstance(text, str) else list(text)
        
        # ── Invoke Request Callbacks ────────────────────────────────────
        if self.on_request_start:
            self.on_request_start("embed")
        
        # ── API Call with Workaround ────────────────────────────────────
        try:
            # IMPORTANT: extra_body workaround for GenAI Studio
            #
            # The server (LiteLLM) automatically injects:
            #   - user: "your-email@purdue.edu" (from API key auth)
            #   - encoding_format: "base64" (from OpenAI SDK default)
            #
            # But Ollama (the backend) doesn't support these parameters,
            # so it returns: "litellm.UnsupportedParamsError"
            #
            # Setting them to None in extra_body tells the OpenAI client
            # to omit these fields from the request entirely.
            response = self.client.embeddings.create(
                model=model,
                input=texts,
                extra_body={"user": None, "encoding_format": None}
            )
        finally:
            if self.on_request_end:
                self.on_request_end("embed")
        
        # ── Build Response Object ───────────────────────────────────────
        embeddings = [d.embedding for d in response.data]
        
        return EmbeddingResponse(
            embeddings=embeddings,
            texts=texts,
            model=model,
            dimension=len(embeddings[0]) if embeddings else 0,
            prompt_tokens=getattr(response.usage, 'prompt_tokens', None),
            raw_response=response
        )
    
    # ════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ════════════════════════════════════════════════════════════════════
    #
    # Helper methods for common operations like similarity computation
    # and connection health checking.
    #
    # ════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        
        Cosine similarity measures the angle between two vectors,
        giving a value between -1 (opposite) and 1 (identical).
        For text embeddings, values typically range from 0.3 to 0.95.
        
        Uses numpy if available for efficiency, otherwise falls back
        to pure Python implementation.
        
        Parameters:
        ----------
        a : list[float]
            First embedding vector.
            
        b : list[float]
            Second embedding vector.
            Must have same dimensionality as a.
        
        Returns:
        -------
        float
            Cosine similarity between the vectors.
            - 1.0: Identical direction (very similar meaning)
            - 0.0: Orthogonal (unrelated)
            - -1.0: Opposite direction (rare for text embeddings)
        
        Note:
        ----
        This is a static method, so it can be called without instantiating
        the class: GenAIStudio.cosine_similarity(v1, v2)
        
        Example:
        -------
        >>> emb1 = ai.embed("happy")
        >>> emb2 = ai.embed("joyful")
        >>> emb3 = ai.embed("sad")
        >>> 
        >>> sim_happy_joyful = ai.cosine_similarity(emb1, emb2)
        >>> sim_happy_sad = ai.cosine_similarity(emb1, emb3)
        >>> 
        >>> print(f"happy-joyful: {sim_happy_joyful:.4f}")  # High
        >>> print(f"happy-sad: {sim_happy_sad:.4f}")        # Lower
        """
        try:
            import numpy as np
            a_arr, b_arr = np.array(a), np.array(b)
            denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
            if denom == 0:
                return 0.0
            return float(np.dot(a_arr, b_arr) / denom)
        except ImportError:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            denom = norm_a * norm_b
            if denom == 0:
                return 0.0
            return dot / denom
    
    def similarity(
        self,
        text1: str,
        text2: str,
        model: str | None = None
    ) -> float:
        """
        Compute semantic similarity between two texts.
        
        This is a convenience method that embeds both texts and computes
        their cosine similarity in one call.
        
        Parameters:
        ----------
        text1 : str
            First text to compare.
            
        text2 : str
            Second text to compare.
            
        model : str, optional
            Model to use for embeddings. Falls back to selected model.
        
        Returns:
        -------
        float
            Cosine similarity between the text embeddings.
            Higher values (closer to 1.0) indicate more similar meaning.
            
            Typical ranges:
            - 0.85+: Very similar (synonyms, paraphrases)
            - 0.70-0.85: Related (same topic)
            - 0.50-0.70: Somewhat related
            - <0.50: Unrelated
        
        Example:
        -------
        >>> ai.select_model("llama3.2:latest")
        >>> 
        >>> # Similar concepts
        >>> sim = ai.similarity("machine learning", "artificial intelligence")
        >>> print(f"ML vs AI: {sim:.4f}")  # ~0.85
        >>> 
        >>> # Unrelated concepts
        >>> sim = ai.similarity("machine learning", "banana bread recipe")
        >>> print(f"ML vs banana: {sim:.4f}")  # ~0.45
        """
        # Embed both texts in single batch call (more efficient)
        embeddings = self.embed([text1, text2], model=model)
        return self.cosine_similarity(embeddings[0], embeddings[1])
    
    def health_check(self) -> dict:
        """
        Check API connection health and availability.
        
        Useful for verifying connectivity, especially in automated
        scripts or at application startup.
        
        Returns:
        -------
        dict
            Status information with keys:
            - status: "connected" or "error"
            - base_url: The API base URL
            - model_count: Number of available models (if connected)
            - selected_model: Currently selected model (if any)
            - error: Error message (if status is "error")
        
        Example:
        -------
        >>> health = ai.health_check()
        >>> if health["status"] == "connected":
        ...     print(f"Connected! {health['model_count']} models available")
        ... else:
        ...     print(f"Connection failed: {health['error']}")
        
        >>> # Use in startup validation
        >>> ai = GenAIStudio()
        >>> health = ai.health_check()
        >>> assert health["status"] == "connected", f"API unavailable: {health['error']}"
        """
        result = {
            "status": "unknown",
            "base_url": self.base_url,
            "model_count": 0,
            "selected_model": self._model,
            "error": None
        }
        
        try:
            # Force a fresh model list fetch to verify connectivity
            models = self.refresh_models()
            result["status"] = "connected"
            result["model_count"] = len(models)
        except GenAIStudioError as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    def __repr__(self) -> str:
        """
        String representation for debugging.
        
        Shows current model and count of available models.
        
        Example:
        -------
        >>> ai = GenAIStudio()
        >>> print(ai)
        GenAIStudio(model=None, available=? models)
        
        >>> ai.select_model("gemma3:12b")
        >>> print(ai)
        GenAIStudio(model=gemma3:12b, available=36 models)
        """
        model_count = len(self._available_models) if self._available_models else "?"
        return f"GenAIStudio(model={self._model}, available={model_count} models)"


# ════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ════════════════════════════════════════════════════════════════════════════
#
# The CLI provides command-line access to all major features.
#
# Commands:
#   models  - List available models
#   chat    - Chat with a model (single query or interactive)
#   embed   - Generate embeddings
#   health  - Check API connectivity
#
# Global options:
#   --timeout SEC     Request timeout
#   --quiet           Suppress progress indicators
#   --no-validate     Skip model validation
#
# ════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point for the CLI.
    
    Parses command-line arguments and dispatches to the appropriate
    command handler.
    """
    import argparse
    
    # ════════════════════════════════════════════════════════════════════
    # ARGUMENT PARSER SETUP
    # ════════════════════════════════════════════════════════════════════
    
    parser = argparse.ArgumentParser(
        prog="genai_studio",
        description="CLI for Purdue GenAI Studio API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s models                              List all available models
  %(prog)s models --filter llama               Filter models by name
  %(prog)s chat -m gemma3:12b "What is AI?"    Single chat query
  %(prog)s chat -m gemma3:12b -i               Interactive chat session
  %(prog)s chat -m gemma3:12b -i --stream      Interactive with streaming
  %(prog)s chat -m gemma3:12b -k <kb_id> "Q?"  Chat with RAG context
  %(prog)s embed -m llama3.2:latest "a" "b" --similarity
  %(prog)s rag upload notes.pdf                Upload file for RAG
  %(prog)s rag create-kb "My Notes"            Create knowledge base
  %(prog)s rag link <kb_id> <file_id>          Link file to KB
  %(prog)s rag query -m gemma3:12b <kb_id> "question"
  %(prog)s rag list-kb                         List knowledge bases
  %(prog)s rag list-files                      List uploaded files
  %(prog)s rag delete-kb <kb_id>               Delete knowledge base
  %(prog)s rag test                            Run RAG lifecycle test
  %(prog)s health                              Check API connectivity
        """
    )
    
    # ── Global Options ──────────────────────────────────────────────────
    # These apply to all commands
    
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        metavar="SEC",
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages and spinners"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip model validation for faster startup"
    )
    
    # ── Subcommands ─────────────────────────────────────────────────────
    subparsers = parser.add_subparsers(dest="command")
    
    # ── models command ──────────────────────────────────────────────────
    models_parser = subparsers.add_parser(
        "models",
        help="List available models"
    )
    models_parser.add_argument(
        "--filter", "-f",
        type=str,
        metavar="PATTERN",
        help="Filter models containing this string (case-insensitive)"
    )
    models_parser.add_argument(
        "--count", "-c",
        action="store_true",
        help="Only show count of models"
    )
    models_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    # ── chat command ────────────────────────────────────────────────────
    chat_parser = subparsers.add_parser(
        "chat",
        help="Chat with a model"
    )
    chat_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        metavar="MODEL",
        help="Model ID to use (required)"
    )
    chat_parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        help="Chat prompt (omit for interactive mode)"
    )
    chat_parser.add_argument(
        "--system", "-s",
        type=str,
        metavar="PROMPT",
        help="System prompt to set assistant behavior"
    )
    chat_parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive chat session with history"
    )
    chat_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream responses token by token"
    )
    chat_parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        metavar="TEMP",
        help="Temperature 0.0-2.0 (default: 0.7)"
    )
    chat_parser.add_argument(
        "--max-tokens",
        type=int,
        metavar="N",
        help="Maximum tokens in response"
    )
    chat_parser.add_argument(
        "--collections", "-k", nargs="*", metavar="KB_ID",
        help="Knowledge base IDs for RAG-grounded chat"
    )
    # ── embed command ───────────────────────────────────────────────────
    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate text embeddings"
    )
    embed_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        metavar="MODEL",
        help="Model ID to use (required)"
    )
    embed_parser.add_argument(
        "texts",
        type=str,
        nargs="+",
        help="Text(s) to embed"
    )
    embed_parser.add_argument(
        "--similarity",
        action="store_true",
        help="Compute pairwise cosine similarities"
    )
    embed_parser.add_argument(
        "--dims",
        type=int,
        default=5,
        metavar="N",
        help="Number of dimensions to preview (default: 5)"
    )
    embed_parser.add_argument(
        "--json",
        action="store_true",
        help="Output full embeddings as JSON"
    )
    rag_parser = subparsers.add_parser(
        "rag",
        help="RAG: file & knowledge base management"
    )
    # ── rag command ─────────────────────────────────────────────────────
    rag_sub = rag_parser.add_subparsers(dest="rag_command")
    
    # rag upload
    rag_upload = rag_sub.add_parser("upload", help="Upload a file")
    rag_upload.add_argument("filepath", type=str, help="Path to file to upload")
    
    # rag list-files
    rag_sub.add_parser("list-files", help="List uploaded files")
    
    # rag delete-file
    rag_del_file = rag_sub.add_parser("delete-file", help="Delete an uploaded file")
    rag_del_file.add_argument("file_id", type=str)
    
    # rag create-kb
    rag_create = rag_sub.add_parser("create-kb", help="Create a knowledge base")
    rag_create.add_argument("name", type=str, help="Knowledge base name")
    rag_create.add_argument("--description", "-d", type=str, default="")
    
    # rag list-kb
    rag_sub.add_parser("list-kb", help="List knowledge bases")
    
    # rag delete-kb
    rag_del_kb = rag_sub.add_parser("delete-kb", help="Delete a knowledge base")
    rag_del_kb.add_argument("kb_id", type=str)
    
    # rag link
    rag_link = rag_sub.add_parser("link", help="Link a file to a knowledge base")
    rag_link.add_argument("kb_id", type=str, help="Knowledge base ID")
    rag_link.add_argument("file_id", type=str, help="File ID")
    
    # rag unlink
    rag_unlink = rag_sub.add_parser("unlink", help="Unlink a file from a knowledge base")
    rag_unlink.add_argument("kb_id", type=str)
    rag_unlink.add_argument("file_id", type=str)
    
    # rag query
    rag_query = rag_sub.add_parser("query", help="Chat with a knowledge base")
    rag_query.add_argument("--model", "-m", type=str, required=True, metavar="MODEL")
    rag_query.add_argument("kb_id", type=str, help="Knowledge base ID")
    rag_query.add_argument("prompt", type=str, help="Question to ask")
    rag_query.add_argument("--stream", action="store_true")
    rag_query.add_argument("--temperature", "-t", type=float, default=0.7)
    
    # rag test
    rag_test = rag_sub.add_parser("test", help="Run full RAG lifecycle test")
    rag_test.add_argument("--model", "-m", type=str, default="mistral:latest")
    rag_test.add_argument("--wait", "-w", type=int, default=10,
                          help="Indexing wait time in seconds")
    # ── health command ──────────────────────────────────────────────────
    health_parser = subparsers.add_parser(
        "health",
        help="Check API connectivity and health"
    )
    
    # ════════════════════════════════════════════════════════════════════
    # PARSE AND DISPATCH
    # ════════════════════════════════════════════════════════════════════
    
    args = parser.parse_args()
    
    # Show help if no command given
    if not args.command:
        parser.print_help()
        return 0
    
    # ── Initialize Client ───────────────────────────────────────────────
    try:
        ai = GenAIStudio(
            timeout=args.timeout,
            validate_model=not args.no_validate
        )
    except AuthenticationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # ── Dispatch to Command Handler ─────────────────────────────────────
    try:
        if args.command == "models":
            return cmd_models(ai, args)
        elif args.command == "chat":
            return cmd_chat(ai, args)
        elif args.command == "embed":
            return cmd_embed(ai, args)
        elif args.command == "rag":
            return cmd_rag(ai, args)
        elif args.command == "health":
            return cmd_health(ai, args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130  # Standard exit code for SIGINT
    except GenAIStudioError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


# ════════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ════════════════════════════════════════════════════════════════════════════

def cmd_models(ai: GenAIStudio, args) -> int:
    """
    Handle the 'models' command - list available models.
    """
    # Show spinner while fetching (unless quiet mode)
    if not args.quiet:
        spinner = Spinner("Fetching models...")
        spinner.start()
    
    try:
        models = ai.models
    finally:
        if not args.quiet:
            spinner.stop()
    
    # Apply filter if specified
    if args.filter:
        models = [m for m in models if args.filter.lower() in m.lower()]
    
    # Output in requested format
    if args.json:
        print(json.dumps({"models": models, "count": len(models)}, indent=2))
    elif args.count:
        print(len(models))
    else:
        print(f"Available models ({len(models)}):\n")
        for model in models:
            print(f"  {model}")
    
    return 0


def cmd_chat(ai: GenAIStudio, args) -> int:
    """
    Handle the 'chat' command - chat with a model.
    """
    # Select model
    try:
        ai.select_model(args.model)
    except ModelNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    kwargs = {"temperature": args.temperature}
    if args.max_tokens:
        kwargs["max_tokens"] = args.max_tokens
    
    collections = getattr(args, 'collections', None)
    
    if args.interactive or not args.prompt:
        return chat_interactive(ai, args.system, kwargs, stream=args.stream,
                                quiet=args.quiet, collections=collections)
    else:
        return chat_single(ai, args.prompt, args.system, kwargs, stream=args.stream,
                           quiet=args.quiet, collections=collections)

def cmd_rag(ai: GenAIStudio, args) -> int:
    """
    Handle all RAG subcommands: file upload/management, knowledge base
    CRUD, querying, and lifecycle testing.
    """
    if not args.rag_command:
        print("Usage: genai_studio rag <subcommand>")
        print("Subcommands: upload, list-files, delete-file, create-kb, list-kb,")
        print("             delete-kb, link, unlink, query, test")
        return 1
    
    subcmd = args.rag_command
    
    # ── upload ──────────────────────────────────────────────────────────
    if subcmd == "upload":
        info = ai.upload_file(args.filepath)
        print(f"✅ Uploaded: {info.filename}")
        print(f"   File ID: {info.id}")
        return 0
    
    # ── list-files ──────────────────────────────────────────────────────
    elif subcmd == "list-files":
        files = ai.list_files()
        if not files:
            print("No files found.")
        else:
            print(f"Files ({len(files)}):\n")
            for f in files:
                print(f"  {f.id}  {f.filename}")
        return 0
    
    # ── delete-file ─────────────────────────────────────────────────────
    elif subcmd == "delete-file":
        ai.delete_file(args.file_id)
        print(f"✅ Deleted file {args.file_id}")
        return 0
    
    # ── create-kb ───────────────────────────────────────────────────────
    elif subcmd == "create-kb":
        kb = ai.create_knowledge_base(args.name, args.description)
        print(f"✅ Created: {kb.name}")
        print(f"   KB ID: {kb.id}")
        return 0
    
    # ── list-kb ─────────────────────────────────────────────────────────
    elif subcmd == "list-kb":
        kbs = ai.list_knowledge_bases()
        if not kbs:
            print("No knowledge bases found.")
        else:
            print(f"Knowledge bases ({len(kbs)}):\n")
            for kb in kbs:
                desc = f" - {kb.description}" if kb.description else ""
                print(f"  {kb.id}  {kb.name}{desc}")
        return 0
    
    # ── delete-kb ───────────────────────────────────────────────────────
    elif subcmd == "delete-kb":
        ai.delete_knowledge_base(args.kb_id)
        print(f"✅ Deleted knowledge base {args.kb_id}")
        return 0
    
    # ── link ────────────────────────────────────────────────────────────
    elif subcmd == "link":
        ai.add_file_to_knowledge_base(args.kb_id, args.file_id)
        print(f"✅ Linked file {args.file_id} → KB {args.kb_id}")
        return 0
    
    # ── unlink ──────────────────────────────────────────────────────────
    elif subcmd == "unlink":
        ai.remove_file_from_knowledge_base(args.kb_id, args.file_id)
        print(f"✅ Unlinked file {args.file_id} from KB {args.kb_id}")
        return 0
    
    # ── query ───────────────────────────────────────────────────────────
    elif subcmd == "query":
        try:
            ai.select_model(args.model)
        except ModelNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        
        kwargs = {"temperature": args.temperature}
        
        if args.stream:
            for chunk in ai.chat_stream(args.prompt, collections=[args.kb_id], **kwargs):
                print(chunk, end="", flush=True)
            print()
        else:
            if not args.quiet:
                spinner = Spinner("Querying...")
                spinner.start()
            try:
                response = ai.chat(args.prompt, collections=[args.kb_id], **kwargs)
            finally:
                if not args.quiet:
                    spinner.stop()
            print(response)
        return 0
    
    # ── test ────────────────────────────────────────────────────────────
    elif subcmd == "test":
        return cmd_rag_test(ai, args)
    
    return 1


def cmd_rag_test(ai: GenAIStudio, args) -> int:
    """
    Run a full RAG lifecycle test.
    
    Creates a file with a known secret, uploads it, builds a knowledge
    base, queries for the secret, and verifies retrieval. Cleans up
    all resources afterward.
    """
    import tempfile
    
    SECRET = "VORTEX-77"
    print("=" * 60)
    print("RAG Lifecycle Test")
    print("=" * 60)
    print(f"  Model:  {args.model}")
    print(f"  Secret: {SECRET}")
    print()
    
    # Create test file with known secret
    content = (
        "CONFIDENTIAL PROJECT OMEGA SPECS:\n"
        "1. The reactor core must be kept at 300 Kelvin.\n"
        f"2. The password for the control panel is '{SECRET}'.\n"
        "3. Do not open valve C under any circumstances.\n"
    )
    filepath = os.path.join(tempfile.gettempdir(), "rag_test_omega.txt")
    with open(filepath, "w") as f:
        f.write(content)
    
    kb = None
    try:
        # Step 1: Upload
        print("  [1] Uploading test file...")
        file_info = ai.upload_file(filepath)
        print(f"      ✅ File ID: {file_info.id}")
        
        time.sleep(2)
        
        # Step 2: Create KB
        print("  [2] Creating knowledge base...")
        kb = ai.create_knowledge_base("RAG Test KB", "Automated lifecycle test")
        print(f"      ✅ KB ID: {kb.id}")
        
        # Step 3: Link
        print("  [3] Linking file → KB...")
        ai.add_file_to_knowledge_base(kb.id, file_info.id)
        print("      ✅ Linked")
        
        # Step 4: Wait for indexing
        wait = args.wait
        print(f"  [4] Waiting {wait}s for indexing...", end="", flush=True)
        for _ in range(wait):
            time.sleep(1)
            print(".", end="", flush=True)
        print(" done")
        
        # Step 5: Query
        print("  [5] Querying for secret...")
        try:
            ai.select_model(args.model)
        except ModelNotFoundError:
            ai._model = args.model  # Skip validation if model not in cached list
        
        response = ai.chat(
            "What is the password for the control panel?",
            collections=[kb.id]
        )
        
        found = SECRET in response
        print(f"      Response: {response[:200]}")
        
        print()
        print("=" * 60)
        if found:
            print("  RESULT: ✅ RAG retrieval PASSED")
        else:
            print("  RESULT: ❌ RAG retrieval FAILED")
        print("=" * 60)
        
        return 0 if found else 1
    
    finally:
        print("\n  Cleanup...")
        if kb:
            try:
                ai.delete_knowledge_base(kb.id)
                print("      ✅ KB deleted")
            except Exception as e:
                print(f"      ⚠️  KB cleanup failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
            print("      ✅ Temp file removed")


def chat_single(
    ai: GenAIStudio,
    prompt: str,
    system: str | None,
    kwargs: dict,
    stream: bool,
    quiet: bool,
    collections: list[str] | None = None  # ← ADD
) -> int:
    """
    Handle a single chat query (non-interactive).
    """
    if stream:
        # Streaming: print tokens as they arrive
        for chunk in ai.chat_stream(prompt, system=system, collections=collections, **kwargs):
            print(chunk, end="", flush=True)
        print()  # Final newline
    else:
        # Non-streaming: show spinner while waiting
        if not quiet:
            spinner = Spinner("Thinking...")
            spinner.start()
        
        try:
            response = ai.chat(prompt, system=system, collections=collections, **kwargs)
        finally:
            if not quiet:
                spinner.stop()
        
        print(response)
    
    return 0


def chat_interactive(
    ai: GenAIStudio,
    system: str | None,
    kwargs: dict,
    stream: bool,
    quiet: bool,
    collections: list[str] | None = None  # ← ADD
) -> int:
    """
    Run an interactive chat session with history.
    """
    print(f"Chat with {ai.model} (type 'exit' or Ctrl+C to quit)")
    print("Commands: /clear, /history, /stream, /help\n")
    
    # Create conversation to track history
    conversation = Conversation(system=system)
    
    if system:
        print(f"System: {system}\n")
    
    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print()
            break
        
        if not user_input:
            continue
        
        # ── Exit Commands ───────────────────────────────────────────────
        if user_input.lower() in ("exit", "quit", "q", "/exit", "/quit", "/q"):
            break
        
        # ── Special Commands ────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower()
            
            if cmd == "/clear":
                conversation.clear()
                print("(conversation cleared)\n")
                continue
            
            elif cmd == "/history":
                if not conversation.messages:
                    print("(no messages yet)\n")
                else:
                    for msg in conversation.messages:
                        role = msg.role.capitalize()
                        content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
                        print(f"  [{role}] {content}")
                    print()
                continue
            
            elif cmd == "/stream":
                stream = not stream
                print(f"(streaming {'enabled' if stream else 'disabled'})\n")
                continue
            
            elif cmd == "/help":
                print("Commands:")
                print("  /clear   - Clear conversation history")
                print("  /history - Show conversation history")
                print("  /stream  - Toggle streaming mode")
                print("  /help    - Show this help")
                print("  exit     - Exit the chat\n")
                continue
            
            else:
                print(f"Unknown command: {cmd}\n")
                continue
        
        # ── Process Chat ────────────────────────────────────────────────
        conversation.add_user(user_input)
        
        try:
            if stream:
                # Streaming response
                print("\nAssistant: ", end="", flush=True)
                full_response = []
                for chunk in ai.chat_conversation(conversation, stream=True, auto_update=False, collections=collections, **kwargs):
                    print(chunk, end="", flush=True)
                    full_response.append(chunk)
                print("\n")
                # Manually add to conversation since auto_update doesn't work with streaming
                conversation.add_assistant("".join(full_response))
            else:
                # Non-streaming response
                if not quiet:
                    spinner = Spinner("Thinking...")
                    spinner.start()
                
                try:
                    response = ai.chat_conversation(conversation, collections=collections, **kwargs)
                finally:
                    if not quiet:
                        spinner.stop()
                
                print(f"\nAssistant: {response.content}\n")
        
        except GenAIStudioError as e:
            print(f"\nError: {e}\n")
            # Remove failed user message from history
            if conversation.messages and conversation.messages[-1].role == "user":
                conversation.messages.pop()
    
    print("Goodbye!")
    return 0


def cmd_embed(ai: GenAIStudio, args) -> int:
    """
    Handle the 'embed' command - generate embeddings.
    """
    # Select model
    try:
        ai.select_model(args.model)
    except ModelNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    texts = args.texts
    
    # Show progress
    if not args.quiet:
        print(f"Embedding {len(texts)} text(s) with {ai.model}...\n")
        spinner = Spinner("Generating embeddings...")
        spinner.start()
    
    try:
        response = ai.embed_complete(texts)
    finally:
        if not args.quiet:
            spinner.stop()
    
    # ── Output Formats ──────────────────────────────────────────────────
    
    if args.json:
        # Full JSON output
        output = {
            "model": response.model,
            "dimension": response.dimension,
            "embeddings": [
                {"text": t, "embedding": e}
                for t, e in zip(response.texts, response.embeddings)
            ]
        }
        print(json.dumps(output, indent=2))
    
    elif args.similarity and len(texts) >= 2:
        # Pairwise similarity matrix
        print(f"Dimension: {response.dimension}\n")
        print("Pairwise cosine similarities:\n")
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = ai.cosine_similarity(response[i], response[j])
                t1 = texts[i][:30] + "..." if len(texts[i]) > 30 else texts[i]
                t2 = texts[j][:30] + "..." if len(texts[j]) > 30 else texts[j]
                print(f"  \"{t1}\" ↔ \"{t2}\": {sim:.4f}")
    
    else:
        # Preview mode - show first N dimensions
        print(f"Dimension: {response.dimension}\n")
        
        for text, emb in zip(response.texts, response.embeddings):
            preview = text[:50] + "..." if len(text) > 50 else text
            dims_preview = ", ".join(f"{v:.6f}" for v in emb[:args.dims])
            print(f"\"{preview}\"")
            print(f"  [{dims_preview}, ...]\n")
    
    return 0


def cmd_health(ai: GenAIStudio, args) -> int:
    """
    Handle the 'health' command - check API connectivity.
    """
    if not args.quiet:
        spinner = Spinner("Checking connection...")
        spinner.start()
    
    try:
        result = ai.health_check()
    finally:
        if not args.quiet:
            spinner.stop()
    
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "connected" else 1


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.exit(main())
