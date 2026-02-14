#!/usr/bin/env python3
"""
Comprehensive Test Suite for GenAI Studio SDK v1.2
====================================================

Tests EVERY public method, setting, and dataclass in genai_studio.py
against the live Purdue GenAI Studio API. Generates toy files in multiple
formats and cycles through available models.

Test Sections:
  1. Client Initialization & Configuration
  2. Model Management
  3. Chat Completions (cycle models)
  4. Streaming
  5. Multi-Turn Conversations
  6. Embeddings & Similarity (cycle models)
  7. RAG: File Uploads (multiple file types)
  8. RAG: Knowledge Base CRUD
  9. RAG: Grounded Queries (cycle models × file types)
 10. Error Handling & Edge Cases
 11. Health Check & Utilities
 12. Dataclass Validation

Usage:
    export GENAI_STUDIO_API_KEY="your-key"
    python test_full_suite.py                       # Default: 3 chat models, 1 embed model
    python test_full_suite.py --all-models           # Test every available model
    python test_full_suite.py --chat-models gemma3:12b mistral:latest
    python test_full_suite.py --embed-models llama3.2:latest
    python test_full_suite.py --wait 20              # Longer RAG indexing wait
    python test_full_suite.py --skip-rag             # Skip RAG tests (faster)
    python test_full_suite.py --skip-embeds          # Skip embedding tests
    python test_full_suite.py --keep                 # Don't cleanup RAG resources
    python test_full_suite.py -v                     # Verbose output
"""

import os
import sys
import csv
import json
import time
import struct
import zlib
import argparse
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
import unicodedata


# Then in RAG assertions:
assert SECRET in normalize(resp), f"Secret not found: {resp[:150]}"
# ── Ensure genai_studio is importable ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genai_studio import (
    GenAIStudio, GenAIStudioError, AuthenticationError, ModelNotFoundError,
    ConnectionError as GSConnectionError, TimeoutError as GSTimeoutError,
    RAGError, ChatMessage, ChatResponse, EmbeddingResponse, Conversation,
    FileInfo, KnowledgeBase, __version__,
    DEFAULT_BASE_URL, DEFAULT_TIMEOUT, DEFAULT_CONNECT_TIMEOUT, ENV_API_KEY
)


# ════════════════════════════════════════════════════════════════════════════
# TEST FRAMEWORK
# ════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @staticmethod
    def enabled():
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


C = Colors if Colors.enabled() else type("NoColor", (), {k: "" for k in
    ["GREEN", "RED", "YELLOW", "CYAN", "DIM", "BOLD", "RESET"]})()



def normalize(text: str) -> str:
    """Normalize unicode dashes/quotes to ASCII for comparison."""
    return unicodedata.normalize("NFKC", text).replace("\u2011", "-").replace("\u2010", "-")

class TestResult:
    """Accumulates test pass/fail/skip results with timing."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []
        self.start_time = time.time()

    @property
    def total(self):
        return len(self.passed) + len(self.failed) + len(self.skipped)

    def record_pass(self, name, detail=""):
        self.passed.append((name, detail))

    def record_fail(self, name, err):
        self.failed.append((name, str(err)))

    def record_skip(self, name, reason):
        self.skipped.append((name, reason))

    def summary(self):
        elapsed = time.time() - self.start_time
        print()
        print(f"{C.BOLD}{'=' * 70}")
        p, f, s = len(self.passed), len(self.failed), len(self.skipped)
        color = C.GREEN if f == 0 else C.RED
        print(f"  RESULTS: {color}{p} passed{C.RESET}, "
              f"{C.RED if f else ''}{f} failed{C.RESET if f else ''}, "
              f"{s} skipped  ({self.total} total in {elapsed:.1f}s)")
        print(f"{'=' * 70}{C.RESET}")

        if self.failed:
            print(f"\n  {C.RED}Failed tests:{C.RESET}")
            for name, err in self.failed:
                print(f"    ❌ {name}")
                for line in err.split("\n")[:3]:
                    print(f"       {C.DIM}{line}{C.RESET}")

        if self.skipped:
            print(f"\n  {C.YELLOW}Skipped:{C.RESET}")
            for name, reason in self.skipped:
                print(f"    ⏭️  {name}: {reason}")

        print()
        return f == 0


def run_test(results: TestResult, name: str, fn, verbose=False):
    """Execute a test function, catch exceptions, record result."""
    num = results.total + 1
    print(f"  [{num:3d}] {name}...", end=" ", flush=True)
    try:
        detail = fn()
        results.record_pass(name, detail)
        if detail and verbose:
            print(f"{C.GREEN}✅{C.RESET} {C.DIM}({detail}){C.RESET}")
        else:
            print(f"{C.GREEN}✅{C.RESET}")
    except AssertionError as e:
        results.record_fail(name, str(e))
        print(f"{C.RED}❌ {e}{C.RESET}")
    except RAGError as e:
        results.record_fail(name, str(e))
        print(f"{C.RED}❌ RAGError: {e}{C.RESET}")
    except Exception as e:
        results.record_fail(name, str(e))
        print(f"{C.RED}❌ {type(e).__name__}: {e}{C.RESET}")
        if verbose:
            traceback.print_exc()


def skip_test(results: TestResult, name: str, reason: str):
    """Record a skipped test."""
    num = results.total + 1
    print(f"  [{num:3d}] {name}... {C.YELLOW}⏭️  {reason}{C.RESET}")
    results.record_skip(name, reason)


# ════════════════════════════════════════════════════════════════════════════
# TOY FILE GENERATORS
# ════════════════════════════════════════════════════════════════════════════
# Each generator creates a small file with known, queryable content.
# The SECRET is embedded in each file so RAG retrieval can be verified.

SECRET = "VORTEX-77"
SECRET_QUESTION = "What is the password for the control panel?"
SECOND_FACT = "PURPLE"
SECOND_QUESTION = "What color badge is needed for level 5 clearance?"


def gen_txt(directory: str) -> str:
    """Plain text file with embedded secrets."""
    path = os.path.join(directory, "test_specs.txt")
    with open(path, "w") as f:
        f.write(
            "PROJECT OMEGA SPECIFICATIONS\n"
            "============================\n\n"
            "Section 1: Security\n"
            f"  The password for the control panel is '{SECRET}'.\n"
            f"  Badge color for level 5 clearance is {SECOND_FACT}.\n\n"
            "Section 2: Operations\n"
            "  The reactor core must maintain 300 Kelvin.\n"
            "  Maximum personnel in reactor room: 3.\n"
        )
    return path


def gen_csv(directory: str) -> str:
    """CSV file with tabular data and embedded secrets."""
    path = os.path.join(directory, "test_data.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "value", "notes"])
        writer.writerow(["control_panel_password", SECRET, "Do not share"])
        writer.writerow(["clearance_badge_color", SECOND_FACT, "Level 5 only"])
        writer.writerow(["reactor_temp_kelvin", "300", "Critical threshold"])
        writer.writerow(["max_personnel", "3", "Reactor room limit"])
        writer.writerow(["emergency_code", "ALPHA-BRAVO-9", "Use in emergencies"])
    return path


def gen_json(directory: str) -> str:
    """JSON file with structured data and embedded secrets."""
    path = os.path.join(directory, "test_config.json")
    data = {
        "project": "OMEGA",
        "security": {
            "control_panel_password": SECRET,
            "clearance_badge_color": SECOND_FACT,
            "emergency_code": "ALPHA-BRAVO-9"
        },
        "operations": {
            "reactor_temp_kelvin": 300,
            "max_personnel": 3,
            "valve_c_status": "LOCKED"
        }
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def gen_md(directory: str) -> str:
    """Markdown file with formatted content and secrets."""
    path = os.path.join(directory, "test_notes.md")
    with open(path, "w") as f:
        f.write(
            "# Project Omega Notes\n\n"
            "## Security Credentials\n\n"
            f"- **Control panel password**: `{SECRET}`\n"
            f"- **Level 5 badge color**: {SECOND_FACT}\n\n"
            "## Operations\n\n"
            "| Parameter | Value |\n"
            "|-----------|-------|\n"
            "| Reactor temp | 300K |\n"
            "| Max personnel | 3 |\n"
        )
    return path


def gen_py(directory: str) -> str:
    """Python source file with secrets in comments/strings."""
    path = os.path.join(directory, "test_config.py")
    with open(path, "w") as f:
        f.write(
            '"""Project Omega Configuration"""\n\n'
            f'CONTROL_PANEL_PASSWORD = "{SECRET}"  # Do not share\n'
            f'CLEARANCE_BADGE_COLOR = "{SECOND_FACT}"  # Level 5\n'
            'REACTOR_TEMP_KELVIN = 300\n'
            'MAX_PERSONNEL = 3\n'
            'EMERGENCY_CODE = "ALPHA-BRAVO-9"\n'
        )
    return path


def gen_html(directory: str) -> str:
    """HTML file (using our guide as inspiration) with secrets."""
    path = os.path.join(directory, "test_report.html")
    with open(path, "w") as f:
        f.write(
            "<!DOCTYPE html><html><head><title>Project Omega Report</title></head>\n"
            "<body>\n"
            "<h1>Project Omega Security Report</h1>\n"
            f"<p>The control panel password is <strong>{SECRET}</strong>.</p>\n"
            f"<p>Level 5 clearance badge color: <em>{SECOND_FACT}</em></p>\n"
            "<table><tr><th>Parameter</th><th>Value</th></tr>\n"
            "<tr><td>Reactor Temp</td><td>300K</td></tr>\n"
            "<tr><td>Max Personnel</td><td>3</td></tr></table>\n"
            "</body></html>\n"
        )
    return path


def gen_xlsx(directory: str) -> str | None:
    """Excel file with tabular data. Requires openpyxl."""
    try:
        import openpyxl
    except ImportError:
        return None
    path = os.path.join(directory, "test_data.xlsx")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Security"
    ws.append(["Item", "Value", "Notes"])
    ws.append(["control_panel_password", SECRET, "Do not share"])
    ws.append(["clearance_badge_color", SECOND_FACT, "Level 5"])
    ws.append(["reactor_temp_kelvin", 300, "Critical"])
    ws.append(["max_personnel", 3, "Reactor room"])
    wb.save(path)
    return path


def gen_pdf(directory: str) -> str | None:
    """
    Minimal valid PDF with embedded secrets.
    Uses raw PDF syntax — no external library needed.
    """
    path = os.path.join(directory, "test_specs.pdf")
    text_line1 = f"The control panel password is {SECRET}."
    text_line2 = f"Level 5 clearance badge color is {SECOND_FACT}."

    # Build a minimal but valid PDF 1.4 from scratch
    objects = []

    # Object 1: Catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # Object 2: Pages
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    # Object 3: Page
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] /Contents 4 0 R /Resources "
        b"<< /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    # Object 4: Content stream
    stream = (
        f"BT /F1 12 Tf 72 720 Td ({text_line1}) Tj "
        f"0 -20 Td ({text_line2}) Tj ET"
    ).encode()
    objects.append(
        f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode()
        + stream + b"\nendstream\nendobj\n"
    )
    # Object 5: Font
    objects.append(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 "
        b"/BaseFont /Helvetica >>\nendobj\n"
    )

    body = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = []
    for obj in objects:
        offsets.append(len(body))
        body += obj

    # Cross-reference table
    xref_offset = len(body)
    xref = f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"
    body += xref.encode()
    body += f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n".encode()
    body += f"startxref\n{xref_offset}\n%%EOF\n".encode()

    with open(path, "wb") as f:
        f.write(body)
    return path


def gen_png(directory: str) -> str:
    """
    Minimal valid PNG image (8x8 red square).
    Uses raw PNG format — no Pillow needed.
    """
    path = os.path.join(directory, "test_image.png")
    width, height = 8, 8

    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)

    # IHDR: 8x8, 8-bit RGB
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))

    # IDAT: red pixels (R=255, G=0, B=0), filter byte 0 per row
    raw = b""
    for _ in range(height):
        raw += b"\x00" + (b"\xff\x00\x00" * width)
    idat = chunk(b"IDAT", zlib.compress(raw))

    # IEND
    iend = chunk(b"IEND", b"")

    with open(path, "wb") as f:
        f.write(sig + ihdr + idat + iend)
    return path


# Mapping of extension → generator function
FILE_GENERATORS = {
    ".txt":  ("Plain text",        gen_txt),
    ".csv":  ("CSV spreadsheet",   gen_csv),
    ".json": ("JSON data",         gen_json),
    ".md":   ("Markdown",          gen_md),
    ".py":   ("Python source",     gen_py),
    ".html": ("HTML document",     gen_html),
    ".xlsx": ("Excel workbook",    gen_xlsx),
    ".pdf":  ("PDF document",      gen_pdf),
    ".png":  ("PNG image",         gen_png),
}


def generate_all_test_files(directory: str) -> dict[str, str]:
    """
    Generate all toy test files. Returns {extension: filepath}.
    Skips files whose generator returns None (missing dependency).
    """
    files = {}
    for ext, (desc, gen_fn) in FILE_GENERATORS.items():
        try:
            path = gen_fn(directory)
            if path:
                files[ext] = path
        except Exception as e:
            print(f"  {C.YELLOW}⚠️  Could not generate {ext}: {e}{C.RESET}")
    return files


# ════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive test suite for GenAI Studio SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--chat-models", nargs="*", metavar="MODEL",
                        help="Chat models to test (default: auto-select 3)")
    parser.add_argument("--embed-models", nargs="*", metavar="MODEL",
                        help="Embedding models to test (default: auto-select 1)")
    parser.add_argument("--all-models", action="store_true",
                        help="Test ALL available models (slow)")
    parser.add_argument("--wait", "-w", type=int, default=12,
                        help="RAG indexing wait in seconds (default: 12)")
    parser.add_argument("--skip-rag", action="store_true",
                        help="Skip RAG tests entirely")
    parser.add_argument("--skip-embeds", action="store_true",
                        help="Skip embedding tests")
    parser.add_argument("--keep", action="store_true",
                        help="Don't cleanup RAG resources after tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    args = parser.parse_args()

    print(f"\n{C.BOLD}{'=' * 70}")
    print(f"  GenAI Studio SDK — Full Test Suite")
    print(f"  Version: {__version__}    Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 70}{C.RESET}\n")

    results = TestResult()

    # ════════════════════════════════════════════════════════════════════
    # SECTION 1: CLIENT INITIALIZATION
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{C.CYAN}── 1. Client Initialization & Configuration ────────{C.RESET}")

    # Test 1: Default initialization
    ai = [None]  # Use list for closure mutation

    def test_default_init():
        client = GenAIStudio(validate_model=False)
        assert client.api_key, "API key is empty"
        assert client.base_url == DEFAULT_BASE_URL, f"Base URL mismatch: {client.base_url}"
        assert client.timeout == DEFAULT_TIMEOUT
        assert client.connect_timeout == DEFAULT_CONNECT_TIMEOUT
        assert client._model is None, "Model should be None initially"
        assert client._available_models is None, "Model cache should be None"
        ai[0] = client
        return f"url={client.base_url}"

    run_test(results, "Default initialization", test_default_init, args.verbose)

    # Test 2: Custom timeout
    def test_custom_timeout():
        client = GenAIStudio(timeout=180, connect_timeout=45, validate_model=False)
        assert client.timeout == 180
        assert client.connect_timeout == 45

    run_test(results, "Custom timeout settings", test_custom_timeout)

    # Test 3: validate_model flag
    def test_validate_flag():
        client = GenAIStudio(validate_model=False)
        assert not client.validate_model
        client.select_model("nonexistent-model-xyz")  # Should NOT raise
        assert client.model == "nonexistent-model-xyz"

    run_test(results, "validate_model=False bypasses check", test_validate_flag)

    # Test 4: Missing API key
    def test_missing_key():
        orig = os.environ.get(ENV_API_KEY)
        try:
            os.environ.pop(ENV_API_KEY, None)
            try:
                GenAIStudio(api_key=None, validate_model=False)
                assert False, "Should have raised AuthenticationError"
            except AuthenticationError:
                pass  # Expected
        finally:
            if orig:
                os.environ[ENV_API_KEY] = orig

    run_test(results, "Missing API key raises AuthenticationError", test_missing_key)

    # Test 5: Callbacks
    def test_callbacks():
        client = GenAIStudio(validate_model=False)
        calls = []
        client.on_request_start = lambda op: calls.append(("start", op))
        client.on_request_end = lambda op: calls.append(("end", op))
        assert client.on_request_start is not None
        assert client.on_request_end is not None

    run_test(results, "Callback registration", test_callbacks)

    # Test 6: __repr__
    def test_repr():
        client = GenAIStudio(validate_model=False)
        r = repr(client)
        assert "GenAIStudio" in r
        assert "model=" in r

    run_test(results, "__repr__ output", test_repr)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 2: MODEL MANAGEMENT
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{C.CYAN}── 2. Model Management ─────────────────────────────{C.RESET}")

    if not ai[0]:
        ai[0] = GenAIStudio(validate_model=False)

    all_models = []

    def test_fetch_models():
        models = ai[0].models
        assert isinstance(models, list), f"Expected list, got {type(models)}"
        assert len(models) > 0, "No models returned"
        assert models == sorted(models), "Models not sorted"
        all_models.extend(models)
        return f"{len(models)} models"

    run_test(results, "Fetch model list (.models property)", test_fetch_models, args.verbose)

    def test_models_cached():
        assert ai[0]._available_models is not None, "Cache should be populated"
        models2 = ai[0].models  # Should use cache, not re-fetch
        assert models2 == all_models[:len(models2)]

    run_test(results, "Model list is cached", test_models_cached)

    def test_refresh_models():
        models = ai[0].refresh_models()
        assert isinstance(models, list)
        assert len(models) > 0

    run_test(results, "refresh_models() returns fresh list", test_refresh_models)

    def test_select_model_valid():
        if not all_models:
            raise AssertionError("No models available")
        ai[0].select_model(all_models[0])
        assert ai[0].model == all_models[0]

    run_test(results, "select_model() with valid model", test_select_model_valid)

    def test_select_model_invalid():
        try:
            temp = GenAIStudio(validate_model=True)
            temp._available_models = all_models  # Pre-populate cache
            temp.select_model("completely-fake-model-12345")
            assert False, "Should have raised ModelNotFoundError"
        except ModelNotFoundError:
            pass

    run_test(results, "select_model() invalid raises ModelNotFoundError", test_select_model_invalid)

    def test_select_model_chaining():
        result = ai[0].select_model(all_models[0])
        assert result is ai[0], "select_model should return self"

    run_test(results, "select_model() returns self (chaining)", test_select_model_chaining)

    def test_model_property_setter():
        ai[0].model = all_models[0]
        assert ai[0].model == all_models[0]

    run_test(results, ".model setter works", test_model_property_setter)

    # ── Determine which models to test ──────────────────────────────────
    # Default chat models: pick 3 diverse ones
    DEFAULT_CHAT_PREFERENCES = [
        "mistral:latest", "gemma3:12b", "llama3.2:latest",
        "deepseek-r1:1.5b", "phi4:latest"
    ]
    DEFAULT_EMBED_PREFERENCES = [
        "llama3.2:latest", "mistral:latest", "gemma3:12b"
    ]

    if args.all_models:
        chat_models = all_models
        embed_models = all_models[:3]  # Embed all would be very slow
    elif args.chat_models:
        chat_models = args.chat_models
    else:
        chat_models = [m for m in DEFAULT_CHAT_PREFERENCES if m in all_models][:3]
        if not chat_models and all_models:
            chat_models = all_models[:2]

    if args.embed_models:
        embed_models = args.embed_models
    elif not args.all_models:
        embed_models = [m for m in DEFAULT_EMBED_PREFERENCES if m in all_models][:1]
        if not embed_models and all_models:
            embed_models = [all_models[0]]

    print(f"\n  {C.DIM}Chat models:  {', '.join(chat_models)}{C.RESET}")
    print(f"  {C.DIM}Embed models: {', '.join(embed_models)}{C.RESET}")

    # ════════════════════════════════════════════════════════════════════
    # SECTION 3: CHAT COMPLETIONS (cycle models)
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{C.CYAN}── 3. Chat Completions ─────────────────────────────{C.RESET}")

    for model_id in chat_models:
        # chat() → str
        def test_chat_str(m=model_id):
            ai[0].select_model(m)
            response = ai[0].chat("Say hello in one word.")
            assert isinstance(response, str), f"Expected str, got {type(response)}"
            assert len(response) > 0, "Empty response"
            return f"len={len(response)}"

        run_test(results, f"chat() → str  [{model_id}]", test_chat_str, args.verbose)

        # chat() with system prompt
        def test_chat_system(m=model_id):
            ai[0].select_model(m)
            response = ai[0].chat(
                "What is 2+2?",
                system="Reply with only the number, nothing else."
            )
            assert isinstance(response, str)
            assert len(response) > 0

        run_test(results, f"chat() + system prompt  [{model_id}]", test_chat_system)

        # chat() with temperature and max_tokens
        def test_chat_params(m=model_id):
            ai[0].select_model(m)
            response = ai[0].chat("Hi", temperature=0.1, max_tokens=10)
            assert isinstance(response, str)

        run_test(results, f"chat() + temp/max_tokens  [{model_id}]", test_chat_params)

        # chat_complete() → ChatResponse
        def test_chat_complete(m=model_id):
            ai[0].select_model(m)
            response = ai[0].chat_complete("Say ok.")
            assert isinstance(response, ChatResponse)
            assert isinstance(response.content, str) and len(response.content) > 0
            assert response.model, "Model field is empty"
            assert response.raw_response is not None
            # Check usage dict
            usage = response.usage
            assert isinstance(usage, dict)
            return f"tokens={response.total_tokens}"

        run_test(results, f"chat_complete() → ChatResponse  [{model_id}]", test_chat_complete, args.verbose)

        # chat() with model= override
        def test_chat_model_override(m=model_id):
            ai[0]._model = None  # Clear selection
            response = ai[0].chat("Hello", model=m)
            assert isinstance(response, str) and len(response) > 0
            ai[0].select_model(m)  # Restore

        run_test(results, f"chat() model= param override  [{model_id}]", test_chat_model_override)

    # No model selected → ValueError
    def test_chat_no_model():
        temp = GenAIStudio(validate_model=False)
        temp._model = None
        try:
            temp.chat("Hello")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    run_test(results, "chat() without model raises ValueError", test_chat_no_model)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 4: STREAMING
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{C.CYAN}── 4. Streaming ────────────────────────────────────{C.RESET}")

    stream_model = chat_models[0] if chat_models else None
    if stream_model:
        def test_stream_basic():
            ai[0].select_model(stream_model)
            chunks = list(ai[0].chat_stream("Count from 1 to 3."))
            assert len(chunks) > 0, "No chunks received"
            full = "".join(chunks)
            assert len(full) > 0, "Empty concatenated response"
            return f"{len(chunks)} chunks, {len(full)} chars"

        run_test(results, f"chat_stream() yields chunks  [{stream_model}]", test_stream_basic, args.verbose)

        def test_stream_with_system():
            ai[0].select_model(stream_model)
            chunks = list(ai[0].chat_stream(
                "Say yes",
                system="Reply with only one word."
            ))
            assert len(chunks) > 0

        run_test(results, f"chat_stream() + system prompt  [{stream_model}]", test_stream_with_system)
    else:
        skip_test(results, "Streaming tests", "No chat models available")

    # ════════════════════════════════════════════════════════════════════
    # SECTION 5: MULTI-TURN CONVERSATIONS
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{C.CYAN}── 5. Multi-Turn Conversations ─────────────────────{C.RESET}")

    conv_model = chat_models[0] if chat_models else None
    if conv_model:
        # Conversation dataclass
        def test_conversation_basics():
            conv = Conversation(system="You are helpful.")
            assert conv.system == "You are helpful."
            assert len(conv) == 0
            conv.add_user("Hello")
            assert len(conv) == 1
            conv.add_assistant("Hi there!")
            assert len(conv) == 2
            msgs = conv.to_messages()
            assert len(msgs) == 3  # system + user + assistant
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert msgs[2]["role"] == "assistant"

        run_test(results, "Conversation dataclass basics", test_conversation_basics)

        def test_conversation_clear():
            conv = Conversation(system="test")
            conv.add_user("Hello").add_assistant("Hi")
            assert len(conv) == 2
            conv.clear()
            assert len(conv) == 0
            assert conv.system == "test"  # System preserved

        run_test(results, "Conversation.clear() preserves system", test_conversation_clear)

        def test_conversation_chaining():
            conv = Conversation()
            result = conv.add_user("Q1").add_assistant("A1").add_user("Q2")
            assert result is conv
            assert len(conv) == 3

        run_test(results, "Conversation method chaining", test_conversation_chaining)

        def test_conversation_no_system():
            conv = Conversation()
            conv.add_user("Hello")
            msgs = conv.to_messages()
            assert len(msgs) == 1
            assert msgs[0]["role"] == "user"

        run_test(results, "Conversation without system prompt", test_conversation_no_system)

        # chat_conversation() — live API
        def test_chat_conversation():
            ai[0].select_model(conv_model)
            conv = Conversation(system="Be concise.")
            conv.add_user("What is 2+2?")
            response = ai[0].chat_conversation(conv)
            assert isinstance(response, ChatResponse)
            assert len(conv) == 2, f"Expected 2 messages, got {len(conv)}"
            # Second turn
            conv.add_user("Add 3 to that.")
            response2 = ai[0].chat_conversation(conv)
            assert len(conv) == 4, f"Expected 4 messages, got {len(conv)}"
            return f"{len(conv)} messages"

        run_test(results, f"chat_conversation() multi-turn  [{conv_model}]", test_chat_conversation, args.verbose)

        # chat_conversation() with streaming
        def test_conv_stream():
            ai[0].select_model(conv_model)
            conv = Conversation(system="Be brief.")
            conv.add_user("Say hello")
            chunks = []
            for chunk in ai[0].chat_conversation(conv, stream=True, auto_update=False):
                chunks.append(chunk)
            assert len(chunks) > 0
            conv.add_assistant("".join(chunks))  # Manual update
            assert len(conv) == 2

        run_test(results, f"chat_conversation() streaming  [{conv_model}]", test_conv_stream)

        # chat_messages() — raw message list
        def test_chat_messages():
            ai[0].select_model(conv_model)
            messages = [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is 1+1?"},
            ]
            response = ai[0].chat_messages(messages)
            assert isinstance(response, ChatResponse)
            assert len(response.content) > 0

        run_test(results, f"chat_messages() raw list  [{conv_model}]", test_chat_messages)

        # chat_messages() with ChatMessage objects
        def test_chat_messages_objects():
            ai[0].select_model(conv_model)
            messages = [
                ChatMessage.system("Be concise."),
                ChatMessage.user("What is 3+3?"),
            ]
            response = ai[0].chat_messages(messages)
            assert isinstance(response, ChatResponse)

        run_test(results, f"chat_messages() with ChatMessage objects  [{conv_model}]", test_chat_messages_objects)

        # ChatMessage factory methods
        def test_chatmessage_factories():
            s = ChatMessage.system("sys")
            assert s.role == "system" and s.content == "sys"
            u = ChatMessage.user("usr")
            assert u.role == "user" and u.content == "usr"
            a = ChatMessage.assistant("ast")
            assert a.role == "assistant" and a.content == "ast"
            d = u.to_dict()
            assert d == {"role": "user", "content": "usr"}

        run_test(results, "ChatMessage factory methods & to_dict()", test_chatmessage_factories)
    else:
        skip_test(results, "Conversation tests", "No chat models available")

    # ════════════════════════════════════════════════════════════════════
    # SECTION 6: EMBEDDINGS & SIMILARITY
    # ════════════════════════════════════════════════════════════════════
    if not args.skip_embeds:
        print(f"\n{C.CYAN}── 6. Embeddings & Similarity ──────────────────────{C.RESET}")

        for emb_model in embed_models:
            # embed() → single vector
            def test_embed_single(m=emb_model):
                ai[0].select_model(m)
                emb = ai[0].embed("hello world")
                assert isinstance(emb, list), f"Expected list, got {type(emb)}"
                assert all(isinstance(v, float) for v in emb[:5])
                assert len(emb) > 100, f"Embedding too small: {len(emb)}"
                return f"dim={len(emb)}"

            run_test(results, f"embed() single text  [{emb_model}]", test_embed_single, args.verbose)

            # embed() → batch
            def test_embed_batch(m=emb_model):
                ai[0].select_model(m)
                embs = ai[0].embed(["cat", "dog", "car"])
                assert isinstance(embs, list) and len(embs) == 3
                assert len(embs[0]) == len(embs[1]) == len(embs[2])
                return f"3 × {len(embs[0])} dims"

            run_test(results, f"embed() batch  [{emb_model}]", test_embed_batch, args.verbose)

            # embed_complete() → EmbeddingResponse
            def test_embed_complete(m=emb_model):
                ai[0].select_model(m)
                resp = ai[0].embed_complete(["alpha", "beta"])
                assert isinstance(resp, EmbeddingResponse)
                assert len(resp) == 2
                assert resp.dimension > 0
                assert resp.model == m
                assert resp.texts == ["alpha", "beta"]
                # Indexing
                v0 = resp[0]
                assert len(v0) == resp.dimension
                # Iteration
                count = sum(1 for _ in resp)
                assert count == 2
                return f"dim={resp.dimension}, tokens={resp.prompt_tokens}"

            run_test(results, f"embed_complete() → EmbeddingResponse  [{emb_model}]", test_embed_complete, args.verbose)

            # cosine_similarity()
            def test_cosine_sim(m=emb_model):
                ai[0].select_model(m)
                e1 = ai[0].embed("happy")
                e2 = ai[0].embed("joyful")
                e3 = ai[0].embed("quantum mechanics")
                sim_close = GenAIStudio.cosine_similarity(e1, e2)
                sim_far = GenAIStudio.cosine_similarity(e1, e3)
                assert -1.0 <= sim_close <= 1.0
                assert -1.0 <= sim_far <= 1.0
                # happy-joyful should generally be more similar
                return f"happy↔joyful={sim_close:.3f}, happy↔quantum={sim_far:.3f}"

            run_test(results, f"cosine_similarity()  [{emb_model}]", test_cosine_sim, args.verbose)

            # similarity() convenience method
            def test_similarity(m=emb_model):
                ai[0].select_model(m)
                sim = ai[0].similarity("dog", "puppy")
                assert isinstance(sim, float)
                assert -1.0 <= sim <= 1.0
                return f"dog↔puppy={sim:.3f}"

            run_test(results, f"similarity() convenience  [{emb_model}]", test_similarity, args.verbose)
    else:
        print(f"\n{C.YELLOW}── 6. Embeddings — SKIPPED ─────────────────────────{C.RESET}")

    # ════════════════════════════════════════════════════════════════════
    # SECTION 7-9: RAG TESTS
    # ════════════════════════════════════════════════════════════════════
    if not args.skip_rag:
        # ── Generate test files ─────────────────────────────────────────
        print(f"\n{C.CYAN}── 7. RAG: File Uploads (multiple types) ───────────{C.RESET}")

        tmpdir = tempfile.mkdtemp(prefix="genai_test_")
        test_files = generate_all_test_files(tmpdir)
        print(f"  {C.DIM}Generated {len(test_files)} test files in {tmpdir}{C.RESET}")

        # Track for cleanup
        uploaded_file_ids = []
        created_kb_ids = []
        file_map = {}  # ext → FileInfo

        # Upload each file type
        for ext, filepath in test_files.items():
            desc = FILE_GENERATORS[ext][0]

            def test_upload(fp=filepath, e=ext, d=desc):
                info = ai[0].upload_file(fp)
                assert isinstance(info, FileInfo)
                assert info.id, "File ID is empty"
                assert info.filename, "Filename is empty"
                file_map[e] = info
                uploaded_file_ids.append(info.id)
                return f"id={info.id[:12]}..."

            run_test(results, f"Upload {desc} ({ext})", test_upload, args.verbose)

        # Upload nonexistent file
        def test_upload_missing():
            try:
                ai[0].upload_file("/tmp/this_does_not_exist_xyz.txt")
                assert False, "Should raise FileNotFoundError"
            except FileNotFoundError:
                pass

        run_test(results, "Upload nonexistent → FileNotFoundError", test_upload_missing)

        # list_files()
        def test_list_files():
            files = ai[0].list_files()
            assert isinstance(files, list)
            ids = [f.id for f in files]
            for fid in uploaded_file_ids[:3]:
                assert fid in ids, f"Uploaded file {fid[:12]}... not in list"
            return f"{len(files)} files"

        run_test(results, "list_files() contains uploaded files", test_list_files, args.verbose)

        # ── Knowledge Base CRUD ─────────────────────────────────────────
        print(f"\n{C.CYAN}── 8. RAG: Knowledge Base CRUD ─────────────────────{C.RESET}")

        time.sleep(2)  # Let uploads settle

        # Create KB for text-based files
        text_kb = [None]

        def test_create_text_kb():
            kb = ai[0].create_knowledge_base("Test Text KB", "Text-based files")
            assert isinstance(kb, KnowledgeBase)
            assert kb.id and kb.name == "Test Text KB"
            text_kb[0] = kb
            created_kb_ids.append(kb.id)
            return f"id={kb.id[:12]}..."

        run_test(results, "Create text knowledge base", test_create_text_kb, args.verbose)

        # Create KB for structured files
        struct_kb = [None]

        def test_create_struct_kb():
            kb = ai[0].create_knowledge_base("Test Structured KB", "CSV/JSON/XLSX files")
            assert isinstance(kb, KnowledgeBase)
            struct_kb[0] = kb
            created_kb_ids.append(kb.id)
            return f"id={kb.id[:12]}..."

        run_test(results, "Create structured data KB", test_create_struct_kb, args.verbose)

        # list_knowledge_bases()
        def test_list_kbs():
            kbs = ai[0].list_knowledge_bases()
            assert isinstance(kbs, list)
            ids = [kb.id for kb in kbs]
            if text_kb[0]:
                assert text_kb[0].id in ids
            return f"{len(kbs)} KBs"

        run_test(results, "list_knowledge_bases()", test_list_kbs, args.verbose)

        # get_knowledge_base()
        def test_get_kb():
            if not text_kb[0]:
                raise AssertionError("KB not created")
            kb = ai[0].get_knowledge_base(text_kb[0].id)
            assert kb.id == text_kb[0].id
            assert kb.name == "Test Text KB"

        run_test(results, "get_knowledge_base() by ID", test_get_kb)

        # ── Link files ──────────────────────────────────────────────────
        text_exts = [".txt", ".md", ".py", ".html", ".pdf"]
        struct_exts = [".csv", ".json", ".xlsx"]

        for ext in text_exts:
            if ext in file_map and text_kb[0]:
                def test_link(e=ext):
                    ai[0].add_file_to_knowledge_base(text_kb[0].id, file_map[e].id)
                run_test(results, f"Link {ext} → Text KB", test_link)

        for ext in struct_exts:
            if ext in file_map and struct_kb[0]:
                def test_link(e=ext):
                    ai[0].add_file_to_knowledge_base(struct_kb[0].id, file_map[e].id)
                run_test(results, f"Link {ext} → Structured KB", test_link)

        # Unlink and re-link (test remove/add cycle)
        if ".md" in file_map and text_kb[0]:
            def test_unlink_relink():
                ai[0].remove_file_from_knowledge_base(text_kb[0].id, file_map[".md"].id)
                time.sleep(1)
                ai[0].add_file_to_knowledge_base(text_kb[0].id, file_map[".md"].id)

            run_test(results, "Unlink → re-link cycle (.md)", test_unlink_relink)

        # ── Grounded Queries ────────────────────────────────────────────
        print(f"\n{C.CYAN}── 9. RAG: Grounded Queries ────────────────────────{C.RESET}")

        # Wait for indexing
        print(f"  {C.DIM}Waiting {args.wait}s for indexing", end="", flush=True)
        for _ in range(args.wait):
            time.sleep(1)
            print(".", end="", flush=True)
        print(f"{C.RESET}")

        # Test retrieval across models and KB types
        rag_chat_models = chat_models[:2]  # Limit to avoid very long runs

        for model_id in rag_chat_models:
            # Query text KB for primary secret
            if text_kb[0]:
                def test_rag_text(m=model_id):
                    ai[0].select_model(m)
                    resp = ai[0].chat(SECRET_QUESTION, collections=[text_kb[0].id])
                    assert SECRET in resp, f"Secret not found: {resp[:150]}"
                    return f"Found '{SECRET}'"

                run_test(results, f"RAG text KB → secret  [{model_id}]", test_rag_text, args.verbose)

            # Query text KB for second fact
            if text_kb[0]:
                def test_rag_second(m=model_id):
                    ai[0].select_model(m)
                    resp = ai[0].chat(SECOND_QUESTION, collections=[text_kb[0].id])
                    assert SECOND_FACT in resp.upper(), f"Fact not found: {resp[:150]}"
                    return f"Found '{SECOND_FACT}'"

                run_test(results, f"RAG text KB → second fact  [{model_id}]", test_rag_second, args.verbose)

            # Query structured KB
            if struct_kb[0]:
                def test_rag_struct(m=model_id):
                    ai[0].select_model(m)
                    resp = ai[0].chat(SECRET_QUESTION, collections=[struct_kb[0].id])
                    assert SECRET in resp, f"Secret not found in structured: {resp[:150]}"
                    return f"Found '{SECRET}'"

                run_test(results, f"RAG structured KB → secret  [{model_id}]", test_rag_struct, args.verbose)

            # RAG + system prompt
            if text_kb[0]:
                def test_rag_system(m=model_id):
                    ai[0].select_model(m)
                    resp = ai[0].chat(
                        SECRET_QUESTION,
                        system="Answer in one sentence.",
                        collections=[text_kb[0].id]
                    )
                    assert SECRET in resp

                run_test(results, f"RAG + system prompt  [{model_id}]", test_rag_system)

            # RAG via chat_complete()
            if text_kb[0]:
                def test_rag_complete(m=model_id):
                    ai[0].select_model(m)
                    resp = ai[0].chat_complete(SECRET_QUESTION, collections=[text_kb[0].id])
                    assert isinstance(resp, ChatResponse)
                    assert SECRET in resp.content

                run_test(results, f"RAG via chat_complete()  [{model_id}]", test_rag_complete)

            # RAG via chat_stream()
            if text_kb[0]:
                def test_rag_stream(m=model_id):
                    ai[0].select_model(m)
                    chunks = list(ai[0].chat_stream(SECRET_QUESTION, collections=[text_kb[0].id]))
                    full = "".join(chunks)
                    assert SECRET in full, f"Secret not in stream: {full[:150]}"
                    return f"{len(chunks)} chunks"

                run_test(results, f"RAG via chat_stream()  [{model_id}]", test_rag_stream, args.verbose)

            # RAG via chat_conversation() multi-turn
            if text_kb[0]:
                def test_rag_conv(m=model_id):
                    ai[0].select_model(m)
                    conv = Conversation(system="Answer based on the documents.")
                    conv.add_user(SECRET_QUESTION)
                    resp = ai[0].chat_conversation(conv, collections=[text_kb[0].id])
                    assert SECRET in resp.content
                    assert len(conv) == 2
                    # Follow-up
                    conv.add_user("What about the badge color?")
                    resp2 = ai[0].chat_conversation(conv, collections=[text_kb[0].id])
                    assert len(conv) == 4
                    return f"{len(conv)} msgs"

                run_test(results, f"RAG via chat_conversation()  [{model_id}]", test_rag_conv, args.verbose)

            # RAG via chat_messages()
            if text_kb[0]:
                def test_rag_messages(m=model_id):
                    ai[0].select_model(m)
                    msgs = [{"role": "user", "content": SECRET_QUESTION}]
                    resp = ai[0].chat_messages(msgs, collections=[text_kb[0].id])
                    assert isinstance(resp, ChatResponse)
                    assert SECRET in resp.content

                run_test(results, f"RAG via chat_messages()  [{model_id}]", test_rag_messages)

            # Query both KBs simultaneously
            if text_kb[0] and struct_kb[0]:
                def test_rag_multi_kb(m=model_id):
                    ai[0].select_model(m)
                    resp = ai[0].chat(
                        SECRET_QUESTION,
                        collections=[text_kb[0].id, struct_kb[0].id]
                    )
                    assert SECRET in resp

                run_test(results, f"RAG multi-KB query  [{model_id}]", test_rag_multi_kb)

        # ── RAG Cleanup Tests ───────────────────────────────────────────
        # Delete struct KB (tested explicitly)
        if struct_kb[0]:
            def test_delete_kb():
                ai[0].delete_knowledge_base(struct_kb[0].id)
                created_kb_ids.remove(struct_kb[0].id)

            run_test(results, "delete_knowledge_base()", test_delete_kb)

            def test_get_deleted_kb():
                try:
                    ai[0].get_knowledge_base(struct_kb[0].id)
                    assert False, "Should raise RAGError"
                except RAGError:
                    pass

            run_test(results, "Get deleted KB → RAGError", test_get_deleted_kb)

        # Delete one file explicitly
        if ".png" in file_map:
            def test_delete_file():
                ai[0].delete_file(file_map[".png"].id)
                if file_map[".png"].id in uploaded_file_ids:
                    uploaded_file_ids.remove(file_map[".png"].id)

            run_test(results, "delete_file()", test_delete_file)

        # Final cleanup
        if not args.keep:
            print(f"\n  {C.DIM}Cleaning up RAG resources...{C.RESET}")
            for kb_id in created_kb_ids[:]:
                try:
                    ai[0].delete_knowledge_base(kb_id)
                    created_kb_ids.remove(kb_id)
                except Exception:
                    pass
            for fid in uploaded_file_ids[:]:
                try:
                    ai[0].delete_file(fid)
                    uploaded_file_ids.remove(fid)
                except Exception:
                    pass
            # Cleanup temp files
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"  {C.DIM}Cleanup done{C.RESET}")
        else:
            print(f"\n  {C.YELLOW}⏭️  Keeping RAG resources (--keep){C.RESET}")
            if created_kb_ids:
                print(f"  KB IDs:   {', '.join(created_kb_ids)}")
            if uploaded_file_ids:
                print(f"  File IDs: {', '.join(uploaded_file_ids)}")
    else:
        print(f"\n{C.YELLOW}── 7-9. RAG Tests — SKIPPED ────────────────────────{C.RESET}")

    # ════════════════════════════════════════════════════════════════════
    # SECTION 10: ERROR HANDLING & EDGE CASES
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{C.CYAN}── 10. Error Handling & Edge Cases ─────────────────{C.RESET}")

    # Exception hierarchy
    def test_exception_hierarchy():
        assert issubclass(AuthenticationError, GenAIStudioError)
        assert issubclass(ModelNotFoundError, GenAIStudioError)
        assert issubclass(GSConnectionError, GenAIStudioError)
        assert issubclass(GSTimeoutError, GenAIStudioError)
        assert issubclass(RAGError, GenAIStudioError)

    run_test(results, "Exception hierarchy correct", test_exception_hierarchy)

    # Catch-all works
    def test_catch_all():
        try:
            raise RAGError("test")
        except GenAIStudioError:
            pass  # Should catch

    run_test(results, "GenAIStudioError catches all subtypes", test_catch_all)

    # Empty prompt
    if chat_models:
        def test_empty_prompt():
            ai[0].select_model(chat_models[0])
            resp = ai[0].chat("")  # Server should handle gracefully
            assert isinstance(resp, str)

        run_test(results, "Empty prompt doesn't crash", test_empty_prompt)

    # Long prompt
    if chat_models:
        def test_long_prompt():
            ai[0].select_model(chat_models[0])
            long = "Hello " * 500  # ~3000 chars
            resp = ai[0].chat(long, max_tokens=20)
            assert isinstance(resp, str)

        run_test(results, "Long prompt (~3000 chars)", test_long_prompt)

    # _build_rag_extra_body static method
    def test_build_rag_extra_body():
        # None in, None out
        assert GenAIStudio._build_rag_extra_body(None, None) is None

        # Collections only
        result = GenAIStudio._build_rag_extra_body(["abc", "def"])
        assert result == {"files": [
            {"type": "collection", "id": "abc"},
            {"type": "collection", "id": "def"}
        ]}

        # Merge with existing extra_body
        result = GenAIStudio._build_rag_extra_body(
            ["abc"],
            {"temperature": 0.5}
        )
        assert result["temperature"] == 0.5
        assert result["files"] == [{"type": "collection", "id": "abc"}]

        # Extra body only, no collections
        result = GenAIStudio._build_rag_extra_body(None, {"key": "val"})
        assert result == {"key": "val"}

    run_test(results, "_build_rag_extra_body() static method", test_build_rag_extra_body)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 11: HEALTH CHECK & UTILITIES
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{C.CYAN}── 11. Health Check & Utilities ────────────────────{C.RESET}")

    def test_health_check():
        result = ai[0].health_check()
        assert isinstance(result, dict)
        assert result["status"] == "connected", f"Status: {result['status']}, error: {result.get('error')}"
        assert result["model_count"] > 0
        assert result["base_url"] == DEFAULT_BASE_URL
        return f"{result['model_count']} models"

    run_test(results, "health_check() returns connected", test_health_check, args.verbose)

    # ════════════════════════════════════════════════════════════════════
    # SECTION 12: DATACLASS VALIDATION
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{C.CYAN}── 12. Dataclass Validation ────────────────────────{C.RESET}")

    def test_fileinfo_dataclass():
        fi = FileInfo(id="abc", filename="test.txt", meta={"size": 100})
        assert fi.id == "abc"
        assert fi.filename == "test.txt"
        assert fi.meta == {"size": 100}
        assert fi.raw_response == {}
        assert "abc" in repr(fi) and "test.txt" in repr(fi)

    run_test(results, "FileInfo dataclass", test_fileinfo_dataclass)

    def test_kb_dataclass():
        kb = KnowledgeBase(id="xyz", name="Test", description="Desc")
        assert kb.id == "xyz"
        assert kb.name == "Test"
        assert kb.description == "Desc"
        assert kb.raw_response == {}
        assert "xyz" in repr(kb) and "Test" in repr(kb)

    run_test(results, "KnowledgeBase dataclass", test_kb_dataclass)

    def test_chatresponse_dataclass():
        cr = ChatResponse(
            content="Hello", model="test-model", finish_reason="stop",
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        assert cr.content == "Hello"
        assert cr.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    run_test(results, "ChatResponse dataclass & usage", test_chatresponse_dataclass)

    def test_embeddingresponse_dataclass():
        er = EmbeddingResponse(
            embeddings=[[1.0, 2.0], [3.0, 4.0]],
            texts=["a", "b"], model="test", dimension=2
        )
        assert len(er) == 2
        assert er[0] == [1.0, 2.0]
        assert er[1] == [3.0, 4.0]
        assert list(er) == [[1.0, 2.0], [3.0, 4.0]]

    run_test(results, "EmbeddingResponse indexing/iteration", test_embeddingresponse_dataclass)

    def test_constants():
        assert __version__ is not None and len(__version__) > 0
        assert DEFAULT_BASE_URL.startswith("https://")
        assert DEFAULT_TIMEOUT > 0
        assert DEFAULT_CONNECT_TIMEOUT > 0
        assert ENV_API_KEY == "GENAI_STUDIO_API_KEY"
        return f"v{__version__}"

    run_test(results, "Module constants valid", test_constants, args.verbose)

    # ════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════════
    all_passed = results.summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())