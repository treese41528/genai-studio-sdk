"""Tests for the hardened, subprocess-isolated python_exec (make_sandboxed_python_exec).

These spawn real subprocesses with OS resource limits, so they are Unix-gated and a
touch slower than the pure-unit tests (the CPU-limit case takes ~1s).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="sandbox needs Unix (os.fork + resource + preexec_fn)")

from genai_studio.agents.datascience.tools import make_sandboxed_python_exec


def test_basic_exec_and_state_persistence():
    px = make_sandboxed_python_exec()
    assert px.run({"code": "x = 21"}).error is None
    res = px.run({"code": "print(x * 2)"})      # x persists across separate processes
    assert res.error is None and "42" in res.content


def test_trailing_expression_is_summarised():
    res = make_sandboxed_python_exec().run({"code": "2 + 3"})
    assert res.error is None and "5" in res.content


def test_cpu_or_wall_limit_is_enforced():
    res = make_sandboxed_python_exec(cpu_s=1, wall_s=5).run({"code": "while True:\n    pass"})
    assert res.error is not None and "limit" in res.error.lower()


def test_memory_limit_is_enforced():
    res = make_sandboxed_python_exec(mem_mb=512).run(
        {"code": "b = bytearray(1500 * 1024 * 1024)"})   # 1.5 GB > 512 MB cap
    assert res.error is not None


def test_network_blocked_by_default():
    res = make_sandboxed_python_exec().run({"code": "import socket; socket.socket()"})
    assert res.error is not None and "network" in res.error.lower()


def test_allow_network_permits_socket_object():
    # creating + closing a socket (no actual connection) is allowed when enabled
    res = make_sandboxed_python_exec(allow_network=True).run(
        {"code": "import socket; s = socket.socket(); s.close(); print('ok')"})
    assert res.error is None and "ok" in res.content


def test_approve_fn_rejects_before_running():
    def gate(code):
        return "no imports allowed" if "import os" in code else True

    px = make_sandboxed_python_exec(approve_fn=gate)
    res = px.run({"code": "import os; os.system('echo hi')"})
    assert res.error and "no imports allowed" in res.error
    assert px.run({"code": "print(1 + 1)"}).error is None   # allowed snippet runs


def test_unpicklable_var_dropped_but_picklable_persists():
    px = make_sandboxed_python_exec()
    res1 = px.run({"code": "f = lambda v: v\ny = 7"})
    assert "not retained" in res1.content and "f" in res1.content   # lambda dropped
    assert "7" in px.run({"code": "print(y)"}).content             # y persisted


def test_error_is_fed_back_not_raised():
    res = make_sandboxed_python_exec().run({"code": "1 / 0"})
    assert res.error is not None and "ZeroDivisionError" in res.error


def test_approve_fn_empty_string_is_treated_as_allow():
    # an empty string is NOT a rejection (footgun fix) — the code runs
    res = make_sandboxed_python_exec(approve_fn=lambda code: "").run({"code": "print(2 + 2)"})
    assert res.error is None and "4" in res.content


def test_spawned_child_does_not_outlive_wall_clock():
    # a child process + a long sleep: the wall-clock timeout SIGKILLs the WHOLE group,
    # so the call returns promptly with a clear error (no 30s hang on the child).
    import time
    px = make_sandboxed_python_exec(wall_s=3)
    t0 = time.time()
    res = px.run({"code": "import subprocess, time\n"
                          "subprocess.Popen(['sleep', '30'])\n"
                          "time.sleep(30)"})
    elapsed = time.time() - t0
    assert res.error is not None and "wall-clock" in res.error
    assert elapsed < 12  # returned near wall_s=3, not the 30s sleep


def test_max_file_mb_enforced(tmp_path):
    target = tmp_path / "big.bin"
    res = make_sandboxed_python_exec(max_file_mb=5).run(
        {"code": f"open({str(target)!r}, 'wb').write(b'0' * (50 * 1024 * 1024))"})  # 50MB > 5MB
    assert res.error is not None


def test_data_is_none_across_process_boundary():
    # documented drop-in limitation: the actual object can't cross back, only its summary
    res = make_sandboxed_python_exec().run({"code": "[1, 2, 3]"})
    assert res.error is None and res.data is None and "[1, 2, 3]" in res.content
