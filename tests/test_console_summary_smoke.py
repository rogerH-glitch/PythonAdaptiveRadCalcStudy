import subprocess, sys


def test_console_summary_has_key_sections():
    cmd = [
        sys.executable, "main.py",
        "--method", "adaptive",
        "--emitter", "5.1", "2.1",
        "--receiver", "5.1", "2.1",
        "--setback", "3",
        "--angle", "30",
        "--receiver-offset", "0.25", "0",
        "--eval-mode", "grid",
        "--plot", "False",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    out = proc.stdout
    for tag in ("[eval]", "[geom]", "[peak]", "[status]"):
        assert out.count(tag) == 1, f"Expected exactly one {tag} line, got {out.count(tag)}.\nSTDOUT:\n{out}"
    if "[grid]" in out:
        assert out.count("[grid]") == 1, f"Expected at most one [grid] line.\nSTDOUT:\n{out}"
    if "[artifacts]" in out:
        assert out.count("[artifacts]") == 1, f"Expected at most one [artifacts] line.\nSTDOUT:\n{out}"


