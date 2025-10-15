import argparse
import json
import sys
import time
import multiprocessing as mp

def _runner(payload, conn):
    try:
        from src.cli import run_calculation
        import sys, io
        # Capture stdout/stderr from the child run
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        _buf_out, _buf_err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = _buf_out, _buf_err
        class A: pass
        args = A()
        for k, v in payload.items():
            setattr(args, k, v)
        res = run_calculation(args)
        # Flush and restore stdout/stderr
        sys.stdout.flush(); sys.stderr.flush()
        out_text, err_text = _buf_out.getvalue(), _buf_err.getvalue()
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        conn.send({"ok": True, "res": res, "stdout": out_text, "stderr": err_text})
    except Exception as e:
        try:
            out_text = _buf_out.getvalue() if '_buf_out' in locals() else ""
            err_text = _buf_err.getvalue() if '_buf_err' in locals() else ""
        except Exception:
            out_text, err_text = "", ""
        conn.send({"ok": False, "err": repr(e), "stdout": out_text, "stderr": err_text})
    finally:
        conn.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--payload", type=str, required=True)
    ns = ap.parse_args()

    parent, child = mp.Pipe()
    p = mp.Process(target=_runner, args=(json.loads(ns.payload), child))
    p.start()
    start = time.time()
    while time.time() - start < ns.timeout and p.is_alive():
        if parent.poll(0.05):
            break
        time.sleep(0.05)
    out = None
    if parent.poll():
                out = parent.recv()
    if p.is_alive():
        p.terminate()
        p.join(timeout=1.0)
        print(json.dumps({"ok": False, "timeout": True}), flush=True)
        sys.exit(124)
        # First, forward child's stdout (if any) so callers can grep it
        if out and isinstance(out, dict) and out.get("stdout"):
            # Ensure trailing newline
            sys.stdout.write(out.get("stdout"))
            if not out.get("stdout").endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
        # Then emit JSON summary for programmatic inspection
        print(json.dumps(out if out is not None else {"ok": False}), flush=True)
    sys.exit(0 if (out and out.get("ok")) else 1)
