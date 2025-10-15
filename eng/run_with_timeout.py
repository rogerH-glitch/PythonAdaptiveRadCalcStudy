import argparse
import json
import sys
import time
import multiprocessing as mp

def _runner(payload, conn):
    try:
        from src.cli import run_calculation
        class A: pass
        args = A()
        for k, v in payload.items():
            setattr(args, k, v)
        res = run_calculation(args)
        conn.send({"ok": True, "res": res})
    except Exception as e:
        conn.send({"ok": False, "err": repr(e)})
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
    print(json.dumps(out if out is not None else {"ok": False}), flush=True)
    sys.exit(0 if (out and out.get("ok")) else 1)
