import subprocess, sys


def test_console_summary_smoke():
    payload = '{"method":"adaptive","emitter":[5.1,2.1],"receiver":[5.1,2.1],"setback":3.0,' \
              '"angle":30.0,"rotate_axis":"z","rotate_target":"emitter","angle_pivot":"toe",' \
              '"align_centres":false,"receiver_offset":[0.25,0.0],"emitter_offset":[0.0,0.0],' \
              '"plot":false,"eval_mode":"grid","rc_mode":"grid","rc_grid_n":21,' \
              '"rc_search_time_limit_s":1.5,"heatmap_n":41}'
    cmd = [sys.executable, "eng/run_with_timeout.py", "--timeout", "15", "--payload", payload]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    assert "[eval]" in out and "[peak]" in out and "[status]" in out


