"""Reproduce the client-disconnect-during-translation scenario.

Fires N concurrent AR analyze requests, then drops the connection as soon as
the stream is mid-translation (before extraction), forcing generator
cancellation while `reasoning_spec` is still unset.
"""
from __future__ import annotations

import concurrent.futures as cf
import sys
import time

import requests

HOST = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TEXT = "زيارة اختبارية اصطناعية. صداع خفيف لمدة يومين. العلامات الحيوية مستقرة."


def one(i: int) -> str:
    try:
        r = requests.post(
            HOST + "/audio/analyze",
            data={
                "text": TEXT, "language": "ar", "mode": "doctor",
                "department": "opd", "use_history": "false",
                "session_id": f"abort-{i}", "clinician_name": "abort",
            },
            stream=True, timeout=60,
        )
        # Drop the connection as soon as translation starts (mid-stream).
        for raw in r.iter_lines(decode_unicode=False):
            line = (raw or b"").decode("utf-8", errors="replace")
            if "translation_delta" in line:
                break
        r.close()  # abrupt client-side disconnect
        return "DROPPED"
    except Exception as exc:  # noqa: BLE001
        return f"EXC {type(exc).__name__}"
    finally:
        time.sleep(0)


with cf.ThreadPoolExecutor(max_workers=N) as ex:
    res = list(ex.map(one, range(N)))
from collections import Counter  # noqa: E402
print(dict(Counter(res)))
