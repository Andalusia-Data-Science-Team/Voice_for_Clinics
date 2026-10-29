# A6 — post-fix verification artifacts

Produced after fixing three bugs (see `docs/FINAL_REPORT.md` §4.5, A6):
1. UnboundLocalError in pipeline finally (server→SSE drop).
2. CancelledError counted as circuit-breaker failure.
3. (same as 2, defense-in-depth) CancelledError in record_failure.

| File | Contents |
| --- | --- |
| step_ar_u{1,5,10,20,50}_*.csv | AR ramp on fixed code — 0 failures at every step |
| out_ar_u{1,5,10,20,50}.log | per-step console output |
| server_postfix.log | server log showing 0 UnboundLocalError, 0 ASGI errors |
| repro_client_disconnect.py | reproducer: drop connection mid-translation |
| burst_pre_fix_7777.txt | pre-fix AR=50 failure capture |
| burst_pre_fix_ar50.txt | pre-fix ChunkedEncodingError capture |

Note: a *second* sustained AR=50 run tripped the breaker once on a genuine
DeepSeek "Request timed out" (5 real backend timeouts) — normal circuit-breaker
behavior, not the disconnect bug.
