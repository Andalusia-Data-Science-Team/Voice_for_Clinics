import asyncio
import time
import json
import statistics
import random
from config import client, DEPLOYMENTS, budget_manager

# ~2500 tokens input
LARGE_INPUT = "Medical history and background. " * 500

async def make_request(deployment, attempt_backoff=False):
    start = time.perf_counter()
    attempts = 0
    max_attempts = 4 if attempt_backoff else 1
    
    while attempts < max_attempts:
        attempts += 1
        try:
            response = await client.chat.completions.with_raw_response.create(
                model=deployment,
                messages=[{"role": "user", "content": LARGE_INPUT}],
                max_tokens=1500
            )
            parsed = response.parse()
            budget_manager.add_cost(deployment, parsed.usage.prompt_tokens, parsed.usage.completion_tokens)
            return {
                "status": 200,
                "latency": time.perf_counter() - start,
                "retry_after": 0
            }
        except Exception as e:
            err_str = str(e)
            retry_after = 0
            
            # Extract retry-after from error message or headers if possible
            # Azure usually returns 429 for rate limits
            if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                retry_after = int(e.response.headers.get("retry-after", 0))
                
            if "429" in err_str:
                status = 429
            elif "50" in err_str or "timeout" in err_str.lower():
                status = 500
            else:
                return {"status": 400, "latency": time.perf_counter() - start, "error": err_str}
                
            if not attempt_backoff or attempts == max_attempts:
                return {"status": status, "latency": time.perf_counter() - start, "retry_after": retry_after}
                
            # Exponential backoff with full jitter honoring Retry-After
            base_delay = 2 ** attempts
            sleep_time = random.uniform(0, base_delay)
            if retry_after > 0:
                sleep_time = max(sleep_time, retry_after)
            await asyncio.sleep(sleep_time)

async def run_ramp(deployment, concurrent_reqs, attempt_backoff=False):
    print(f"    Concurrency: {concurrent_reqs} (Backoff={attempt_backoff})")
    start = time.perf_counter()
    
    tasks = [make_request(deployment, attempt_backoff) for _ in range(concurrent_reqs)]
    results = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start
    
    latencies = [r["latency"] for r in results if r["status"] == 200]
    errors_429 = sum(1 for r in results if r["status"] == 429)
    other_errors = sum(1 for r in results if r["status"] not in (200, 429))
    retry_afters = [r["retry_after"] for r in results if r.get("retry_after", 0) > 0]
    
    p50 = statistics.median(latencies) if latencies else 0
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0)
    p99 = max(latencies) if latencies else 0  # roughly for small sizes
    
    print(f"      Success: {len(latencies)} | 429s: {errors_429} | Other errs: {other_errors}")
    if latencies:
        print(f"      p50: {p50:.2f}s | p95: {p95:.2f}s | p99: {p99:.2f}s | Throughput: {len(latencies)/total_time:.1f} req/s")
    if retry_afters:
        print(f"      Retry-after values: {set(retry_afters)}")
        
    return {
        "concurrency": concurrent_reqs,
        "backoff": attempt_backoff,
        "success": len(latencies),
        "errors_429": errors_429,
        "other_errors": other_errors,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "throughput": len(latencies) / total_time if total_time > 0 else 0,
        "retry_afters": list(set(retry_afters))
    }

async def run():
    print("--- Concurrency & Rate Limits ---")
    results = {}
    
    ramps = [1, 5, 10, 20, 30, 50]
    
    for deployment in DEPLOYMENTS:
        print(f"\nTesting {deployment}...")
        results[deployment] = []
        
        deployment_ramps = [1, 5, 10] if 'sol' in deployment.lower() else ramps
        
        for c in deployment_ramps:
            res = await run_ramp(deployment, c, False)
            results[deployment].append(res)
            
        # Backoff tests for 30, 50
        if 'sol' not in deployment.lower():
            for c in [30, 50]:
                res = await run_ramp(deployment, c, True)
                results[deployment].append(res)

    with open("results/concurrency.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    asyncio.run(run())
