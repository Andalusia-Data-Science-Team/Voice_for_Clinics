import asyncio
import time
import json
import statistics
from config import client, DEPLOYMENTS, budget_manager

REALISTIC_TEXT = "Patient presents with severe headache for 3 days. No fever. " * 50  # Roughly 500 words, ~600 tokens

async def simulate_structuring(deployment):
    start = time.perf_counter()
    ttft = None
    full_content = ""
    
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Structure this text into History, Examination, Plan."},
            {"role": "user", "content": REALISTIC_TEXT}
        ],
        stream=True,
        max_tokens=300
    )
    
    async for chunk in response:
        if ttft is None:
            ttft = time.perf_counter() - start
        if chunk.choices and chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content
            
    # We estimate tokens for streaming since usage isn't always returned on Azure streams
    # roughly prompt=600, completion=300
    budget_manager.add_cost(deployment, 600, 300)
    
    return ttft, full_content

async def simulate_extraction(deployment, structured_text):
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Extract entities into JSON"},
            {"role": "user", "content": structured_text}
        ],
        response_format={"type": "json_object"},
        max_tokens=150
    )
    budget_manager.add_cost(deployment, response.usage.prompt_tokens, response.usage.completion_tokens)
    return time.perf_counter() - start

async def simulate_reasoning(deployment, structured_text):
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Reason about diagnosis into JSON"},
            {"role": "user", "content": structured_text}
        ],
        response_format={"type": "json_object"},
        max_tokens=250
    )
    budget_manager.add_cost(deployment, response.usage.prompt_tokens, response.usage.completion_tokens)
    return time.perf_counter() - start

async def run_pipeline(deployment):
    start_total = time.perf_counter()
    
    # 1. Structuring (streamed)
    ttft, structured_text = await simulate_structuring(deployment)
    
    # 2. Parallel Extraction and Reasoning
    await asyncio.gather(
        simulate_extraction(deployment, structured_text),
        simulate_reasoning(deployment, structured_text)
    )
    
    total_time = time.perf_counter() - start_total
    return ttft, total_time

async def run():
    print("--- Pipeline Latency ---")
    results = {}
    
    for deployment in DEPLOYMENTS:
        print(f"\nTesting {deployment} (20 runs)...")
        results[deployment] = {"ttft": [], "total_time": []}
        
        for i in range(20):
            try:
                ttft, total = await run_pipeline(deployment)
                results[deployment]["ttft"].append(ttft)
                results[deployment]["total_time"].append(total)
            except Exception as e:
                print(f"  Run {i+1} failed: {e}")
                
        if results[deployment]["total_time"]:
            totals = results[deployment]["total_time"]
            ttfts = results[deployment]["ttft"]
            p50 = statistics.median(totals)
            p95 = statistics.quantiles(totals, n=20)[18] if len(totals) >= 20 else max(totals)
            
            p50_ttft = statistics.median(ttfts)
            print(f"  TTFT p50: {p50_ttft:.2f}s | Total p50: {p50:.2f}s | Total p95: {p95:.2f}s")

    with open("results/latency.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    asyncio.run(run())
