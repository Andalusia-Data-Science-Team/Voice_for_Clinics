import asyncio
import json
from config import client, DEPLOYMENTS, budget_manager

async def test_metadata(deployment):
    print(f"\nFetching metadata for {deployment}...")
    try:
        raw_response = await client.chat.completions.with_raw_response.create(
            model=deployment,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=2
        )
        
        parsed = raw_response.parse()
        if hasattr(parsed, 'usage') and parsed.usage:
            budget_manager.add_cost(deployment, parsed.usage.prompt_tokens, parsed.usage.completion_tokens)
            
        headers = raw_response.headers
        
        metadata = {
            "status_code": raw_response.status_code,
            "x-ms-region": headers.get("x-ms-region", "NOT_PRESENT"),
            "x-request-id": headers.get("x-request-id", "NOT_PRESENT"),
            "apim-request-id": headers.get("apim-request-id", "NOT_PRESENT"),
            "retry-after": headers.get("retry-after", "NOT_PRESENT"),
            "ratelimit-remaining-tokens": headers.get("x-ratelimit-remaining-tokens", "NOT_PRESENT"),
            "ratelimit-remaining-requests": headers.get("x-ratelimit-remaining-requests", "NOT_PRESENT"),
        }
        
        print(f"Region: {metadata['x-ms-region']}")
        return metadata
    except Exception as e:
        print(f"Failed to fetch metadata for {deployment}: {e}")
        return {"error": str(e)}

async def run():
    print("--- Metadata & Region Mapping ---")
    results = {}
    
    for deployment in DEPLOYMENTS:
        metadata = await test_metadata(deployment)
        results[deployment] = metadata
            
    with open("results/metadata.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    asyncio.run(run())
