import asyncio
import json
from config import client, DEPLOYMENTS, budget_manager

async def test_capability(deployment, name, kwargs, assert_content=None):
    try:
        response = await client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Say 'OK'"}],
            **kwargs
        )
        # Cost tracking for capability tests (they are very small, but still)
        if hasattr(response, 'usage') and response.usage:
            budget_manager.add_cost(deployment, response.usage.prompt_tokens, response.usage.completion_tokens)
            
        if assert_content:
            return "Supported"
        
        # Check stream
        if kwargs.get('stream'):
            async for chunk in response:
                pass
        return "Supported"
    except Exception as e:
        return f"Unsupported: {e.__class__.__name__} - {str(e)}"

async def run():
    print("--- Capability Matrix ---")
    results = {}
    
    # 20k token input string (approx 20k words, roughly 25k tokens, we will use a safe 100k chars ~25k tokens)
    long_input = "hello " * 20000 
    
    for deployment in DEPLOYMENTS:
        print(f"\nTesting {deployment}...")
        results[deployment] = {}
        
        tests = {
            "Basic Chat": {"max_tokens": 5},
            "Streaming": {"max_tokens": 5, "stream": True},
            "JSON Object": {"max_tokens": 15, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": "You output JSON."}, {"role": "user", "content": "Output {}"}]},
            "JSON Schema (Strict)": {
                "messages": [{"role": "user", "content": "Output a test"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "test",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"val": {"type": "string"}},
                            "required": ["val"],
                            "additionalProperties": False
                        }
                    }
                }
            },
            "System Role": {"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}], "max_tokens": 5},
            "Developer Role": {"messages": [{"role": "developer", "content": "You are helpful."}, {"role": "user", "content": "hi"}], "max_tokens": 5},
            "Temperature/Top_P": {"temperature": 0.5, "top_p": 0.9, "max_tokens": 5},
            "Max Tokens": {"max_tokens": 5},
            "Max Completion Tokens": {"max_completion_tokens": 5},
            "Reasoning Effort": {"reasoning_effort": "low"},
            "20k Context Input": {"messages": [{"role": "user", "content": f"Repeat 'OK'. {long_input}"}], "max_tokens": 2},
            "Arabic I/O": {"messages": [{"role": "user", "content": "رد بكلمة 'مرحبا' فقط"}], "max_tokens": 10},
        }
        
        for test_name, test_kwargs in tests.items():
            # Create a copy to prevent mutating the original templates
            kwargs = test_kwargs.copy()
            # Default messages if not overridden
            if "messages" not in kwargs:
                kwargs["messages"] = [{"role": "user", "content": "Say 'OK'"}]
                
            res = await test_capability(deployment, test_name, kwargs)
            results[deployment][test_name] = res
            print(f"  {test_name}: {res[:100]}..." if len(res) > 100 else f"  {test_name}: {res}")
            
    with open("results/capability_matrix.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    asyncio.run(run())
