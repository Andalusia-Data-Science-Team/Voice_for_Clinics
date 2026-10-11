import json
import yaml
from config import DEPLOYMENTS, prices, budget_manager

def calculate_cost(model, input_tokens, output_tokens):
    model_prices = prices.get(model, {"input_1m": 0.0, "output_1m": 0.0})
    cost = (input_tokens / 1_000_000) * model_prices.get("input_1m", 0) + \
           (output_tokens / 1_000_000) * model_prices.get("output_1m", 0)
    return cost

def run():
    print("--- Cost Analysis ---")
    results = {}
    
    # Workload estimates per encounter
    structuring_tokens = {"in": 6000, "out": 2500}
    extraction_tokens = {"in": 3000, "out": 700}
    reasoning_tokens = {"in": 2500, "out": 1500}
    
    for deployment in DEPLOYMENTS:
        print(f"\nCost projections for {deployment}:")
        
        c_struct = calculate_cost(deployment, structuring_tokens["in"], structuring_tokens["out"])
        c_extract = calculate_cost(deployment, extraction_tokens["in"], extraction_tokens["out"])
        c_reason = calculate_cost(deployment, reasoning_tokens["in"], reasoning_tokens["out"])
        
        c_encounter = c_struct + c_extract + c_reason
        c_10k = c_encounter * 10000
        
        results[deployment] = {
            "structuring_per_encounter": c_struct,
            "extraction_per_encounter": c_extract,
            "reasoning_per_encounter": c_reason,
            "total_per_encounter": c_encounter,
            "total_per_10k_encounters": c_10k
        }
        
        print(f"  Structuring: ${c_struct:.5f}")
        print(f"  Extraction:  ${c_extract:.5f}")
        print(f"  Reasoning:   ${c_reason:.5f}")
        print(f"  Total/1:     ${c_encounter:.5f}")
        print(f"  Total/10k:   ${c_10k:.2f}")

    print(f"\nTotal budget spent during these probes: ${budget_manager.current_spend:.4f}")
    
    with open("results/cost.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run()
