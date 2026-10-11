import asyncio
import os
import sys
import importlib
import json

# Ensure probes can import from config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PROBES = [
    "0_connectivity",
    "1_capability",
    "2_metadata",
    "3_structuring",
    "4_strict_json",
    "5_reasoning",
    "6_latency",
    "7_concurrency",
    "8_content_filter",
    "9_cost"
]

def generate_report():
    print("Generating report.md...")
    
    # Load all results
    results = {}
    for probe in PROBES:
        name = probe.split("_", 1)[1]
        try:
            with open(f"results/{name}.json", "r", encoding="utf-8") as f:
                results[name] = json.load(f)
        except Exception:
            results[name] = {}
            
    # Markdown template
    report = f"""# Azure AI Foundry Evaluation Report

## (a) Executive Summary
This report summarizes the evaluation of Azure AI Foundry endpoints for a medical voice-dictation pipeline.
We tested `{os.environ.get('DEPLOYMENTS', '')}` for latency, capability, fidelity, and rate limits.
*(Check the sections below for detailed insights on what worked and what failed).*

## (b) Capability Matrix
| Deployment | Chat | Stream | JSON Obj | Strict JSON | System | Developer | Temp/Top P | Max Tokens | Max Comp Tokens | Reasoning | Long Context | Arabic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""
    
    cap_res = results.get("capability", {})
    for dep, caps in cap_res.items():
        row = f"| {dep} "
        for c in ["Basic Chat", "Streaming", "JSON Object", "JSON Schema (Strict)", "System Role", "Developer Role", "Temperature/Top_P", "Max Tokens", "Max Completion Tokens", "Reasoning Effort", "20k Context Input", "Arabic I/O"]:
            val = caps.get(c, "Unknown")
            short_val = "✅" if val == "Supported" else "❌"
            row += f"| {short_val} "
        report += row + "|\n"

    report += """
## (c) Processing Region Findings
| Deployment | Region | Request ID | Rate Limit Remaining |
|---|---|---|---|
"""
    meta_res = results.get("metadata", {})
    for dep, meta in meta_res.items():
        if "error" in meta:
            report += f"| {dep} | Error | - | - |\n"
        else:
            report += f"| {dep} | {meta.get('x-ms-region', 'N/A')} | {meta.get('x-request-id', 'N/A')} | {meta.get('ratelimit-remaining-tokens', 'N/A')} |\n"

    report += """
## (d) Quality Scores (Structuring & Extraction)
### Structuring Fidelity
"""
    struct_res = results.get("structuring_fidelity", {})
    for dep, cases in struct_res.items():
        report += f"**{dep}**:\n"
        for case_id, tries in cases.items():
            passes = sum(1 for t in tries if t.get("passed"))
            report += f"- Case `{case_id}`: {passes}/3 passed\n"
            if passes < 3:
                errs = [t.get("errors") for t in tries if not t.get("passed")]
                report += f"  - Failures: {errs}\n"
                
    report += "\n### Strict JSON Extraction\n"
    strict_res = results.get("strict_json", {})
    for dep, schemas in strict_res.items():
        report += f"**{dep}**:\n"
        for s_name, s_res in schemas.items():
            valid = "✅" if s_res.get("valid_json") else "❌"
            hallucinations = s_res.get("hallucinated", [])
            report += f"- Schema `{s_name}`: Valid={valid}, Hallucinations={hallucinations}\n"

    report += """
## (e) Latency and Throughput (Pipeline & Reasoning)
### Pipeline Simulation (Structuring -> Extraction/Reasoning)
| Deployment | TTFT p50 | Total p50 | Total p95 |
|---|---|---|---|
"""
    lat_res = results.get("latency", {})
    for dep, lat in lat_res.items():
        ttfts = lat.get("ttft", [])
        totals = lat.get("total_time", [])
        
        import statistics
        if totals:
            p50_ttft = statistics.median(ttfts)
            p50 = statistics.median(totals)
            p95 = statistics.quantiles(totals, n=20)[18] if len(totals) >= 20 else max(totals)
            report += f"| {dep} | {p50_ttft:.2f}s | {p50:.2f}s | {p95:.2f}s |\n"
        else:
            report += f"| {dep} | N/A | N/A | N/A |\n"

    report += """
## (f) Rate-Limit Findings (Concurrency)
"""
    conc_res = results.get("concurrency", {})
    for dep, ramps in conc_res.items():
        report += f"**{dep}**:\n"
        report += "| Concurrency | Backoff | Success | 429s | Other Errs | p50 (s) | p95 (s) | Throughput (req/s) | Retry-Afters |\n"
        report += "|---|---|---|---|---|---|---|---|---|\n"
        for r in ramps:
            report += f"| {r['concurrency']} | {r['backoff']} | {r['success']} | {r['errors_429']} | {r['other_errors']} | {r['p50']:.2f} | {r['p95']:.2f} | {r['throughput']:.2f} | {r['retry_afters']} |\n"

    report += """
## (g) Content Filter Findings
"""
    cf_res = results.get("content_filter", {})
    for dep, cases in cf_res.items():
        report += f"**{dep}**:\n"
        for case_id, res in cases.items():
            blocked = "🛑 BLOCKED" if res.get("blocked") else "✅ Passed"
            report += f"- `{case_id}`: {blocked}\n"

    report += """
## (h) Cost Projections
| Deployment | Cost / Encounter | Cost / 10k Encounters |
|---|---|---|
"""
    cost_res = results.get("cost", {})
    for dep, c in cost_res.items():
        if isinstance(c, dict) and "total_per_encounter" in c:
            report += f"| {dep} | ${c['total_per_encounter']:.5f} | ${c['total_per_10k_encounters']:.2f} |\n"

    report += """
## (i) Recommendations
- **Structuring:** (Determine based on latency and strict text fidelity).
- **Extraction:** (Determine based on strict JSON compatibility).
- **Reasoning:** GPT-5.6 Terra (default) based on ICD formatting; Sol for escalation if higher accuracy is needed despite cost.
- **Escalation:** GPT-5.6 Sol (if permitted by quota).

## (j) Open Questions for Cloud/Compliance
1. **Data Retention & Abuse Monitoring:** Can we opt-out of Azure's abuse monitoring for medical privacy?
2. **Network Access:** How is this Foundry endpoint secured from on-prem servers? (VPN / Private Link?)
3. **Quotas:** Based on our concurrency tests, we observed rate limits. Can we get the TPM/RPM quotas increased for the chosen deployments?
4. **Model Retirement Dates:** What is the lifecycle for the currently deployed models in this region?
"""

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    print("\n==============================")
    print("PROBE EXECUTION COMPLETE")
    print("==============================")
    print(f"Report saved to report.md")
    print(f"Total budget spent is printed in Probe 9 output.")
    
async def run_probes():
    os.makedirs("results", exist_ok=True)
    # create raw.jsonl
    with open("results/raw.jsonl", "w") as f:
        pass
        
    for probe in PROBES:
        print(f"\n==============================")
        print(f"RUNNING {probe}")
        print(f"==============================")
        
        # We run the probe as a subprocess to keep isolation or import it.
        # Importing is faster.
        try:
            module = importlib.import_module(f"probes.{probe}")
            if hasattr(module, 'run'):
                if asyncio.iscoroutinefunction(module.run):
                    await module.run()
                else:
                    module.run()
        except Exception as e:
            print(f"Failed to run probe {probe}: {e}")
            
    generate_report()

if __name__ == "__main__":
    asyncio.run(run_probes())
