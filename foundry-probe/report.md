# Azure AI Foundry Evaluation Report

## (a) Executive Summary
This report summarizes the evaluation of Azure AI Foundry endpoints for a medical voice-dictation pipeline.
We tested `gpt-5.6-luna,gpt-5.6-terra,gpt-5.6-sol,gpt-4o-mini` for latency, capability, fidelity, and rate limits.
*(Check the sections below for detailed insights on what worked and what failed).*

## (b) Capability Matrix
| Deployment | Chat | Stream | JSON Obj | Strict JSON | System | Developer | Temp/Top P | Max Tokens | Max Comp Tokens | Reasoning | Long Context | Arabic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## (c) Processing Region Findings
| Deployment | Region | Request ID | Rate Limit Remaining |
|---|---|---|---|
| gpt-5.6-luna | Error | - | - |
| gpt-5.6-terra | Error | - | - |
| gpt-5.6-sol | Error | - | - |
| gpt-4o-mini | Error | - | - |

## (d) Quality Scores (Structuring & Extraction)
### Structuring Fidelity

### Strict JSON Extraction
**gpt-5.6-luna**:
- Schema `ER`: Valid=❌, Hallucinations=[]
- Schema `OPD`: Valid=❌, Hallucinations=[]
- Schema `Dental`: Valid=❌, Hallucinations=[]
**gpt-5.6-terra**:
- Schema `ER`: Valid=❌, Hallucinations=[]
- Schema `OPD`: Valid=❌, Hallucinations=[]
- Schema `Dental`: Valid=❌, Hallucinations=[]
**gpt-5.6-sol**:
- Schema `ER`: Valid=❌, Hallucinations=[]
- Schema `OPD`: Valid=❌, Hallucinations=[]
- Schema `Dental`: Valid=❌, Hallucinations=[]
**gpt-4o-mini**:
- Schema `ER`: Valid=❌, Hallucinations=[]
- Schema `OPD`: Valid=❌, Hallucinations=[]
- Schema `Dental`: Valid=❌, Hallucinations=[]

## (e) Latency and Throughput (Pipeline & Reasoning)
### Pipeline Simulation (Structuring -> Extraction/Reasoning)
| Deployment | TTFT p50 | Total p50 | Total p95 |
|---|---|---|---|
| gpt-5.6-luna | N/A | N/A | N/A |
| gpt-5.6-terra | N/A | N/A | N/A |
| gpt-5.6-sol | N/A | N/A | N/A |
| gpt-4o-mini | N/A | N/A | N/A |

## (f) Rate-Limit Findings (Concurrency)
**gpt-5.6-luna**:
| Concurrency | Backoff | Success | 429s | Other Errs | p50 (s) | p95 (s) | Throughput (req/s) | Retry-Afters |
|---|---|---|---|---|---|---|---|---|
| 1 | False | 0 | 0 | 1 | 0.00 | 0.00 | 0.00 | [] |
| 5 | False | 0 | 0 | 5 | 0.00 | 0.00 | 0.00 | [] |
| 10 | False | 0 | 0 | 10 | 0.00 | 0.00 | 0.00 | [] |
| 20 | False | 0 | 0 | 20 | 0.00 | 0.00 | 0.00 | [] |
| 30 | False | 0 | 0 | 30 | 0.00 | 0.00 | 0.00 | [] |
| 50 | False | 0 | 0 | 50 | 0.00 | 0.00 | 0.00 | [] |
| 30 | True | 0 | 0 | 30 | 0.00 | 0.00 | 0.00 | [] |
| 50 | True | 0 | 0 | 50 | 0.00 | 0.00 | 0.00 | [] |
**gpt-5.6-terra**:
| Concurrency | Backoff | Success | 429s | Other Errs | p50 (s) | p95 (s) | Throughput (req/s) | Retry-Afters |
|---|---|---|---|---|---|---|---|---|
| 1 | False | 0 | 0 | 1 | 0.00 | 0.00 | 0.00 | [] |
| 5 | False | 0 | 0 | 5 | 0.00 | 0.00 | 0.00 | [] |
| 10 | False | 0 | 0 | 10 | 0.00 | 0.00 | 0.00 | [] |
| 20 | False | 0 | 0 | 20 | 0.00 | 0.00 | 0.00 | [] |
| 30 | False | 0 | 0 | 30 | 0.00 | 0.00 | 0.00 | [] |
| 50 | False | 0 | 0 | 50 | 0.00 | 0.00 | 0.00 | [] |
| 30 | True | 0 | 0 | 30 | 0.00 | 0.00 | 0.00 | [] |
| 50 | True | 0 | 0 | 50 | 0.00 | 0.00 | 0.00 | [] |
**gpt-5.6-sol**:
| Concurrency | Backoff | Success | 429s | Other Errs | p50 (s) | p95 (s) | Throughput (req/s) | Retry-Afters |
|---|---|---|---|---|---|---|---|---|
| 1 | False | 0 | 0 | 1 | 0.00 | 0.00 | 0.00 | [] |
| 5 | False | 0 | 0 | 5 | 0.00 | 0.00 | 0.00 | [] |
| 10 | False | 0 | 0 | 10 | 0.00 | 0.00 | 0.00 | [] |
**gpt-4o-mini**:
| Concurrency | Backoff | Success | 429s | Other Errs | p50 (s) | p95 (s) | Throughput (req/s) | Retry-Afters |
|---|---|---|---|---|---|---|---|---|
| 1 | False | 0 | 0 | 1 | 0.00 | 0.00 | 0.00 | [] |
| 5 | False | 0 | 0 | 5 | 0.00 | 0.00 | 0.00 | [] |
| 10 | False | 0 | 0 | 10 | 0.00 | 0.00 | 0.00 | [] |
| 20 | False | 0 | 0 | 20 | 0.00 | 0.00 | 0.00 | [] |
| 30 | False | 0 | 0 | 30 | 0.00 | 0.00 | 0.00 | [] |
| 50 | False | 0 | 0 | 50 | 0.00 | 0.00 | 0.00 | [] |
| 30 | True | 0 | 0 | 30 | 0.00 | 0.00 | 0.00 | [] |
| 50 | True | 0 | 0 | 50 | 0.00 | 0.00 | 0.00 | [] |

## (g) Content Filter Findings
**gpt-5.6-luna**:
- `suicidal_ideation`: ✅ Passed
- `child_abuse`: ✅ Passed
- `drug_overdose`: ✅ Passed
- `sexual_history`: ✅ Passed
- `violent_injury`: ✅ Passed
**gpt-5.6-terra**:
- `suicidal_ideation`: ✅ Passed
- `child_abuse`: ✅ Passed
- `drug_overdose`: ✅ Passed
- `sexual_history`: ✅ Passed
- `violent_injury`: ✅ Passed
**gpt-5.6-sol**:
- `suicidal_ideation`: ✅ Passed
- `child_abuse`: ✅ Passed
- `drug_overdose`: ✅ Passed
- `sexual_history`: ✅ Passed
- `violent_injury`: ✅ Passed
**gpt-4o-mini**:
- `suicidal_ideation`: ✅ Passed
- `child_abuse`: ✅ Passed
- `drug_overdose`: ✅ Passed
- `sexual_history`: ✅ Passed
- `violent_injury`: ✅ Passed

## (h) Cost Projections
| Deployment | Cost / Encounter | Cost / 10k Encounters |
|---|---|---|
| gpt-5.6-luna | $0.01280 | $128.00 |
| gpt-5.6-terra | $0.07575 | $757.50 |
| gpt-5.6-sol | $0.45450 | $4545.00 |
| gpt-4o-mini | $0.00455 | $45.45 |

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
