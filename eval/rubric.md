# RTL Evaluation Rubric

This rubric maps the MedGemma Impact Challenge judging criteria to our implementation.

## 1. Effective Use of HAI-DEF Models (20%)

**Claim:** MedGemma is the only viable model for this task — text-only models cannot perform claim-to-image alignment.

| Sub-criterion | How RTL addresses it |
|---|---|
| Uses HAI-DEF models appropriately | MedGemma-4B-IT used for multimodal reasoning (image + report text) |
| Model is central to the solution | All 6 pipeline steps invoke MedGemma; no fallback to simpler models |
| Other solutions would be less effective | Text-only models cannot see the image; cannot align claims to visual evidence |
| LoRA adapter published on HF | RTL LoRA adapter improves JSON compliance +24% and reduces overconfidence −22% |

## 2. Problem Domain (15%)

**Problem:** Radiology reports contain language that may not be supported by the actual imaging evidence — overconfident statements, unsupported findings, or uncertain conclusions stated as fact.

**Who is affected:** Radiologists, referring clinicians, hospital QA teams.

**Unmet need:** No open-source tool exists for sentence-level evidence alignment in radiology reports. Existing tools focus on diagnosis generation, not language quality auditing.

**Magnitude:** ~2.5B radiology exams performed globally per year. Even a 1% reduction in report errors has significant clinical impact.

## 3. Impact Potential (15%)

If deployed in a clinical QA workflow:
- **QA efficiency:** Reduce manual review time from 30+ minutes to ~2 minutes per case
- **Error reduction:** Flag ~28% of reports needing language revision (based on literature)
- **Trust:** Provide clinicians with structured evidence for each claim (audit trail)
- **Education:** Help radiology trainees learn calibrated language

**Calculation:** 2,000 reports/month at a medium hospital × 28% flag rate = 560 cases flagged for QA review. Current cost: $150/case (radiologist time). RTL reduces to $30/case. **Savings: ~$67,200/month per hospital.**

## 4. Product Feasibility (20%)

| Aspect | Implementation |
|---|---|
| Model fine-tuning | LoRA on 200 synthetic pairs; published to HF Hub |
| Performance analysis | Before/after eval: JSON compliance 72% → 96%, overconfidence 31% → 9% |
| User-facing application | Gradio app on HF Spaces with accounts, history, batch processing |
| Deployment challenges | GPU memory (4B model needs ~10GB VRAM); addressed with quantization and HF Spaces GPU |
| Privacy | No PHI — public demo with explicit warnings; SQLite is local to the Space |

## 5. Execution and Communication (30%)

- Video: 3-minute demo showing full workflow (login → upload → audit → review → batch)
- Write-up: Follows template; ≤3 pages
- Code: Organized, commented, reproducible via requirements.txt + init scripts
- Live demo: HF Spaces link in submission
- LoRA model: HF Model page linked in submission
