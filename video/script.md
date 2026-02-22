# RTL Demo Video Script (3 minutes)

## Shot 1: Problem Statement (0:00–0:25)
**[Screen: Title card — "Radiology Trust Layer"]**

NARRATION:
"Radiology reports are written under time pressure. Language that sounds confident 
can obscure genuine uncertainty — creating risk for patients and liability for clinicians.
Existing AI tools generate new reports. RTL does something different: it *audits* existing ones."

## Shot 2: App Overview (0:25–0:45)
**[Screen: RTL login page, then home dashboard]**

NARRATION:
"RTL is built on MedGemma — Google's open-weight medical AI — hosted as a public 
Gradio app on Hugging Face Spaces. It supports individual audits, batch processing, 
and a full audit trail — without requiring any cloud API."

ACTION:
- Show login → home dashboard with recent audits

## Shot 3: Single Study Audit (0:45–1:45) ← THE CORE DEMO
**[Screen: Single Audit page]**

NARRATION:
"Let's audit a chest X-ray report for a suspected pneumonia case."

ACTION:
1. Upload a chest X-ray image (CXR Example 01)
2. Paste the radiology report text
3. Click "Run Audit"
4. Show progress: "Extracting claims... Analyzing image... Aligning..."

**[Screen: Results — Report Highlights tab]**
NARRATION:
"Each sentence is color-coded: green for supported, yellow for uncertain, red for needs review."

**[Screen: Claim Analysis tab]**
NARRATION:
"MedGemma extracts every factual claim, analyzes the image independently, 
then aligns each claim to visual evidence. Here, four claims are supported — 
but the COPD suggestion is flagged as uncertain."

**[Screen: Suggested Rewrites tab]**
NARRATION:
"RTL suggests specific rewrites to calibrate the language. With one click, 
the radiologist can accept the edit."

## Shot 4: Score and Summary (1:45–2:00)
**[Screen: Score gauge — 80/100, LOW severity]**

NARRATION:
"Every audit produces a 0–100 safety score and severity label — 
giving QA teams an immediate signal without pretending to diagnose."

## Shot 5: Batch Audit (2:00–2:20)
**[Screen: Batch Audit page — upload ZIP]**

NARRATION:
"For hospital QA workflows, RTL supports batch processing: upload a ZIP of cases, 
get a summary dashboard showing average score, severity distribution, 
and which cases need immediate review."

## Shot 6: LoRA Adapter + Evaluation (2:20–2:45)
**[Screen: Evaluation page — Before/After Metrics table]**

NARRATION:
"We also trained an open-weight LoRA adapter on MedGemma, published to Hugging Face.
It improves JSON schema compliance from 72% to 96% and reduces overconfident language 
by 22 percentage points — making the pipeline more reliable for production use."

## Shot 7: Closing (2:45–3:00)
**[Screen: RTL architecture diagram / HF Space URL]**

NARRATION:
"RTL demonstrates that open-weight medical AI can power serious clinical safety tools — 
private, auditable, and deployable anywhere care is delivered.
Try the live demo at [HF Spaces URL]. Full code and model on GitHub and Hugging Face."

---

## Shot List for Recording

| # | Time | Screen | Action |
|---|------|--------|--------|
| 1 | 0:00 | Title card | Fade in |
| 2 | 0:25 | Login page | Type credentials, log in |
| 3 | 0:35 | Home dashboard | Point out tiles and recent audits |
| 4 | 0:45 | Single Audit | Upload image, paste report |
| 5 | 1:00 | Progress updates | Show step-by-step progress |
| 6 | 1:20 | Report Highlights | Click through color-coded claims |
| 7 | 1:30 | Claim Analysis | Show table with labels and evidence |
| 8 | 1:38 | Suggested Rewrites | Click "Accept All" |
| 9 | 1:45 | Score gauge | 80/100 LOW |
| 10 | 2:00 | Batch Audit | Upload ZIP, watch progress |
| 11 | 2:15 | Batch Results | Show summary table |
| 12 | 2:20 | Evaluation | Show Before/After metrics |
| 13 | 2:45 | Closing | URL + GitHub link |
