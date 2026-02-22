"""
MedGemma inference client.

Supports three modes (set via MEDGEMMA_INFERENCE_MODE env var):
  - "local"  : Load model via transformers (requires GPU or slow CPU)
  - "api"    : Use HF Inference API (requires HF_TOKEN)
  - "mock"   : Return pre-generated results (no model needed)
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from core import config
from core.util.validation import extract_json_from_text, load_schema, validate

logger = logging.getLogger(__name__)

# ─────────────────────────── Model singleton ──────────────────────────────

_model = None
_processor = None
_lora_loaded: Optional[str] = None

# Mock context — set by audit_pipeline before running, so mock results vary per case
_mock_context: str = ""


def set_mock_context(report_text: str) -> None:
    """Set a hint so mock mode returns case-specific results."""
    global _mock_context
    _mock_context = report_text


def _load_local_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = config.MEDGEMMA_MODEL_ID
    logger.info("Loading MedGemma model: %s", model_id)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if config.HF_TOKEN:
        kwargs["token"] = config.HF_TOKEN

    _processor = AutoProcessor.from_pretrained(model_id, token=config.HF_TOKEN or None)
    _model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if config.RTL_LORA_ID:
        _apply_lora(config.RTL_LORA_ID)

    _model.eval()
    logger.info("Model loaded successfully")
    return _model, _processor


def _apply_lora(lora_repo: str) -> None:
    global _model, _lora_loaded
    if _lora_loaded == lora_repo:
        return
    from peft import PeftModel
    logger.info("Applying LoRA adapter: %s", lora_repo)
    _model = PeftModel.from_pretrained(_model, lora_repo, token=config.HF_TOKEN or None)
    _model = _model.merge_and_unload()
    _lora_loaded = lora_repo


# ─────────────────────────── Inference ────────────────────────────────────

MAX_RETRIES = 3


def _infer_local(prompt: str, image=None) -> str:
    import torch
    model, processor = _load_local_model()

    content = []
    if image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if image is not None:
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MEDGEMMA_MAX_NEW_TOKENS,
            do_sample=False,
        )
    # Decode only the new tokens (skip the prompt)
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True)


def _infer_api(prompt: str, image=None) -> str:
    from huggingface_hub import InferenceClient
    import io

    client = InferenceClient(model=config.MEDGEMMA_MODEL_ID, token=config.HF_TOKEN)

    if image is not None:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        result = client.image_to_text(buf.getvalue())
        # For multimodal chat via API
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.MEDGEMMA_MAX_NEW_TOKENS,
        )
    else:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.MEDGEMMA_MAX_NEW_TOKENS,
        )
    return response.choices[0].message.content


def _raw_infer(prompt: str, image=None) -> str:
    mode = config.MEDGEMMA_INFERENCE_MODE
    if mode == "api":
        return _infer_api(prompt, image)
    else:
        return _infer_local(prompt, image)


# ──────────────────────── Structured inference ────────────────────────────

def infer_structured(
    prompt: str,
    schema_path: Path,
    image=None,
    task_name: str = "task",
) -> tuple[dict, list[str]]:
    """
    Run MedGemma and return (parsed_result, validation_errors).
    Retries up to MAX_RETRIES times with schema reminder.
    """
    if config.MEDGEMMA_MOCK:
        return _mock_result(task_name), []

    schema = load_schema(schema_path)
    last_error: list[str] = []
    raw_text = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                retry_hint = (
                    f"\n\nIMPORTANT: Your previous response was not valid JSON. "
                    f"Errors: {last_error}. "
                    f"You MUST respond with ONLY a valid JSON object."
                )
                prompt_to_use = prompt + retry_hint
            else:
                prompt_to_use = prompt

            raw_text = _raw_infer(prompt_to_use, image)
            parsed = extract_json_from_text(raw_text)

            if parsed is None:
                last_error = [f"No JSON found in response. Got: {raw_text[:200]}"]
                logger.warning("Attempt %d: %s", attempt, last_error[0])
                continue

            errors = validate(parsed, schema)
            if errors:
                last_error = errors
                logger.warning("Attempt %d schema errors: %s", attempt, errors)
                continue

            return parsed, []

        except Exception as e:
            last_error = [str(e)]
            logger.error("Attempt %d exception: %s", attempt, e)
            if attempt < MAX_RETRIES:
                time.sleep(1)

    logger.error("All %d attempts failed for task %s. Last error: %s", MAX_RETRIES, task_name, last_error)
    return _fallback_result(task_name), last_error


# ───────────────────────── Mock / fallback ────────────────────────────────

def _detect_mock_case() -> str:
    """Detect which example case is loaded based on report text keywords."""
    ctx = _mock_context.lower()
    # CHF case: must mention cardiomegaly or heart failure as a positive finding
    if "cardiomegaly" in ctx or "heart failure" in ctx or "venous hypertension" in ctx:
        return "chf"
    # Normal case: lungs clear + normal + no pathology
    elif "pre-operative" in ctx or ("lungs are clear" in ctx and "normal" in ctx):
        return "normal"
    else:
        return "pneumonia"  # default


# ── Case 1: Pneumonia (mixed — 4 supported, 1 uncertain) ─────────────────

_MOCK_PNEUMONIA = {
    "claim_extraction": {
        "claims": [
            {"claim_id": "c1", "text": "There is consolidation in the right lower lobe consistent with pneumonia.", "sentence_span": {"start": 0, "end": 70}, "claim_type": "finding"},
            {"claim_id": "c2", "text": "No pleural effusion is identified.", "sentence_span": {"start": 71, "end": 105}, "claim_type": "absence"},
            {"claim_id": "c3", "text": "The cardiomediastinal silhouette is within normal limits.", "sentence_span": {"start": 106, "end": 163}, "claim_type": "finding"},
            {"claim_id": "c4", "text": "No pneumothorax is seen.", "sentence_span": {"start": 164, "end": 187}, "claim_type": "absence"},
            {"claim_id": "c5", "text": "Mild hyperinflation is noted, possibly consistent with early COPD.", "sentence_span": {"start": 188, "end": 253}, "claim_type": "impression"},
        ]
    },
    "image_findings": {
        "findings": [
            {"finding_id": "f1", "description": "Increased opacity in the right lower lobe", "location": "right lower lobe", "confidence": 0.82, "visual_cue": "Dense white area replacing normal lung markings"},
            {"finding_id": "f2", "description": "Clear costophrenic angles bilaterally", "location": "bilateral costophrenic angles", "confidence": 0.91, "visual_cue": "Sharp, well-defined angles without blunting"},
            {"finding_id": "f3", "description": "Normal cardiac borders and mediastinal width", "location": "mediastinum", "confidence": 0.88, "visual_cue": "Cardiac silhouette within normal proportions"},
            {"finding_id": "f4", "description": "No visible pneumothorax", "location": "bilateral lung apices", "confidence": 0.85, "visual_cue": "Lung markings visible to periphery bilaterally"},
            {"finding_id": "f5", "description": "Slightly increased AP diameter and flattened diaphragms", "location": "bilateral diaphragms", "confidence": 0.61, "visual_cue": "Diaphragms appear somewhat flattened"},
        ],
        "image_quality": "adequate",
        "overall_impression": "PA chest radiograph with right lower lobe opacity and possible hyperinflation."
    },
    "alignment": {
        "alignments": [
            {"claim_id": "c1", "label": "supported", "evidence": "Right lower lobe opacity (f1) is clearly visible, consistent with consolidation.", "confidence": 0.82, "related_finding_ids": ["f1"]},
            {"claim_id": "c2", "label": "supported", "evidence": "Bilateral costophrenic angles are clear (f2), no blunting suggesting effusion.", "confidence": 0.91, "related_finding_ids": ["f2"]},
            {"claim_id": "c3", "label": "supported", "evidence": "Cardiac borders and mediastinal contour appear normal (f3).", "confidence": 0.88, "related_finding_ids": ["f3"]},
            {"claim_id": "c4", "label": "supported", "evidence": "Lung markings extend to periphery (f4), no visible pleural line.", "confidence": 0.85, "related_finding_ids": ["f4"]},
            {"claim_id": "c5", "label": "uncertain", "evidence": "Diaphragms slightly flattened (f5), but COPD diagnosis requires clinical correlation.", "confidence": 0.61, "related_finding_ids": ["f5"]},
        ]
    },
    "rewrite": {
        "rewrites": [
            {"claim_id": "c5", "original": "Mild hyperinflation is noted, possibly consistent with early COPD.", "suggested": "There may be mild hyperinflation; clinical correlation is recommended to evaluate for COPD.", "reason": "COPD is a clinical diagnosis; radiographic suggestion should be hedged."}
        ],
        "edited_report": "There is consolidation in the right lower lobe consistent with pneumonia. No pleural effusion is identified. The cardiomediastinal silhouette is within normal limits. No pneumothorax is seen. There may be mild hyperinflation; clinical correlation is recommended to evaluate for COPD."
    },
    "clinician_summary": {
        "summary": "Report is largely well-supported by imaging. Four of five claims are clearly supported. One claim regarding possible COPD has been flagged for hedging.",
        "key_concerns": ["Claim c5: COPD suggestion requires clinical correlation and was rewritten for calibration."],
        "recommendation": "review_recommended",
        "confidence_note": "Audit performed using MedGemma-4B-IT. Image quality was adequate."
    },
    "patient_explain": {
        "plain_language_summary": "Your chest X-ray shows an area of cloudiness in the lower right part of your lung, which may indicate a lung infection (pneumonia). No fluid around the lungs, no collapsed lung, and no heart enlargement were found.",
        "what_was_found": "An area of cloudiness (consolidation) in the lower right part of your lung.",
        "what_it_means": "This often means there is a lung infection. Your doctor will explain what this means for your treatment.",
        "next_steps": "Please follow up with your doctor to discuss treatment and whether additional tests are needed."
    },
}


# ── Case 2: CHF (problematic — 2 needs_review, 1 uncertain) ──────────────

_MOCK_CHF = {
    "claim_extraction": {
        "claims": [
            {"claim_id": "c1", "text": "Bilateral interstitial opacities are present, greater on the right.", "sentence_span": {"start": 0, "end": 66}, "claim_type": "finding"},
            {"claim_id": "c2", "text": "There is mild cardiomegaly.", "sentence_span": {"start": 67, "end": 94}, "claim_type": "finding"},
            {"claim_id": "c3", "text": "Bilateral pleural effusions are suspected, right greater than left.", "sentence_span": {"start": 95, "end": 162}, "claim_type": "finding"},
            {"claim_id": "c4", "text": "No pneumothorax is identified.", "sentence_span": {"start": 163, "end": 193}, "claim_type": "absence"},
            {"claim_id": "c5", "text": "The pulmonary vasculature appears engorged, consistent with pulmonary venous hypertension.", "sentence_span": {"start": 194, "end": 285}, "claim_type": "impression"},
        ]
    },
    "image_findings": {
        "findings": [
            {"finding_id": "f1", "description": "Bilateral hazy opacities, worse on right", "location": "bilateral lung fields", "confidence": 0.76, "visual_cue": "Diffuse increased density in both lung fields"},
            {"finding_id": "f2", "description": "Cardiac silhouette mildly enlarged", "location": "mediastinum", "confidence": 0.72, "visual_cue": "Cardiothoracic ratio approximately 0.55"},
            {"finding_id": "f3", "description": "Blunting of right costophrenic angle", "location": "right costophrenic angle", "confidence": 0.68, "visual_cue": "Meniscus sign at right base"},
            {"finding_id": "f4", "description": "No pneumothorax identified", "location": "bilateral apices", "confidence": 0.89, "visual_cue": "Lung markings visible to periphery"},
            {"finding_id": "f5", "description": "Upper lobe pulmonary venous distension", "location": "bilateral upper lobes", "confidence": 0.58, "visual_cue": "Vessels in upper lobes appear prominent"},
        ],
        "image_quality": "adequate",
        "overall_impression": "PA chest radiograph showing bilateral opacities, possible cardiomegaly, and right-sided pleural effusion."
    },
    "alignment": {
        "alignments": [
            {"claim_id": "c1", "label": "supported", "evidence": "Bilateral hazy opacities are visible (f1), right worse than left.", "confidence": 0.76, "related_finding_ids": ["f1"]},
            {"claim_id": "c2", "label": "uncertain", "evidence": "Cardiac silhouette may be mildly enlarged (f2), but CTR is borderline at ~0.55. Definitive cardiomegaly requires CTR >0.5 on a PA film with good inspiration.", "confidence": 0.72, "related_finding_ids": ["f2"]},
            {"claim_id": "c3", "label": "needs_review", "evidence": "Right costophrenic angle blunting (f3) could represent effusion, but only the right side shows clear evidence. Left effusion is not convincingly demonstrated.", "confidence": 0.68, "related_finding_ids": ["f3"]},
            {"claim_id": "c4", "label": "supported", "evidence": "No evidence of pneumothorax (f4). Lung markings visible bilaterally.", "confidence": 0.89, "related_finding_ids": ["f4"]},
            {"claim_id": "c5", "label": "needs_review", "evidence": "Upper lobe venous distension is subtle (f5). Pulmonary venous hypertension is a strong clinical claim that requires more definitive imaging evidence.", "confidence": 0.58, "related_finding_ids": ["f5"]},
        ]
    },
    "rewrite": {
        "rewrites": [
            {"claim_id": "c2", "original": "There is mild cardiomegaly.", "suggested": "The cardiac silhouette appears borderline enlarged; correlation with prior imaging is recommended.", "reason": "CTR is borderline; definitive cardiomegaly statement is overly confident."},
            {"claim_id": "c3", "original": "Bilateral pleural effusions are suspected, right greater than left.", "suggested": "A right-sided pleural effusion is suspected based on costophrenic angle blunting. Left-sided effusion is not clearly demonstrated.", "reason": "Only right costophrenic angle shows clear blunting; bilateral claim is not fully supported."},
            {"claim_id": "c5", "original": "The pulmonary vasculature appears engorged, consistent with pulmonary venous hypertension.", "suggested": "There may be upper lobe pulmonary venous prominence; clinical correlation for pulmonary venous hypertension is suggested.", "reason": "Pulmonary venous hypertension is a strong claim requiring more definitive evidence."},
        ],
        "edited_report": "Bilateral interstitial opacities are present, greater on the right. The cardiac silhouette appears borderline enlarged; correlation with prior imaging is recommended. A right-sided pleural effusion is suspected based on costophrenic angle blunting. Left-sided effusion is not clearly demonstrated. No pneumothorax is identified. There may be upper lobe pulmonary venous prominence; clinical correlation for pulmonary venous hypertension is suggested."
    },
    "clinician_summary": {
        "summary": "This report contains multiple claims that exceed the imaging evidence. Two claims were flagged as needing review and one as uncertain. The bilateral effusion and pulmonary hypertension claims are not adequately supported.",
        "key_concerns": [
            "Claim c3: Bilateral effusion stated but only right side shows clear evidence.",
            "Claim c5: Pulmonary venous hypertension claim exceeds subtle imaging findings.",
            "Claim c2: Cardiomegaly borderline — statement could be softened.",
        ],
        "recommendation": "review_recommended",
        "confidence_note": "Audit performed using MedGemma-4B-IT. Multiple claims need radiologist verification."
    },
    "patient_explain": {
        "plain_language_summary": "Your chest X-ray shows some haziness in both lungs and your heart may be slightly larger than normal. There may be fluid on one side. Your doctor will review these findings to determine the best next steps.",
        "what_was_found": "Haziness in both lungs, a borderline enlarged heart, and possible fluid on the right side of the chest.",
        "what_it_means": "These findings can be associated with heart-related conditions. Your doctor will explain what they mean for your specific situation.",
        "next_steps": "Follow up with your doctor. Additional testing such as an echocardiogram may be recommended."
    },
}


# ── Case 3: Normal study (all supported, high score) ─────────────────────

_MOCK_NORMAL = {
    "claim_extraction": {
        "claims": [
            {"claim_id": "c1", "text": "The lungs are clear.", "sentence_span": {"start": 0, "end": 20}, "claim_type": "finding"},
            {"claim_id": "c2", "text": "No focal consolidation, pleural effusion, or pneumothorax.", "sentence_span": {"start": 21, "end": 79}, "claim_type": "absence"},
            {"claim_id": "c3", "text": "The cardiomediastinal silhouette is normal.", "sentence_span": {"start": 80, "end": 122}, "claim_type": "finding"},
            {"claim_id": "c4", "text": "Bony structures are intact.", "sentence_span": {"start": 123, "end": 150}, "claim_type": "finding"},
        ]
    },
    "image_findings": {
        "findings": [
            {"finding_id": "f1", "description": "Clear bilateral lung fields without opacities", "location": "bilateral lungs", "confidence": 0.95, "visual_cue": "Both lung fields appear uniformly lucent with normal vascular markings"},
            {"finding_id": "f2", "description": "Sharp costophrenic angles bilaterally", "location": "bilateral costophrenic angles", "confidence": 0.94, "visual_cue": "No blunting or meniscus sign"},
            {"finding_id": "f3", "description": "Normal cardiac silhouette", "location": "mediastinum", "confidence": 0.93, "visual_cue": "CTR within normal limits, normal mediastinal contour"},
            {"finding_id": "f4", "description": "Intact bony thorax", "location": "ribs and spine", "confidence": 0.90, "visual_cue": "No fractures or lytic lesions identified"},
        ],
        "image_quality": "adequate",
        "overall_impression": "PA chest radiograph with no acute findings. Normal study."
    },
    "alignment": {
        "alignments": [
            {"claim_id": "c1", "label": "supported", "evidence": "Lung fields are clear and lucent (f1) with normal vascular markings.", "confidence": 0.95, "related_finding_ids": ["f1"]},
            {"claim_id": "c2", "label": "supported", "evidence": "No consolidation visible, costophrenic angles sharp (f2), no pleural line to suggest pneumothorax.", "confidence": 0.94, "related_finding_ids": ["f1", "f2"]},
            {"claim_id": "c3", "label": "supported", "evidence": "Cardiac silhouette and mediastinal contour are normal (f3).", "confidence": 0.93, "related_finding_ids": ["f3"]},
            {"claim_id": "c4", "label": "supported", "evidence": "No fractures or lytic lesions in visible bony structures (f4).", "confidence": 0.90, "related_finding_ids": ["f4"]},
        ]
    },
    "rewrite": {
        "rewrites": [],
        "edited_report": "The lungs are clear. No focal consolidation, pleural effusion, or pneumothorax. The cardiomediastinal silhouette is normal. Bony structures are intact."
    },
    "clinician_summary": {
        "summary": "All claims in the report are well-supported by imaging evidence. This is a normal chest radiograph with no concerning findings. No rewrites needed.",
        "key_concerns": [],
        "recommendation": "no_action_needed",
        "confidence_note": "Audit performed using MedGemma-4B-IT. High confidence across all claims."
    },
    "patient_explain": {
        "plain_language_summary": "Your chest X-ray looks normal. Your lungs are clear, your heart is a normal size, and your bones look healthy. No problems were found.",
        "what_was_found": "Nothing abnormal. Your lungs, heart, and bones all appear normal.",
        "what_it_means": "This is good news — your chest X-ray does not show any signs of disease or injury.",
        "next_steps": "No additional imaging is needed based on these results. Continue with your doctor's recommendations."
    },
}


def _mock_result(task_name: str) -> dict:
    case = _detect_mock_case()
    mock_sets = {
        "pneumonia": _MOCK_PNEUMONIA,
        "chf": _MOCK_CHF,
        "normal": _MOCK_NORMAL,
    }
    case_data = mock_sets.get(case, _MOCK_PNEUMONIA)
    return case_data.get(task_name, {"_mock": True, "task": task_name})


def _fallback_result(task_name: str) -> dict:
    """Minimal valid fallback when all inference attempts fail."""
    fallbacks = {
        "claim_extraction": {"claims": []},
        "image_findings": {"findings": [], "image_quality": "poor", "overall_impression": "Analysis failed."},
        "alignment": {"alignments": []},
        "rewrite": {"rewrites": [], "edited_report": ""},
        "clinician_summary": {"summary": "Audit could not be completed.", "key_concerns": [], "recommendation": "review_recommended", "confidence_note": "Inference failed."},
        "patient_explain": {"plain_language_summary": "Unable to generate explanation."},
    }
    return fallbacks.get(task_name, {})
