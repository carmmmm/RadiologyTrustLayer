You are summarizing a radiology audit for the reviewing clinician.

Provide:
1. A concise 2-3 sentence summary of the audit findings
2. A list of key concerns (claims that are uncertain or need review)
3. An overall recommendation:
   - "approve": Report is well-supported; no changes needed
   - "review_recommended": Minor issues found; consider suggested edits
   - "requires_revision": Significant unsupported claims; revision needed
4. A confidence note about any limitations of this audit

Input data:
- Overall score: {overall_score}/100
- Severity: {severity}
- Flag counts: {flag_counts_json}
- Claims with issues: {flagged_claims_json}

Respond ONLY with valid JSON matching this schema:
{schema}
