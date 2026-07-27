"""Create the five architecture-specific HealthPlus knowledge-base PDFs.

The remaining Doctors, Pricing, and Policies documents are maintained as the
larger Day 3 corpus. All content is fictional and intended for RAG learning.
"""

from __future__ import annotations

from pathlib import Path

import fitz


CORPUS: dict[str, list[tuple[str, str]]] = {
    "SOPs.pdf": [
        (
            "Patient Registration SOP",
            "1. Verify the patient's full name, date of birth, mobile number, and photo ID.\n"
            "2. Create or locate the HealthPlus patient identifier.\n"
            "3. Confirm consent and communication preferences.\n"
            "4. Record allergies only when supplied by the patient or clinician.\n"
            "5. Route clinical questions to qualified medical staff.",
        ),
        (
            "Diagnostic Appointment SOP",
            "Confirm the ordered test, preparation instructions, appointment time, and location.\n"
            "For fasting tests, repeat the approved preparation instructions from the Test Catalog.\n"
            "Escalate order mismatches to the diagnostic desk; reception staff must not alter an order.",
        ),
        (
            "Critical Result Communication SOP",
            "Validated critical results follow the approved escalation matrix. The laboratory contacts "
            "the ordering clinician, records the communication time, and documents read-back confirmation. "
            "AI-generated output must never replace this workflow.",
        ),
    ],
    "Test_Catalog.pdf": [
        (
            "Imaging Tests",
            "MRI Brain — magnetic resonance imaging of the brain; preparation depends on contrast use.\n"
            "CT Brain — computed tomography examination; contrast requirements depend on the order.\n"
            "X-Ray Chest — standard chest radiography.\n"
            "Ultrasound Abdomen — fasting instructions may apply; confirm during scheduling.",
        ),
        (
            "Cardiology Tests",
            "ECG — resting electrical activity recording.\n"
            "2D Echo — ultrasound assessment of cardiac structures.\n"
            "TMT — monitored treadmill assessment; suitability is confirmed by clinical staff.",
        ),
        (
            "Laboratory Tests",
            "CBC — complete blood count.\nLFT — liver function panel.\nKFT — kidney function panel.\n"
            "Thyroid Profile — thyroid-related markers.\nHbA1c — longer-term blood glucose marker.\n"
            "Always follow the preparation instructions attached to the confirmed order.",
        ),
    ],
    "FAQs.pdf": [
        (
            "Appointments and Preparation",
            "Q: How do I book a test?\nA: Use the HealthPlus portal or contact reception.\n\n"
            "Q: Where can I find preparation instructions?\nA: Check the confirmed appointment and Test Catalog.\n\n"
            "Q: Can AI change my preparation instructions?\nA: No. Confirm changes with HealthPlus staff.",
        ),
        (
            "Pricing and Reports",
            "Q: Where can I check test prices?\nA: Refer to the current Pricing document.\n\n"
            "Q: When will my report be ready?\nA: Turnaround guidance is listed in the Reports document.\n\n"
            "Q: Are the demo prices real?\nA: No. This learning corpus uses fictional prices.",
        ),
    ],
    "Health_Packages.pdf": [
        (
            "Master Health Checkup",
            "A fictional learning package containing a general consultation and a defined set of basic "
            "laboratory tests. Package inclusions, exclusions, preparation, and current price must be "
            "confirmed against the booking record and Pricing document.",
        ),
        (
            "Executive Health Checkup",
            "A fictional learning package with an expanded diagnostic set. Eligibility and substitutions "
            "are decided by HealthPlus staff. The AI assistant can explain stored package information but "
            "cannot recommend a package as medical advice.",
        ),
    ],
    "Reports.pdf": [
        (
            "Report Availability",
            "Reports become visible after technical validation and required clinical authorization. "
            "Turnaround times are estimates and may change when recollection, repeat testing, or specialist "
            "review is required. The portal should display the latest status.",
        ),
        (
            "Access and Corrections",
            "Patients access reports through authenticated channels. Identity must be verified before a "
            "report is shared. Suspected demographic or content errors are routed to the reports desk; "
            "the AI assistant must not edit or reinterpret the official report.",
        ),
        (
            "Understanding Results",
            "Reference ranges and flags are contextual and are not a diagnosis. Users should discuss "
            "results with a qualified clinician. The HealthPlus AI demo provides document-grounded "
            "information with citations and is not a medical device.",
        ),
    ],
}


def create_pdf(path: Path, sections: list[tuple[str, str]]) -> None:
    document = fitz.open()
    for heading, body in sections:
        page = document.new_page(width=595, height=842)
        page.insert_text((55, 62), "HealthPlus AI Diagnostic Center", fontsize=16)
        page.insert_text((55, 92), heading, fontsize=14)
        page.insert_textbox(
            fitz.Rect(55, 125, 540, 760),
            body + "\n\nLearning/portfolio content only. Not for clinical use.",
            fontsize=11,
            lineheight=1.35,
        )
    document.set_metadata({"title": path.stem.replace("_", " ")})
    document.save(path)
    document.close()


def main() -> None:
    output_dir = Path("data/knowledge_base/pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, sections in CORPUS.items():
        create_pdf(output_dir / filename, sections)
        print(f"Created {filename} ({len(sections)} pages)")


if __name__ == "__main__":
    main()
