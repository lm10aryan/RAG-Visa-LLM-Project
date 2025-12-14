from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DATA_PATH = ROOT / "data/evaluation/test_questions.json"


FACT_QUESTIONS = [
    (
        "What is the H1B cap?",
        "65,000 visas for the regular cap plus 20,000 for the advanced degree exemption each fiscal year.",
        ["65,000", "20,000"],
        ["65,000", "20,000"],
    ),
    (
        "How much is the H1B filing fee?",
        "The base filing fee for Form I-129 H-1B petitions is $460.",
        ["460"],
        ["460"],
    ),
    (
        "What is the ACWIA fee?",
        "The ACWIA fee is $750 for employers with 25 or fewer workers and $1,500 for employers with 26 or more.",
        ["750", "1,500"],
        ["750", "1,500"],
    ),
    (
        "What is the premium processing fee?",
        "Premium processing currently costs $2,805 via Form I-907 for H-1B petitions.",
        ["2,805"],
        ["2,805"],
    ),
    (
        "When does the H1B cap season start?",
        "USCIS typically opens H-1B registration at the start of March for the upcoming fiscal year.",
        ["march"],
        ["march"],
    ),
    (
        "How long is H1B valid initially?",
        "Initial H-1B approval can be granted for up to three years.",
        ["three"],
        ["three"],
    ),
    (
        "What is the advanced degree cap?",
        "There are 20,000 additional H-1B visas reserved for beneficiaries with U.S. advanced degrees.",
        ["20,000"],
        ["20,000"],
    ),
    (
        "When can H1B start work?",
        "H-1B employment tied to the cap generally cannot begin before October 1 of the fiscal year.",
        ["october 1"],
        ["october"],
    ),
    (
        "What is the fraud prevention fee?",
        "Most initial H-1B petitions require a $500 Fraud Prevention and Detection fee.",
        ["500"],
        ["500"],
    ),
    (
        "How many H1Bs are issued annually?",
        "Congress authorizes 65,000 regular H-1B visas plus 20,000 for advanced degree holders each year.",
        ["65,000", "20,000"],
        ["65,000"],
    ),
]

REQUIREMENT_QUESTIONS = [
    (
        "What education is required for H1B?",
        "H-1B roles require a minimum of a bachelor's degree or its equivalent in the specialty occupation.",
        ["bachelor", "degree"],
    ),
    (
        "Can I self-petition for H1B?",
        "No, an employer must file the H-1B petition on behalf of the worker; self-petitioning is not allowed.",
        ["employer", "file"],
    ),
    (
        "Does H1B allow dual intent?",
        "Yes, H-1B is a dual-intent classification allowing an intent to pursue permanent residence.",
        ["dual", "intent"],
    ),
    (
        "Who must file the H1B petition?",
        "The U.S. employer files Form I-129 with a certified Labor Condition Application before hiring the worker.",
        ["employer", "i-129"],
    ),
    (
        "Is work experience equivalent to degree?",
        "In limited cases, three years of specialized training or experience may substitute for one year of education toward the bachelor's requirement.",
        ["experience", "education"],
    ),
    (
        "Can H1B change employers?",
        "Yes, an H-1B worker can change employers if the new employer files a petition and portability rules are met.",
        ["change", "petition"],
    ),
    (
        "What is a specialty occupation?",
        "It is a position that requires theoretical and practical application of highly specialized knowledge and at least a bachelor's degree in the field.",
        ["specialized", "bachelor"],
    ),
    (
        "Do I need labor certification for H1B?",
        "H-1B petitions require a certified Labor Condition Application from the Department of Labor, not a PERM labor certification.",
        ["labor condition application"],
    ),
    (
        "Can H1B be extended beyond 6 years?",
        "Extensions beyond six years may be possible with approved I-140 petitions or AC21 provisions.",
        ["six", "ac21"],
    ),
    (
        "Is there a minimum salary for H1B?",
        "Employers must pay at least the higher of the prevailing wage or the actual wage paid to similar workers.",
        ["prevailing", "wage"],
    ),
]

COMPLEX_QUESTIONS = [
    (
        "I'm on F1 OPT, can I apply for H1B?",
        "Yes, OPT students may apply and can benefit from cap-gap provisions that bridge status until October 1.",
        ["yes", "cap-gap", "october"],
    ),
    (
        "If H1B not selected in lottery, can I stay on F1?",
        "If not selected, you remain in F-1 status only until OPT or grace periods expire; cap-gap requires selection.",
        ["not selected", "status"],
    ),
    (
        "Can I work while H1B petition is pending?",
        "You may continue working for the same employer if you maintain valid work authorization such as OPT or existing H-1B status.",
        ["continue", "authorization"],
    ),
    (
        "If I change employers, do I need new H1B?",
        "Yes, each new employer must file an H-1B petition, though portability allows work upon USCIS receipt.",
        ["new employer", "portability"],
    ),
    (
        "Can H1B lead to green card?",
        "Yes, H-1B holders can pursue permanent residence through employment-based petitions while maintaining H-1B status.",
        ["green card", "employment-based"],
    ),
    (
        "What happens if H1B is denied after F1 expires?",
        "If H-1B is denied after OPT or F-1 status ends, you generally must depart the U.S. unless another status applies.",
        ["depart", "status"],
    ),
    (
        "Can I travel while H1B is pending?",
        "Travel is risky; departing the U.S. during change-of-status processing can abandon that request unless you have another visa for reentry.",
        ["travel", "risky"],
    ),
    (
        "Does cap-exempt H1B count toward cap later?",
        "Time spent in cap-exempt H-1B status does not count toward the cap, but moving to a cap-subject employer requires lottery selection.",
        ["cap-exempt", "cap-subject"],
    ),
    (
        "Can I have multiple H1B petitions?",
        "Multiple bona fide employers may file separate H-1B petitions for the same worker; duplicate petitions by the same employer are not allowed.",
        ["multiple", "employers"],
    ),
    (
        "If laid off on H1B, how long can I stay?",
        "You generally receive a 60-day grace period (or until I-94 expiry) to find new employment, change status, or depart.",
        ["60-day", "grace"],
    ),
]


def build_ground_truth(answer: str, key_terms: List[str], must_include: List[str], tier: int) -> Dict:
    return {
        "answer": answer,
        "key_facts": key_terms,
        "must_include": must_include,
        "must_not_include": [],
        "acceptable_variations": [],
        "tier": tier,
        "source_urls": ["https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations"],
    }


def generate_test_dataset() -> List[Dict]:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    questions: List[Dict] = []

    for idx, (question, answer, key_facts, must) in enumerate(FACT_QUESTIONS, start=1):
        questions.append(
            {
                "id": idx,
                "category": "fact",
                "difficulty": "easy",
                "question": question,
                "ground_truth": build_ground_truth(answer, key_facts, must, tier=1),
            }
        )

    base = len(questions)
    for offset, (question, answer, key_facts) in enumerate(REQUIREMENT_QUESTIONS, start=1):
        questions.append(
            {
                "id": base + offset,
                "category": "requirement",
                "difficulty": "medium",
                "question": question,
                "ground_truth": build_ground_truth(answer, key_facts, key_facts, tier=2),
            }
        )

    base = len(questions)
    for offset, (question, answer, key_facts) in enumerate(COMPLEX_QUESTIONS, start=1):
        questions.append(
            {
                "id": base + offset,
                "category": "complex",
                "difficulty": "hard",
                "question": question,
                "ground_truth": build_ground_truth(answer, key_facts, key_facts[:2], tier=2),
            }
        )

    with DATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(questions, handle, indent=2)

    print(f"✅ Generated {len(questions)} test questions -> {DATA_PATH}")
    return questions


if __name__ == "__main__":
    generate_test_dataset()
