from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List

DB_PATH = Path("data/structured/h1b_facts.db")

FACTS: List[Dict[str, object]] = [
    {
        "question": "What is the regular H-1B statutory cap?",
        "category": "cap",
        "fact": "The regular H-1B statutory cap authorizes 65,000 new visas each fiscal year.",
        "metadata": {"cap": 65000, "fiscal_year": "annual"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "question": "How many additional H-1B visas exist for advanced degree holders?",
        "category": "cap",
        "fact": "The H-1B cap includes an additional 20,000 H-1B visas reserved for beneficiaries who hold a U.S. master's or higher degree, on top of the regular 65,000 cap.",
        "metadata": {"cap": 20000, "beneficiaries": "US advanced degree"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "question": "Do I need to register before filing a cap-subject H-1B petition?",
        "category": "registration",
        "fact": "Cap-subject petitioners must submit an electronic registration and be selected before filing an H-1B petition.",
        "metadata": {"requirement": "registration before filing"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "question": "What is the H-1B registration fee per beneficiary?",
        "category": "registration",
        "fact": "Each H-1B registration submission requires a non-refundable $10 fee per beneficiary.",
        "metadata": {"fee_usd": 10, "form": "online registration"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "question": "When was the FY 2024 initial H-1B registration window?",
        "category": "timeline",
        "fact": "USCIS ran the FY 2024 initial H-1B registration between March 1 and March 17, 2023.",
        "metadata": {"window_start": "2023-03-01", "window_end": "2023-03-17"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-fiscal-year-fy-2024-cap-season",
        "source_title": "H-1B FY 2024 Cap Season",
    },
    {
        "question": "When did USCIS complete FY 2024 H-1B selections?",
        "category": "timeline",
        "fact": "USCIS completed the FY 2024 initial H-1B selection on March 27, 2023 and notified registrants.",
        "metadata": {"selection_date": "2023-03-27"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-fiscal-year-fy-2024-cap-season",
        "source_title": "H-1B FY 2024 Cap Season",
    },
    {
        "question": "How do employers submit H-1B registrations?",
        "category": "registration",
        "fact": "Employers must use a USCIS online account to submit registrations electronically.",
        "metadata": {"channel": "myUSCIS online account"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "question": "What happens if an employer submits duplicate H-1B registrations?",
        "category": "registration",
        "fact": "Duplicate registrations for the same beneficiary and employer will render all associated registrations invalid.",
        "metadata": {"violation": "duplicate registration"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "question": "What education level is required for H-1B specialty occupations?",
        "category": "eligibility",
        "fact": "H-1B classification is for specialty occupations that require a bachelor's degree or higher in a specific specialty.",
        "metadata": {"education_level": "bachelor or higher"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-specialty-occupations",
        "source_title": "H-1B Specialty Occupations",
    },
    {
        "question": "How long can H-1B status be granted?",
        "category": "employment",
        "fact": "The initial H-1B validity period can be granted for up to three years and extended to a maximum of six years in most cases.",
        "metadata": {"initial_years": 3, "max_years": 6},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-specialty-occupations",
        "source_title": "H-1B Specialty Occupations",
    },
    {
        "question": "When can H-1B employment begin in the fiscal year?",
        "category": "employment",
        "fact": "Employment in H-1B status generally cannot begin earlier than October 1 of the fiscal year for which the petition was approved.",
        "metadata": {"earliest_start": "October 1"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "question": "Who qualifies for H-1B cap exemption?",
        "category": "cap",
        "fact": "Institutions of higher education, related nonprofit entities, and government research organizations are cap-exempt.",
        "metadata": {"cap_exempt": True},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-specialty-occupations",
        "source_title": "H-1B Specialty Occupations",
    },
    {
        "question": "Do I need a Labor Condition Application for H-1B?",
        "category": "filing",
        "fact": "A certified Labor Condition Application (LCA) from the Department of Labor is required before filing an H-1B petition.",
        "metadata": {"form": "ETA 9035"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/how-do-i-apply-for-h-1b-status",
        "source_title": "How Do I Apply for H-1B Status",
    },
    {
        "question": "Which USCIS form is filed for H-1B classification?",
        "category": "filing",
        "fact": "Petitioners must file Form I-129, Petition for a Nonimmigrant Worker, for each H-1B beneficiary.",
        "metadata": {"form": "I-129"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/how-do-i-apply-for-h-1b-status",
        "source_title": "How Do I Apply for H-1B Status",
    },
    {
        "question": "What is the base filing fee for Form I-129 (H-1B)?",
        "category": "fees",
        "fact": "The base filing fee for Form I-129 is $460 for H-1B petitions.",
        "metadata": {"fee_usd": 460, "form": "I-129"},
        "source_url": "https://www.uscis.gov/forms/filing-fees",
        "source_title": "Filing Fees",
    },
    {
        "question": "How much does filing an H-1B petition cost?",
        "category": "fees",
        "fact": "Most employers pay the $460 base filing fee, the ACWIA fee of $750 or $1,500 depending on company size, and the $500 Fraud Prevention and Detection fee for initial H-1B petitions.",
        "metadata": {"base_fee": 460, "acwia_fee": "750/1500", "fraud_fee": 500},
        "source_url": "https://www.uscis.gov/forms/filing-fees",
        "source_title": "Filing Fees",
    },
    {
        "question": "What is the ACWIA fee for H-1B petitions?",
        "category": "fees",
        "fact": "The American Competitiveness and Workforce Improvement Act (ACWIA) fee is $1,500 for employers with 26 or more full-time employees, or $750 for 25 or fewer.",
        "metadata": {"fee_large": 1500, "fee_small": 750},
        "source_url": "https://www.uscis.gov/forms/filing-fees",
        "source_title": "Filing Fees",
    },
    {
        "question": "Do H-1B petitions require a fraud prevention fee?",
        "category": "fees",
        "fact": "Most initial H-1B petitions require the $500 Fraud Prevention and Detection fee.",
        "metadata": {"fee_usd": 500, "applies_to": "initial change of employer"},
        "source_url": "https://www.uscis.gov/forms/filing-fees",
        "source_title": "Filing Fees",
    },
    {
        "question": "When is the $4,000 Pub. L. 114-113 fee required?",
        "category": "fees",
        "fact": "Certain employers with 50 or more employees, with more than half in H-1B or L-1 status, must pay an additional $4,000 Pub. L. 114-113 fee.",
        "metadata": {"fee_usd": 4000, "threshold": "50 employees, >50% H-1B/L"},
        "source_url": "https://www.uscis.gov/forms/filing-fees",
        "source_title": "Filing Fees",
    },
    {
        "question": "Can USCIS run multiple H-1B selection rounds?",
        "category": "process",
        "fact": "USCIS may conduct additional H-1B selections if needed to reach the cap based on FY 2024 guidance.",
        "metadata": {"policy": "multiple selection rounds possible"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/h-1b-fiscal-year-fy-2024-cap-season",
        "source_title": "H-1B FY 2024 Cap Season",
    },
    {
        "question": "What wage must employers pay H-1B workers?",
        "category": "compliance",
        "fact": "Employers must pay the higher of the prevailing wage or the actual wage paid to similarly qualified workers for the H-1B role.",
        "metadata": {"wage_rule": "higher of prevailing or actual"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/how-do-i-apply-for-h-1b-status",
        "source_title": "How Do I Apply for H-1B Status",
    },
    {
        "question": "When can an H-1B cap-gap extension apply?",
        "category": "cap-gap",
        "fact": "Cap-subject H-1B petitions properly and timely filed for an eligible F-1 student requesting a change of status qualify for a cap-gap extension.",
        "metadata": {"beneficiary": "F-1 students with OPT"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/extension-of-post-completion-optional-practical-training-opt-and-f-1-status-for-eligible-students",
        "source_title": "Cap-Gap Extension Guidance",
    },
]


def create_facts_database(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            category TEXT NOT NULL,
            fact TEXT NOT NULL,
            metadata TEXT,
            source_url TEXT,
            source_title TEXT
        )
        """
    )
    cursor.execute("DELETE FROM facts")

    for fact in FACTS:
        cursor.execute(
            """
            INSERT INTO facts (question, category, fact, metadata, source_url, source_title)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                fact["question"],
                fact["category"],
                fact["fact"],
                json.dumps(fact.get("metadata", {}), ensure_ascii=False),
                fact["source_url"],
                fact["source_title"],
            ),
        )

    connection.commit()
    connection.close()


__all__ = ["create_facts_database", "FACTS"]
