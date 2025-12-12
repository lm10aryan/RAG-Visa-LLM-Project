from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List

DB_PATH = Path("data/structured/h1b_facts.db")

FACTS: List[Dict[str, object]] = [
    {
        "category": "cap",
        "fact": "The regular H-1B statutory cap authorizes 65,000 new visas each fiscal year.",
        "metadata": {"cap": 65000, "fiscal_year": "annual"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "category": "cap",
        "fact": "An additional 20,000 H-1B visas are reserved for beneficiaries who hold a U.S. master’s or higher degree.",
        "metadata": {"cap": 20000, "beneficiaries": "US advanced degree"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "category": "registration",
        "fact": "Cap-subject petitioners must submit an electronic registration and be selected before filing an H-1B petition.",
        "metadata": {"requirement": "registration before filing"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "category": "registration",
        "fact": "Each H-1B registration submission requires a non-refundable $10 fee per beneficiary.",
        "metadata": {"fee_usd": 10, "form": "online registration"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "category": "timeline",
        "fact": "USCIS ran the FY 2024 initial H-1B registration between March 1 and March 17, 2023.",
        "metadata": {"window_start": "2023-03-01", "window_end": "2023-03-17"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-fiscal-year-fy-2024-cap-season",
        "source_title": "H-1B FY 2024 Cap Season",
    },
    {
        "category": "timeline",
        "fact": "USCIS completed the FY 2024 initial H-1B selection on March 27, 2023 and notified registrants.",
        "metadata": {"selection_date": "2023-03-27"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-fiscal-year-fy-2024-cap-season",
        "source_title": "H-1B FY 2024 Cap Season",
    },
    {
        "category": "registration",
        "fact": "Employers must use a USCIS online account to submit registrations electronically.",
        "metadata": {"channel": "myUSCIS online account"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "category": "registration",
        "fact": "Duplicate registrations for the same beneficiary and employer will render all associated registrations invalid.",
        "metadata": {"violation": "duplicate registration"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "category": "eligibility",
        "fact": "H-1B classification is for specialty occupations that require a bachelor’s degree or higher in a specific specialty.",
        "metadata": {"education_level": "bachelor or higher"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations",
        "source_title": "H-1B Specialty Occupations",
    },
    {
        "category": "employment",
        "fact": "The initial H-1B validity period can be granted for up to three years and extended to a maximum of six years in most cases.",
        "metadata": {"initial_years": 3, "max_years": 6},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations",
        "source_title": "H-1B Specialty Occupations",
    },
    {
        "category": "employment",
        "fact": "Employment in H-1B status generally cannot begin earlier than October 1 of the fiscal year for which the petition was approved.",
        "metadata": {"earliest_start": "October 1"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-cap-season",
        "source_title": "H-1B Cap Season",
    },
    {
        "category": "cap",
        "fact": "Institutions of higher education, related nonprofit entities, and government research organizations are cap-exempt.",
        "metadata": {"cap_exempt": True},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations",
        "source_title": "H-1B Specialty Occupations",
    },
    {
        "category": "filing",
        "fact": "A certified Labor Condition Application (LCA) from the Department of Labor is required before filing an H-1B petition.",
        "metadata": {"form": "ETA 9035"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/how-do-i-apply-for-h-1b-status",
        "source_title": "How Do I Apply for H-1B Status",
    },
    {
        "category": "filing",
        "fact": "Petitioners must file Form I-129, Petition for a Nonimmigrant Worker, for each H-1B beneficiary.",
        "metadata": {"form": "I-129"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/how-do-i-apply-for-h-1b-status",
        "source_title": "How Do I Apply for H-1B Status",
    },
    {
        "category": "fees",
        "fact": "The base filing fee for Form I-129 is $460 for H-1B petitions.",
        "metadata": {"fee_usd": 460, "form": "I-129"},
        "source_url": "https://www.uscis.gov/forms/h-and-l-filing-fees",
        "source_title": "H and L Filing Fees",
    },
    {
        "category": "fees",
        "fact": "The American Competitiveness and Workforce Improvement Act (ACWIA) fee is $1,500 for employers with 26 or more full-time employees, or $750 for 25 or fewer.",
        "metadata": {"fee_large": 1500, "fee_small": 750},
        "source_url": "https://www.uscis.gov/forms/h-and-l-filing-fees",
        "source_title": "H and L Filing Fees",
    },
    {
        "category": "fees",
        "fact": "Most initial H-1B petitions require the $500 Fraud Prevention and Detection fee.",
        "metadata": {"fee_usd": 500, "applies_to": "initial change of employer"},
        "source_url": "https://www.uscis.gov/forms/h-and-l-filing-fees",
        "source_title": "H and L Filing Fees",
    },
    {
        "category": "fees",
        "fact": "Certain employers with 50 or more employees, with more than half in H-1B or L-1 status, must pay an additional $4,000 Pub. L. 114-113 fee.",
        "metadata": {"fee_usd": 4000, "threshold": "50 employees, >50% H-1B/L"},
        "source_url": "https://www.uscis.gov/forms/h-and-l-filing-fees",
        "source_title": "H and L Filing Fees",
    },
    {
        "category": "process",
        "fact": "USCIS may conduct additional H-1B selections if needed to reach the cap based on FY 2024 guidance.",
        "metadata": {"policy": "multiple selection rounds possible"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/h-1b-fiscal-year-fy-2024-cap-season",
        "source_title": "H-1B FY 2024 Cap Season",
    },
    {
        "category": "compliance",
        "fact": "Employers must pay the higher of the prevailing wage or the actual wage paid to similarly qualified workers for the H-1B role.",
        "metadata": {"wage_rule": "higher of prevailing or actual"},
        "source_url": "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/how-do-i-apply-for-h-1b-status",
        "source_title": "How Do I Apply for H-1B Status",
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
            INSERT INTO facts (category, fact, metadata, source_url, source_title)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
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
