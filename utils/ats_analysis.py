# utils/ats_analysis.py
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nltk.stem import PorterStemmer

# Reuse existing cleaning/stemming if needed, or reimplement for specific density checks
stemmer = PorterStemmer()


@dataclass
class FeedbackItem:
    problem: str
    impact: str
    fix: str
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class AnalysisResult:
    score: float
    max_score: float
    issues: List[FeedbackItem]
    details: Dict[str, any]


@dataclass
class ATSReport:
    master_score: float
    parsed_skills: List[str]
    missing_critical_skills: List[str]
    keyword_score: float
    format_score: float
    manipulation_score: float
    feedback: List[FeedbackItem]
    pass_probability: str  # Low, Medium, High


from utils.analysis import STOP_WORDS


class ATSAnalyzer:
    def __init__(self):
        self.stop_words = STOP_WORDS

    def _clean_tokens(self, text: str) -> List[str]:
        # Simple tokenization for density analysis - preserve accented chars
        # Replace non-alphanumeric (except + and #) with space to avoid merging e.g. "SQLAlchemy/SQLModel"
        text = re.sub(r"[^\w\s\+#]", " ", text)
        words = text.lower().split()
        return [w for w in words if w not in self.stop_words and len(w) > 2]

    def analyze_keywords(
        self, resume_text: str, jd_text: str, critical_skills: List[str] = None
    ) -> AnalysisResult:
        """
        Analyzes keyword density and critical skills coverage.
        """
        resume_tokens = self._clean_tokens(resume_text)
        jd_tokens = self._clean_tokens(jd_text)

        resume_counter = Counter(resume_tokens)
        jd_counter = Counter(jd_tokens)

        # Prepare normalized text for phrase matching
        resume_text_lower = resume_text.lower()

        issues = []
        score = 100

        # 1. Critical Skills Check
        # If no explicit critical skills list provided, derive top 5 frequent words from JD
        if not critical_skills:
            # Simple heuristic: top 10 most frequent words in JD that are not stopwords
            critical_skills = [word for word, count in jd_counter.most_common(10)]

        missing_critical = []
        for skill in critical_skills:
            # Normalize skill
            skill_norm = skill.lower().strip()

            # Check 1: Exact phrase match in full text (handles "Async Python", "C++")
            if skill_norm in resume_text_lower:
                match_count = resume_text_lower.count(skill_norm)
            else:
                # Check 2: Fallback to token counter if it's a single word
                match_count = resume_counter.get(skill_norm, 0)

            if match_count == 0:
                score -= 20
                missing_critical.append(skill)
                # Ensure we don't report missing if it's actually there just with punctuation
                # Use regex bound check as a final fallback for single words
                # (skipped for brevity, assuming string match is good enough for now)

                issues.append(
                    FeedbackItem(
                        problem=f"Critical skill missing: '{skill}'",
                        impact="ATS often rejects resumes missing exact matches for top requirements.",
                        fix=f"Add the exact keyword '{skill}' to your Skills or Experience section.",
                        severity="high",
                    )
                )
            elif match_count < 2:
                # Weak match
                score -= 2
                issues.append(
                    FeedbackItem(
                        problem=f"Weak keyword validation: '{skill}'",
                        impact="Mentioning a key skill only once may look like an accident.",
                        fix=f"Use '{skill}' at least 2-3 times in different contexts.",
                        severity="low",
                    )
                )

        # 2. Density Check (Ratio)
        # We don't want deep deep analysis here for this MVP, just check if overall vocabulary overlaps well?
        # Let's stick to the prompt's "0 vs 3+" logic which we handled above basically.

        return AnalysisResult(
            score=max(0, score),
            max_score=100,
            issues=issues,
            details={"missing_critical": missing_critical},
        )

    def analyze_formatting(self, resume_text: str) -> AnalysisResult:
        """
        Detects ATS-unfriendly formatting risks.
        """
        issues = []
        score = 100

        # 1. Tables detection (heuristic: many "|" or widely spaced columns in text view?)
        # Since we use pdfplumber/docx, tables might come out as linear text.
        # But if we see many vertical bars, that's a sign of ASCII tables or bad extraction.
        if resume_text.count("|") > 5:
            # Often used as separators "Skill | Skill", which is fine, but sometimes tables.
            # Let's check for "│" or box characters
            pass

        if "│" in resume_text or "┌" in resume_text:
            score -= 20
            issues.append(
                FeedbackItem(
                    problem="Detected table borders or grid characters.",
                    impact="Old ATS parsers often scramble content inside tables.",
                    fix="Use standard columns or tabs instead of tables. Keep layout simple.",
                    severity="high",
                )
            )

        # 2. Graphics placeholders (pdfplumber often outputs [Image] or similar if configured,
        # but here we deal with raw text. We can only detect what's in text.)
        # Heuristic: Check for excessive whitespace or weird gaps?
        if "\n\n\n\n" in resume_text:
            score -= 5
            issues.append(
                FeedbackItem(
                    problem="Large vertical gaps detected.",
                    impact="Might indicate text boxes or empty sections that confuse parsers.",
                    fix="Remove extra newlines and check for floating text boxes.",
                    severity="medium",
                )
            )

        # 3. Non-standard Headers
        standard_sections = [
            # English
            "experience",
            "education",
            "skills",
            "projects",
            "summary",
            "profile",
            "work history",
            # Czech
            "pracovní zkušenosti",
            "zkušenosti",
            "profesní zkušenosti",
            "vzdělání",
            "dovednosti",
            "technické dovednosti",
            "jazyky",
            "projekty",
            "shrnutí",
            "profil",
        ]
        found_sections = 0
        lines = resume_text.split("\n")
        # Check first few words of lines that look like headers (short, uppercase or capitalized)
        for line in lines:
            clean_line = line.strip().lower()
            # Allow some extra chars like colons e.g. "Skills:"
            clean_line = clean_line.replace(":", "")
            if len(clean_line) < 40 and clean_line in standard_sections:
                found_sections += 1

        if found_sections < 2:
            score -= 15
            issues.append(
                FeedbackItem(
                    problem="Could not clearly identify standard section headers.",
                    impact="ATS parsers rely on headers like 'Experience' to segment your data.",
                    fix="Use standard headers: Experience, Education, Skills. Avoid 'Where I've Been' etc.",
                    severity="high",
                )
            )

        return AnalysisResult(
            score=max(0, score),
            max_score=100,
            issues=issues,
            details={"found_sections": found_sections},
        )

    def analyze_manipulation(self, resume_text: str) -> AnalysisResult:
        """
        Detects keyword stuffing and hidden text.
        """
        issues = []
        score = 100
        tokens = self._clean_tokens(resume_text)
        counts = Counter(tokens)

        # 1. Keyword Stuffing
        for word, count in counts.most_common(5):
            if count > 15:  # Arbitrary threshold
                score -= 10
                issues.append(
                    FeedbackItem(
                        problem=f"Excessive modification: '{word}' appears {count} times.",
                        impact="Looks like keyword stuffing to rank higher.",
                        fix=f"Reduce usage of '{word}'. Use synonyms.",
                        severity="medium",
                    )
                )

        # 2. Invisible chars (Zero width space, etc)
        # \u200b is zero width space
        if "\u200b" in resume_text or "\ufeff" in resume_text:
            score -= 50
            issues.append(
                FeedbackItem(
                    problem="Hidden characters detected (Zero Width Spaces).",
                    impact="Immediate red flag for fraud detection systems.",
                    fix="Remove all invisible formatting characters.",
                    severity="critical",
                )
            )

        # 3. White text (difficult to detect in raw text without OCR/PDF metadata,
        # but sometimes it manifests as weird repeated blocks).
        # We'll skip for text-only version unless we have advanced PDF analysis.

        return AnalysisResult(
            score=max(0, score), max_score=100, issues=issues, details={}
        )

    def calculate_master_score(
        self, resume_text: str, jd_text: str, jd_data: dict = None
    ) -> ATSReport:
        # Get extracted criteria from JD structured data if available
        critical_skills = jd_data.get("required_skills", []) if jd_data else []
        # Fallback to text analysis if no structured data

        kw_result = self.analyze_keywords(resume_text, jd_text, critical_skills)
        fmt_result = self.analyze_formatting(resume_text)
        man_result = self.analyze_manipulation(resume_text)

        # Weighted Average
        # Keyword: 60%, Format: 20%, Manipulation: 20%
        master_score = (
            (kw_result.score * 0.6)
            + (fmt_result.score * 0.2)
            + (man_result.score * 0.2)
        )

        # Probability bucket
        if master_score >= 80:
            prob = "High"
        elif master_score >= 50:
            prob = "Medium"
        else:
            prob = "Low"

        all_issues = kw_result.issues + fmt_result.issues + man_result.issues
        # Sort by severity (Critical -> High -> Medium -> Low)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_issues.sort(key=lambda x: severity_order.get(x.severity, 4))

        return ATSReport(
            master_score=round(master_score, 1),
            parsed_skills=[],  # Placeholder
            missing_critical_skills=kw_result.details.get("missing_critical", []),
            keyword_score=kw_result.score,
            format_score=fmt_result.score,
            manipulation_score=man_result.score,
            feedback=all_issues,
            pass_probability=prob,
        )


# Example Usage / Unit Test
if __name__ == "__main__":
    analyzer = ATSAnalyzer()

    sample_resume = """
    EXPERIENCE
    Software Engineer | Google | 2020-Present
    - Wrote python code.
    - Used AWS.

    EDUCATION
    B.S. CS
    """

    sample_jd = """
    We need a Python expert with AWS and Docker experience.
    Critical skills: Python, AWS, Docker, Kubernetes.
    """

    report = analyzer.calculate_master_score(sample_resume, sample_jd)
    print(f"Master Score: {report.master_score}")
    print(f"Pass Prob: {report.pass_probability}")
    for item in report.feedback:
        print(f"[{item.severity.upper()}] {item.problem} -> {item.fix}")
