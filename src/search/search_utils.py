# src/search/search_utils.py
"""Utility functions used by both the search engine and Streamlit app."""

def format_case_card(case: dict) -> dict:
    """
    Clean and format a case dict for display in the UI.
    Handles missing fields gracefully.
    """
    return {
        "case_id":         case.get("case_id", "N/A"),
        "title":           case.get("case_title", "Untitled Case"),
        "year":            case.get("year", "Unknown"),
        "court":           case.get("court", "Unknown Court"),
        "outcome":         case.get("outcome", "unknown").title(),
        "petitioner":      case.get("petitioner", "N/A"),
        "respondent":      case.get("respondent", "N/A"),
        "judges":          case.get("judges", "N/A"),
        "keywords":        case.get("legal_keywords", ""),
        "citations":       case.get("citations", ""),
        "preview":         case.get("text_preview", "")[:400] + "...",
        "similarity":      case.get("similarity_score", 0.0),
        "outcome_color":   _outcome_color(case.get("outcome", "")),
    }


def _outcome_color(outcome: str) -> str:
    """Return a Streamlit color tag for the outcome badge."""
    mapping = {
        "allowed":        "green",
        "dismissed":      "red",
        "partly allowed": "orange",
        "remanded":       "blue",
        "unknown":        "gray",
    }
    return mapping.get(outcome.lower(), "gray")


def highlight_keywords(text: str, keywords: list) -> str:
    """Wrap matched keywords in markdown bold for display."""
    for kw in keywords:
        text = text.replace(kw, f"**{kw}**")
    return text


def parse_query_filters(year=None, outcome=None, court=None) -> dict:
    """Build a MongoDB filter dict from optional UI inputs."""
    filters = {}
    if year and year != "All":
        filters["year"] = str(year)
    if outcome and outcome != "All":
        filters["outcome"] = outcome.lower()
    if court and court != "All":
        filters["court"] = court
    return filters if filters else None