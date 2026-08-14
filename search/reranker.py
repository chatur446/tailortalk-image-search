import pandas as pd


def split_values(value):
    """
    Convert pipe-separated metadata into a set.
    """

    if pd.isna(value) or not str(value).strip():
        return set()

    return {
        item.strip().lower()
        for item in str(value).split("|")
        if item.strip()
    }


def calculate_overlap(query_values, candidate_values):
    """
    Calculate how much metadata overlaps.

    Returns a score between 0 and 1.
    """

    if not query_values or not candidate_values:
        return 0.0

    intersection = query_values.intersection(candidate_values)

    if not intersection:
        return 0.0

    # Jaccard-style similarity
    union = query_values.union(candidate_values)

    return len(intersection) / len(union)


def metadata_similarity(query_row, candidate_row):
    """
    Calculate metadata similarity between two products.
    """

    query_fabrics = split_values(query_row["fabrics"])
    candidate_fabrics = split_values(candidate_row["fabrics"])

    query_colours = split_values(query_row["colours"])
    candidate_colours = split_values(candidate_row["colours"])

    query_designs = split_values(query_row["designs"])
    candidate_designs = split_values(candidate_row["designs"])

    fabric_score = calculate_overlap(
        query_fabrics,
        candidate_fabrics
    )

    colour_score = calculate_overlap(
        query_colours,
        candidate_colours
    )

    design_score = calculate_overlap(
        query_designs,
        candidate_designs
    )

    # Weighted metadata score
    score = (
        0.45 * fabric_score +
        0.35 * colour_score +
        0.20 * design_score
    )

    return score


def calculate_final_score(
    visual_score,
    metadata_score
):
    """
    Combine visual and metadata similarity.

    Visual similarity remains dominant.
    """

    return (
        0.75 * visual_score +
        0.25 * metadata_score
    )