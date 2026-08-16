import pandas as pd


def split_values(value):
    """
    Convert pipe-separated metadata into a normalized set.
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
    Calculate query-oriented metadata overlap.

    Returns:
        0.0 to 1.0
    """

    if not query_values or not candidate_values:
        return 0.0

    intersection = query_values.intersection(candidate_values)

    if not intersection:
        return 0.0

    return len(intersection) / len(query_values)


def metadata_similarity(query_row, candidate_row):
    """
    Calculate metadata similarity between two products.

    Fabric, colour and design have different importance.
    Missing metadata does not count as a match, but it also
    does not receive the full penalty of an explicit mismatch.
    """

    query_fabrics = split_values(query_row["fabrics"])
    candidate_fabrics = split_values(candidate_row["fabrics"])

    query_colours = split_values(query_row["colours"])
    candidate_colours = split_values(candidate_row["colours"])

    query_designs = split_values(query_row["designs"])
    candidate_designs = split_values(candidate_row["designs"])

    # Calculate overlaps only when query metadata exists.
    fabric_score = (
        calculate_overlap(query_fabrics, candidate_fabrics)
        if query_fabrics and candidate_fabrics
        else None
    )

    colour_score = (
        calculate_overlap(query_colours, candidate_colours)
        if query_colours and candidate_colours
        else None
    )

    design_score = (
        calculate_overlap(query_designs, candidate_designs)
        if query_designs and candidate_designs
        else None
    )

    # Original importance of each attribute
    weights = {
        "fabric": 0.40,
        "colour": 0.40,
        "design": 0.20
    }

    scores = {
        "fabric": fabric_score,
        "colour": colour_score,
        "design": design_score
    }

    weighted_score = 0.0
    available_weight = 0.0

    for attribute in scores:

        score = scores[attribute]
        weight = weights[attribute]

        if score is not None:
            weighted_score += score * weight
            available_weight += weight

        else:
            # Missing metadata gets partial credit for neutrality,
            # rather than being treated as either a match or mismatch.
            weighted_score += 0.25 * weight
            available_weight += weight

    if available_weight == 0:
        return 0.0

    return weighted_score / available_weight


def calculate_final_score(
    visual_score,
    metadata_score
):
    """
    Combine visual and metadata similarity.

    Visual similarity remains dominant.
    """

    return (
        0.85 * visual_score +
        0.15 * metadata_score
    )