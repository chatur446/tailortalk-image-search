import re


FABRIC_TERMS = {
    "silk": [
        "silk",
        "semi silk",
        "mysore silk",
        "kanchipuram silk",
        "banarasi silk",
    ],
    "banarasi": [
        "banarasi",
        "banaras",
    ],
    "tussar": [
        "tussar",
    ],
    "georgette": [
        "georgette",
    ],
    "organza": [
        "organza",
    ],
    "cotton": [
        "cotton",
    ],
    "pashmina": [
        "pashmina",
        "pasmina",
    ],
    "crepe": [
        "crepe",
        "crape",
    ],
    "kora": [
        "kora",
    ],
    "kanchipuram": [
        "kanchipuram",
    ],
    "chanderi": [
        "chanderi",
    ],
    "maheshwari": [
        "maheshwari",
    ],
}


COLOUR_TERMS = [
    "pink",
    "rani pink",
    "dusty rose",
    "rose",
    "red",
    "maroon",
    "wine",
    "burgundy",
    "blue",
    "navy blue",
    "navy",
    "royal blue",
    "sky blue",
    "turquoise blue",
    "turquoise",
    "teal",
    "green",
    "seafoam green",
    "mint green",
    "olive green",
    "pastel green",
    "yellow",
    "mustard",
    "lemon yellow",
    "orange",
    "sunset orange",
    "peach",
    "coral",
    "purple",
    "royal purple",
    "deep purple",
    "violet",
    "lavender",
    "pastel lavender",
    "cream",
    "white",
    "half white",
    "off white",
    "black",
    "gold",
    "silver",
    "silver grey",
    "grey",
    "gray",
    "beige",
    "magenta",
]


DESIGN_TERMS = [
    "floral",
    "floral print",
    "print",
    "printed",
    "embroidery",
    "embroidered",
    "zari",
    "border",
    "brocade",
    "motif",
    "checks",
    "checked",
    "stripe",
    "striped",
    "abstract",
    "mirror work",
    "chikankari",
    "kutch work",
    "jacquard",
    "sequence",
    "sequins",
    "paisley",
    "butta",
    "jaal",
    "ikat",
    "pochampally",
]


def normalize_text(text):
    """
    Normalize product text for matching.
    """

    text = str(text).lower()

    # Replace punctuation with spaces
    text = re.sub(
        r"[^a-z0-9\s&-]",
        " ",
        text
    )

    # Normalize repeated spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def extract_fabrics(name):
    """
    Extract fabric/material categories.
    """

    text = normalize_text(name)

    fabrics = []

    for category, terms in FABRIC_TERMS.items():

        for term in terms:

            if term in text:
                fabrics.append(category)
                break

    return sorted(set(fabrics))


def extract_colours(name):
    """
    Extract colour terms.
    """

    text = normalize_text(name)

    colours = []

    for colour in COLOUR_TERMS:

        if colour in text:
            colours.append(colour)

    return sorted(set(colours))


def extract_designs(name):
    """
    Extract design/pattern terms.
    """

    text = normalize_text(name)

    designs = []

    for design in DESIGN_TERMS:

        if design in text:
            designs.append(design)

    return sorted(set(designs))


def parse_product_name(name):
    """
    Extract structured metadata from a product name.
    """

    return {
        "fabrics": extract_fabrics(name),
        "colours": extract_colours(name),
        "designs": extract_designs(name),
    }