"""Numbers as the handout's reporting rule (C6) prints them."""


def fixed(v):
    """Four decimal places; a value that rounds to zero prints without a minus sign."""
    return f"{round(v, 4) + 0.0:.4f}"


def sci(v):
    """Scientific notation, for the residuals that are checked against 1e-8."""
    return f"{v:.1e}"
