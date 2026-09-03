"""Rating-to-sentiment proxy labeling."""


def rating_to_label(rating: float) -> str:
    rating = int(rating)
    if rating <= 2:
        return "negatif"
    if rating == 3:
        return "netral"
    return "positif"
