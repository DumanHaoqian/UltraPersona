"""Build the final edit instruction from three parts: background, people count, style.

Image order sent to the model: Picture 1 = the user's photo, Picture 2 = the PolyU background.
"""
from __future__ import annotations

NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

KEEP_ID = ("Keep each person's face, facial features, hairstyle, skin tone and clothing exactly as in "
           "Picture 1. Do not add accessories that are not in Picture 1.")
KEEP_BG = "Keep the buildings, signs and logos of Picture 2 unchanged."
PHOTO_FINISH = ("Match the lighting, shadows, colour and perspective of Picture 2 so the result looks like "
                "one real photograph.")


def people_part(n: int, framing: str) -> str:
    """Wording depends on how many people were detected and how they are framed."""
    body = {"closeup": "shown from the chest up", "half_body": "shown from the waist up",
            "wide": "shown full body"}.get(framing, "")
    if n == 1:
        return f"Place the person from Picture 1, {body}, " if body else "Place the person from Picture 1 "
    word = NUM_WORDS.get(n, str(n))
    order = "in the same left-to-right order as in Picture 1, standing close together like a group photo"
    return f"Place all {word} people from Picture 1 ({word} people in total, nobody added or removed), {order}, "


def compose_prompt(n: int, framing: str, bg: dict) -> str:
    """Step 1 of the two-step method, and the photo-only output."""
    return (f"{people_part(n, framing)}{bg['stand']} in Picture 2 ({bg['scene']}). "
            f"{KEEP_ID} {KEEP_BG} {PHOTO_FINISH}")


def stylize_prompt(n: int, style: dict) -> str:
    """Step 2 of the two-step method. The only input image is the composed photo."""
    who = "the person" if n == 1 else f"all {NUM_WORDS.get(n, n)} people"
    return (f"Turn this photo into {style['prompt']}. Keep the composition, the background layout and "
            f"{who} in the same places. Keep each face recognisable as the same person: same face shape, "
            f"eyes, nose, mouth and hairstyle.")


def onestep_prompt(n: int, framing: str, bg: dict, style: dict) -> str:
    """One-step method: compose and stylize in a single edit."""
    return (f"{people_part(n, framing)}{bg['stand']} in Picture 2 ({bg['scene']}). "
            f"{KEEP_ID} Keep the layout of Picture 2. Render the whole picture as {style['prompt']}, "
            f"while every face stays recognisable as the same person from Picture 1.")


NEGATIVE = ("extra people, missing people, duplicated person, deformed face, distorted face, extra fingers, "
            "blurry, low quality, watermark, text artifacts")
