"""Names chosen in code, because a small model keeps picking the same few
(Kael, Kaelin, Elara...). The model writes the role; code supplies the name."""

from __future__ import annotations

import random
from typing import Iterable, Optional

FIRST_NAMES = (
    "Odalric", "Mirabel", "Tamsin", "Grimsby", "Ysolde", "Bertram", "Hesper", "Corwin", "Agatha", "Dunstan",
    "Sorcha", "Ilsabet", "Wulfric", "Perpetua", "Ansel", "Odette", "Rurik", "Linnea", "Fenwick", "Maeve",
    "Osric", "Zephyrine", "Halvard", "Brigid", "Tobias", "Nerys", "Casimir", "Philippa", "Garrick", "Ottoline",
    "Bram", "Isolde", "Magnus", "Cordelia", "Yarrow", "Jory", "Thessaly", "Eamon", "Quillon", "Radmila",
    "Aldous", "Beatrix", "Cormac", "Dorothea", "Emrys", "Fiora", "Gideon", "Hilda", "Ignatius", "Jocasta",
)

# One word each, so "First Last" can always be split on the first space.
LAST_NAMES = (
    "Venn", "Ostrander", "Rook", "Holt", "Marrow", "Quill", "Dunmore", "Fallowfield", "Pell", "Ballantyne",
    "Crane", "Ashdown", "Gale", "Merrick", "Brightwater", "Stonehelm", "Vask", "Tolliver", "Corrigan", "Plumb",
    "Moss", "Ketch", "Oakenshaw", "Wren", "Dray", "Sorrel", "Lowe", "Pryce", "Hollister", "Varga",
    "Teague", "Finch", "Penhallow", "Blackwell", "Marsh", "Vesper", "Abbot", "Ivers", "Harrow", "Whitlock",
    "Ravenscroft", "Underhill", "Kettering", "Mallory", "Sable", "Tarrant", "Lockwood", "Draycott", "Fenn", "Cobb",
)


def pick_patron_name(rng: Optional[random.Random] = None, exclude: Iterable[str] = ()) -> str:
    """A random 'First Last' name, never one in `exclude` (case-insensitive)."""
    r = rng or random
    banned = {e.strip().lower() for e in exclude}
    name = ""
    for _ in range(20):
        name = f"{r.choice(FIRST_NAMES)} {r.choice(LAST_NAMES)}"
        if name.lower() not in banned:
            return name
    for first in FIRST_NAMES:                     # nearly everything excluded: scan for a free one
        for last in LAST_NAMES:
            if f"{first} {last}".lower() not in banned:
                return f"{first} {last}"
    return name
