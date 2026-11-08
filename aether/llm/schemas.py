"""JSON schemas for every structured LLM call.

They are deliberately forgiving: only the field the call cannot work without
is required. Small models often leave optional fields out, and a missing
optional field should never cost a whole retry.

Text lengths are not limited here (schema length limits are not reliably
enforced by the model). `calls.py` truncates and cleans afterwards.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# --- options + referee (one call after each narration) -------------------------------------

class OptionItem(BaseModel):
    text: str = Field(..., description="The action, in a few words.")
    skill: Optional[str] = Field(None, description="The skill this action would test, or null if no roll is needed.")
    category: Optional[str] = Field(None, description="The slot label this action answers.")


class NewNpc(BaseModel):
    name: str = Field(..., description="Name of a character who appears in the scene.")
    type: Literal["ally", "enemy", "merchant"] = Field("ally", description="Role of the character.")
    is_persistent: bool = Field(False, description="True only for named, important recurring characters.")
    brief_backstory: Optional[str] = Field(None, description="One or two sentences, only if persistent.")
    template_id: Optional[str] = Field(None, description="Creature kind such as 'goblin', if applicable.")


class OptionsResponse(BaseModel):
    options: List[OptionItem]
    new_npcs: List[NewNpc] = Field(default_factory=list)
    beat_achieved: bool = False


# --- custom action judge ----------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    verdict: Literal["feasible", "risky", "impossible"]
    skill: Optional[str] = Field(None, description="Skill to roll for a risky action.")
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    reason: str = Field("", description="One short sentence.")
    advances_beat: bool = False


# --- compaction ---------------------------------------------------------------------------------

class NpcNote(BaseModel):
    name: str
    notes: str

class MemoryFacts(BaseModel):
    open_threads: List[str] = Field(default_factory=list)
    promises_debts: List[str] = Field(default_factory=list)


# --- story beats (campaign creation) ------------------------------------------------------------

class BeatDraft(BaseModel):
    title: str
    objective: str
    # A plain string on purpose: if the model proposes a type we no longer
    # allow, calls.py downgrades it instead of failing the whole reply.
    condition_type: str = "referee"
    condition_value: str = ""


class BeatsBlueprint(BaseModel):
    beats: List[BeatDraft]
