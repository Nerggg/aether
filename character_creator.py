import os
import re
import sqlite3
import ollama
from pydantic import BaseModel, Field
from typing import Literal, Optional

from md_manager import MarkdownManager
from llm_manager import LLMController, PromptBuilder
from db import get_db_connection
from vector_db_manager import VectorDBManager

ALLOWED_RACES = Literal["Human", "Elf", "Dwarf", "Halfling", "Dragonborn", "Gnome", "Tiefling"]
ALLOWED_CLASSES = Literal["Fighter", "Rogue", "Wizard", "Cleric", "Paladin", "Ranger", "Bard"]

DND_RULES_TEXT = """
APPROVED CHARACTER CREATION OPTIONS:

Approved Races:
- **Human:** +1 to all stats. Base speed: 30ft.
- **Elf:** +2 Dexterity. Speed: 30ft. Keen Senses.
- **Dwarf:** +2 Constitution. Speed: 25ft. Poison resistance.
- **Halfling:** +2 Dexterity. Speed: 25ft. Lucky (re-roll 1s on d20).
- **Dragonborn:** +2 Strength, +1 Charisma. Elemental breath weapon.
- **Gnome:** +2 Intelligence. Speed: 25ft. Magic resistance saves.
- **Tiefling:** +2 Charisma, +1 Intelligence. Fire resistance.

Approved Classes:
- **Fighter:** Hit Die: d10. Armor: Heavy (Chain Mail).
- **Rogue:** Hit Die: d8. Armor: Light (Leather).
- **Wizard:** Hit Die: d6. Armor: None.
- **Cleric:** Hit Die: d8. Armor: Medium (Scale Mail).
- **Paladin:** Hit Die: d10. Armor: Heavy (Chain Mail).
- **Ranger:** Hit Die: d10. Armor: Medium (Scale Mail).
- **Bard:** Hit Die: d8. Armor: Light (Leather).
"""


class CharacterChecklistParser(BaseModel):
    """Extraction schema to capture player options during conversation."""
    name: Optional[str] = Field(None, description="The chosen character name if mentioned, otherwise null.")
    race: Optional[ALLOWED_RACES] = Field(None, description="The chosen race from the allowed list if mentioned, otherwise null.")
    character_class: Optional[ALLOWED_CLASSES] = Field(None, description="The chosen class from the allowed list if mentioned, otherwise null.")
    backstory: Optional[str] = Field(None, description="A brief summary of their background, personality, or origins if discussed, otherwise null.")


class CompiledCharacterPayload(BaseModel):
    """Schema to compile the final backstory and clean character inputs."""
    name: str = Field(..., description="The finalized character name.")
    race: ALLOWED_RACES = Field(..., description="Must be selected strictly from the approved races list.")
    character_class: ALLOWED_CLASSES = Field(..., description="Must be selected strictly from the approved classes list.")
    background_lore: str = Field(..., description="A rich backstory linking the character to the active campaign.")


class CharacterCreator:
    
    def __init__(self, campaign_slug: str):
        self.campaign_slug = campaign_slug
        self.md_manager = MarkdownManager()
        self.vector_db = VectorDBManager()
        self.llm = LLMController()
        self.chat_history = []

        self.checklist = {
            "name": None,
            "race": None,
            "character_class": None,
            "backstory": None
        }

    @staticmethod
    def slugify(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"[-\s]+", "_", text)

    def run_creation_loop(self) -> None:
        """Runs the interactive console creation loop."""
        print(DND_RULES_TEXT)
        print("\n=======================================================")
        print("          AETHER: CHARACTER CREATION TERMINAL")
        print("=======================================================")
        print(f"Active Campaign: {self.campaign_slug}")
        print("Introduce your character's name, race, and class to begin!")
        print("Once all parameters are decided, the sheet will automatically write.\n")

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() == "complete":
                print("\nProcessing finalized character... Please wait.")
                self.generate_and_save_character()
                break

            self.chat_history.append({"role": "user", "content": user_input})

            if len(self.chat_history) >= 2:
                try:
                    parser_system = """
                    You are a strict data extraction system. Analyze the recent conversation history and extract the D&D character options into the required JSON schema format.
                    Only extract choices explicitly decided by the user. Default to null for missing fields. Do not invent details.
                    """
                    parser_messages = [
                        {"role": "system", "content": parser_system}
                    ] + self.chat_history[-4:]
                    
                    parsed_response = self.llm.generate_structured_action(
                        messages=parser_messages,
                        response_schema=CharacterChecklistParser,
                        model=self.llm.referee_model,
                        temperature=0.0
                    )
                    
                    if parsed_response:
                        if parsed_response.name: self.checklist["name"] = parsed_response.name
                        if parsed_response.race: self.checklist["race"] = parsed_response.race
                        if parsed_response.character_class: self.checklist["character_class"] = parsed_response.character_class
                        if parsed_response.backstory: self.checklist["backstory"] = parsed_response.backstory
                        
                    print(f"[Checklist Status] Name: {'OK' if self.checklist['name'] else 'PENDING'} | Race: {'OK' if self.checklist['race'] else 'PENDING'} | Class: {'OK' if self.checklist['character_class'] else 'PENDING'} | Backstory: {'OK' if self.checklist['backstory'] else 'PENDING'}")
                except Exception as e:
                    print(f"[Debug] Parsing warning: {e}")

            if all(self.checklist.values()):
                print(f"\n[System] All core traits established: {self.checklist}")
                self.generate_and_save_character()
                break

            active_persona = f"""
            You are the Character Creation Guide. Help the player build a D&D character.
            
            {DND_RULES_TEXT}
            
            Here is the current status of the character checklist:
            - name: {'COMPLETED ({})'.format(self.checklist['name']) if self.checklist['name'] else 'PENDING'}
            - race: {'COMPLETED ({})'.format(self.checklist['race']) if self.checklist['race'] else 'PENDING'}
            - character_class: {'COMPLETED ({})'.format(self.checklist['character_class']) if self.checklist['character_class'] else 'PENDING'}
            - backstory: {'COMPLETED' if self.checklist['backstory'] else 'PENDING'}

            INSTRUCTIONS:
            - Focus on asking questions to resolve the PENDING traits. Ask about ONE trait at a time.
            - You must ALWAYS present your questions as a multiple-choice selection structured EXACTLY in this format:
              A) [Suggestion 1]
              B) [Suggestion 2]
              C) [Suggestion 3]
              D) Something else / Player's custom input
            - Keep your introductory text highly concise (under 50 words) and speak in a highly collaborative, creative, and immersive world-building tone.
            """
            
            messages = PromptBuilder.compile_messages(
                system_persona=active_persona,
                rag_context_chunks=[],
                chat_history=self.chat_history
            )
            
            print("\nGuide: ", end="", flush=True)
            response_chunks = []
            for chunk in self.llm.generate_narrative_stream(messages, model=self.llm.model_name, temperature=0.7):
                print(chunk, end="", flush=True)
                response_chunks.append(chunk)
            print()
            
            response = "".join(response_chunks)
            self.chat_history.append({"role": "assistant", "content": response})

    def _calculate_character_stats(self, race: str, char_class: str):
        """Pure Python D&D 5e Mechanics Engine."""
        stat_priorities = {
            "Fighter": ["strength", "constitution", "dexterity", "wisdom", "charisma", "intelligence"],
            "Rogue": ["dexterity", "intelligence", "charisma", "constitution", "wisdom", "strength"],
            "Wizard": ["intelligence", "dexterity", "constitution", "wisdom", "charisma", "strength"],
            "Cleric": ["wisdom", "constitution", "strength", "intelligence", "dexterity", "charisma"],
            "Paladin": ["strength", "charisma", "constitution", "wisdom", "dexterity", "intelligence"],
            "Ranger": ["dexterity", "wisdom", "constitution", "strength", "intelligence", "charisma"],
            "Bard": ["charisma", "dexterity", "constitution", "wisdom", "intelligence", "strength"]
        }
        
        standard_array = [15, 14, 13, 12, 10, 8]
        priority = stat_priorities.get(char_class, ["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"])
        
        stats = {priority[i]: standard_array[i] for i in range(6)}
        
        racial_mods = {
            "Human": {"strength": 1, "dexterity": 1, "constitution": 1, "intelligence": 1, "wisdom": 1, "charisma": 1},
            "Elf": {"dexterity": 2},
            "Dwarf": {"constitution": 2},
            "Halfling": {"dexterity": 2},
            "Dragonborn": {"strength": 2, "charisma": 1},
            "Gnome": {"intelligence": 2},
            "Tiefling": {"charisma": 2, "intelligence": 1}
        }
        
        mods = racial_mods.get(race, {})
        for stat, bonus in mods.items():
            stats[stat] += bonus
            
        def get_mod(score):
            return (score - 10) // 2
            
        dex_mod = get_mod(stats["dexterity"])
        con_mod = get_mod(stats["constitution"])
        
        hit_dies = {
            "Fighter": 10, "Paladin": 10, "Ranger": 10,
            "Rogue": 8, "Cleric": 8, "Bard": 8,
            "Wizard": 6
        }
        hp = hit_dies.get(char_class, 8) + con_mod
        
        if char_class in ["Fighter", "Paladin"]:
            ac = 16 
        elif char_class in ["Rogue", "Bard"]:
            ac = 11 + dex_mod
        elif char_class in ["Cleric", "Ranger"]:
            ac = 14 + min(2, dex_mod)
        else:
            ac = 10 + dex_mod
            
        return stats, hp, ac

    def generate_and_save_character(self) -> None:
        print("\n[1/3] Compiling finalized character sheet and background narrative...")
        
        compiler_prompt = f"""
        Analyze the character creation history and translate the chosen options into the required CompiledCharacterPayload JSON.
        Make sure you select the race and class strictly from the approved lists:
        Races: {", ".join(ALLOWED_RACES.__args__)}
        Classes: {", ".join(ALLOWED_CLASSES.__args__)}
        """
        
        compiled_messages = PromptBuilder.compile_messages(
            system_persona=compiler_prompt,
            rag_context_chunks=[],
            chat_history=self.chat_history
        )
        
        try:
            response = ollama.chat(
                model=self.llm.referee_model,
                messages=compiled_messages,
                format=CompiledCharacterPayload.model_json_schema(),
                options={"temperature": 0.5}
            )
            raw_json = response["message"]["content"]
            character_data = CompiledCharacterPayload.model_validate_json(raw_json)
            
            print("[2/3] Executing deterministic math operations for D&D statistics...")
            stats, hp, ac = self._calculate_character_stats(character_data.race, character_data.character_class)
            
            print("[3/3] Saving compiled character profile to disk and database...")
            self._save_to_database_and_disk(character_data, stats, hp, ac)
            
        except Exception as e:
            print(f"\n[Error] Character sheet compilation failed: {e}")
            print("Please run creation again or refine details.")

    def _save_to_database_and_disk(self, data: CompiledCharacterPayload, stats: dict, hp: int, ac: int) -> None:
        char_slug = self.slugify(data.name)
        starting_items = ["Leather armor", "Short sword", "Stale bread"]
        
        actor_metadata = {
            "id": char_slug,
            "campaign_slug": self.campaign_slug,
            "name": data.name,
            "race": data.race,
            "class": data.character_class,
            "type": "player",
            "status": "alive",
            "inventory": starting_items
        }
        
        inventory_str = ", ".join(starting_items)
        
        actor_content = (
            f"# Character Sheet: {data.name}\n\n"
            f"**Race:** {data.race} | **Class:** {data.character_class}\n\n"
            f"## Backstory & Background Lore\n{data.background_lore}\n\n"
            f"## Determined Stats (Standard Array + Racial)\n"
            f"- STR: {stats['strength']} | DEX: {stats['dexterity']} | CON: {stats['constitution']}\n"
            f"- INT: {stats['intelligence']} | WIS: {stats['wisdom']} | CHA: {stats['charisma']}\n\n"
            f"## Combat Values\n"
            f"- **HP:** {hp}/{hp}\n"
            f"- **AC:** {ac}\n\n"
            f"## Inventory & Equipment\n"
            f"- {inventory_str}\n"
        )
        
        filename = f"{self.campaign_slug}_{char_slug}"
        self.md_manager.write_file(
            category="actors",
            filename=filename,
            metadata=actor_metadata,
            content=actor_content
        )

        self.vector_db.upsert_markdown_file(category="actors", filename=filename)

        initiative_bonus = (stats["dexterity"] - 10) // 2
        
        conn = get_db_connection(self.campaign_slug)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO characters (name, type, strength, dexterity, constitution, intelligence, wisdom, charisma, hp, max_hp, ac, initiative_bonus)
            VALUES (?, 'player', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.name, 
                stats["strength"], stats["dexterity"], stats["constitution"], 
                stats["intelligence"], stats["wisdom"], stats["charisma"],
                hp, hp, ac, initiative_bonus
            ))
            conn.commit()
            
            print("\n=======================================================")
            print("          SUCCESS: CHARACTER REGISTERED")
            print("=======================================================")
            print(f"SQLite Profile Path: data/campaigns/{self.campaign_slug}/game.db")
            print(f"Markdown Sheet Path: data/actors/{filename}.md")
            print(f"Calculated Stats:    HP: {hp} | AC: {ac} | DEX-Mod: {initiative_bonus}")
            print("=======================================================\n")
            
        except sqlite3.Error as e:
            print(f"[Error] SQLite insertion failed during character registration: {e}")
        finally:
            conn.close()


if __name__ == "__main__":
    creator = CharacterCreator(campaign_slug="whispers_of_the_ancient_dark")
    creator.run_creation_loop()
