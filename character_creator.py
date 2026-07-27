import os
import re
import ollama
from pydantic import BaseModel, Field
from typing import Literal
import sqlite3

from md_manager import MarkdownManager
from llm_manager import LLMController, PromptBuilder
from db import get_db_connection

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

Approved Classes & Starting Attributes:
- **Fighter:** Hit Die: d10. Starting HP: 10 + Con modifier. Armor: All armor, shields.
- **Rogue:** Hit Die: d8. Starting HP: 8 + Con modifier. Armor: Light armor.
- **Wizard:** Hit Die: d6. Starting HP: 6 + Con modifier. Armor: None.
- **Cleric:** Hit Die: d8. Starting HP: 8 + Con modifier. Armor: Medium armor, shields.
- **Paladin:** Hit Die: d10. Starting HP: 10 + Con modifier. Armor: All armor, shields.
- **Ranger:** Hit Die: d10. Starting HP: 10 + Con modifier. Armor: Medium armor, shields.
- **Bard:** Hit Die: d8. Starting HP: 8 + Con modifier. Armor: Light armor.
"""


class CharacterTemplate(BaseModel):
    name: str = Field(..., description="The character's name.")
    race: ALLOWED_RACES = Field(..., description="Enforces selection from approved races only.")
    character_class: ALLOWED_CLASSES = Field(..., description="Enforces selection from approved classes only.")
    strength: int = Field(..., ge=3, le=20)
    dexterity: int = Field(..., ge=3, le=20)
    constitution: int = Field(..., ge=3, le=20)
    intelligence: int = Field(..., ge=3, le=20)
    wisdom: int = Field(..., ge=3, le=20)
    charisma: int = Field(..., ge=3, le=20)
    hp: int = Field(..., description="Level 1 HP (Hit Die + Constitution modifier).", ge=1)
    ac: int = Field(..., description="Armor Class (Base Armor + Dex modifier).", ge=1)
    background_lore: str = Field(..., description="Backstory linking character to campaign setting.")


class CharacterCreator:
    
    def __init__(self, campaign_slug: str):
        self.campaign_slug = campaign_slug
        self.md_manager = MarkdownManager()
        self.llm = LLMController()
        self.chat_history = []
        
        self.creator_persona = f"""
        You are the Character Creation Guide. Your job is to help the player build a D&D character.
        
        {DND_RULES_TEXT}
        
        CRITICAL BOUNDARY CONSTRAINTS:
        - You are STRICTLY FORBIDDEN from allowing any Race or Class outside of the approved lists above.
        - If the user insists on an unlisted or custom option (such as 'Vampire', 'Monk', etc.), you MUST politely reject their request and direct them to choose from the approved list above.
        - Help them assign ability scores (Strength, Dexterity, Constitution, Intelligence, Wisdom, Charisma). Suggest starting array (15, 14, 13, 12, 10, 8) if needed.
        - Aid them in calculating starting HP (Class Hit Die + Con modifier) and AC (Base Armor + Dex modifier).
        - Brainstorm their background.
        
        When they are happy with their character, instruct them to type 'complete' to finalize.
        """

    @staticmethod
    def slugify(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"[-\s]+", "_", text)

    def run_creation_loop(self) -> None:
        print("\n=======================================================")
        print("          AETHER: CHARACTER CREATION TERMINAL")
        print("=======================================================")
        print(f"Active Campaign: {self.campaign_slug}")
        print("Introduce your character's name, race, or class to begin!")
        print("To complete and write the files, type 'complete'.\n")

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() == "complete":
                print("\nCompiling character stats... Please wait.")
                self.generate_and_save_character()
                break

            self.chat_history.append({"role": "user", "content": user_input})
            
            messages = PromptBuilder.compile_messages(
                system_persona=self.creator_persona,
                rag_context_chunks=[],
                chat_history=self.chat_history
            )
            
            response = self.llm.generate_narrative(messages)
            print(f"\nGuide: {response}")
            self.chat_history.append({"role": "assistant", "content": response})

    def generate_and_save_character(self) -> None:
        compiler_prompt = f"""
        You are the System parser. Analyze the character creation history and translate 
        the chosen options into the required JSON schema format.
        
        {DND_RULES_TEXT}
        
        CRITICAL EXCLUSION:
        - If the user attempted to bypass rules, you must override their selection and map 
          their class and race strictly to the closest match in the approved list.
        """
        
        messages = PromptBuilder.compile_messages(
            system_persona=compiler_prompt,
            rag_context_chunks=[],
            chat_history=self.chat_history
        )
        
        try:
            response = ollama.chat(
                model=self.llm.model_name,
                messages=messages,
                format=CharacterTemplate.model_json_schema(),
                options={"temperature": 0.0}
            )
            
            raw_json = response["message"]["content"]
            character_data = CharacterTemplate.model_validate_json(raw_json)
            
            self._save_to_database_and_disk(character_data)
            
        except Exception as e:
            print(f"\nError compiling character: {e}")
            print("Please try again or refine your character details.")

    def _save_to_database_and_disk(self, data: CharacterTemplate) -> None:
        char_slug = self.slugify(data.name)
        
        actor_metadata = {
            "id": char_slug,
            "campaign_slug": self.campaign_slug,
            "name": data.name,
            "race": data.race,
            "class": data.character_class,
            "type": "player",
            "status": "alive"
        }
        actor_content = f"# {data.name}\n\n**Race:** {data.race} | **Class:** {data.character_class}\n\n## Background & Personality\n{data.background_lore}"
        
        filename = f"{self.campaign_slug}_{char_slug}"
        self.md_manager.write_file(
            category="actors",
            filename=filename,
            metadata=actor_metadata,
            content=actor_content
        )

        initiative_bonus = (data.dexterity - 10) // 2
        
        conn = get_db_connection(self.campaign_slug)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO characters (name, type, strength, dexterity, constitution, intelligence, wisdom, charisma, hp, max_hp, ac, initiative_bonus)
            VALUES (?, 'player', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.name, 
                data.strength, data.dexterity, data.constitution, 
                data.intelligence, data.wisdom, data.charisma,
                data.hp, data.hp, data.ac, initiative_bonus
            ))
            conn.commit()
            
            print("\n=======================================================")
            print("          SUCCESS: CHARACTER REGISTERED")
            print("=======================================================")
            print(f"Database Updated: data/campaigns/{self.campaign_slug}/game.db")
            print(f"Profile Created:  data/actors/{filename}.md")
            print("=======================================================\n")
            
        except sqlite3.Error as e:
            print(f"An error occurred while writing to SQLite: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    creator = CharacterCreator(campaign_slug="your_generated_campaign_slug_here")
    creator.run_creation_loop()
