import os
import re
import ollama
from pydantic import BaseModel, Field
from typing import Literal, Optional
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

class CharacterChecklistParser(BaseModel):
    """Extraction schema to programmatically capture traits during the conversation."""
    name: Optional[str] = Field(None, description="The chosen name if mentioned, otherwise null.")
    race: Optional[ALLOWED_RACES] = Field(None, description="The chosen race from the allowed list if mentioned, otherwise null.")
    character_class: Optional[ALLOWED_CLASSES] = Field(None, description="The chosen class from the allowed list if mentioned, otherwise null.")


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

        # The state checklist
        self.checklist = {
            "name": None,
            "race": None,
            "character_class": None
        }

    @staticmethod
    def slugify(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"[-\s]+", "_", text)

    def run_creation_loop(self) -> None:
        """Runs the interactive console creation loop."""
        clear_screen = lambda: os.system("cls" if os.name == "nt" else "clear")
        clear_screen()
        
        # Front-load valid rules at start
        print(DND_RULES_TEXT)
        
        print("\n=======================================================")
        print("          AETHER: CHARACTER CREATION TERMINAL")
        print("=======================================================")
        print(f"Active Campaign: {self.campaign_slug}")
        print("Introduce your character's name, race, or class to begin!")
        print("Once all traits are decided, the sheet will automatically write.")
        print("Or you could type 'complete' to save and build the files at any point.\n")

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() == "complete":
                print("\nProcessing finalized character... Please wait.")
                self.generate_and_save_character()
                break

            # Append user input to history so the parser has context
            self.chat_history.append({"role": "user", "content": user_input})

            # 1. CONTEXT-AWARE DETECTOR: Send recent chat history for extraction [2]
            try:
                parser_system = "You are a data extraction system. Analyze the recent conversation history and extract the D&D character options into the required JSON schema format."
                # We compile the system prompt + the last 4 turns of context
                parser_messages = [{"role": "system", "content": parser_system}] + self.chat_history[-4:]
                
                detect_response = ollama.chat(
                    model=self.llm.model_name,
                    messages=parser_messages,
                    format=CharacterChecklistParser.model_json_schema(),
                    options={"temperature": 0.0}
                )
                parsed_traits = CharacterChecklistParser.model_validate_json(detect_response["message"]["content"])
                
                # Update checklist
                if parsed_traits.name: self.checklist["name"] = parsed_traits.name
                if parsed_traits.race: self.checklist["race"] = parsed_traits.race
                if parsed_traits.character_class: self.checklist["character_class"] = parsed_traits.character_class
                
                print(f"[Checklist Status] Name: {self.checklist['name']} | Race: {self.checklist['race']} | Class: {self.checklist['character_class']}")
            except Exception as e:
                pass # Proceed silently with conversation on parsing errors

            # 2. AUTO-CLOSE CHECK: If all properties are successfully extracted, terminate loop [2]
            if all(self.checklist.values()):
                print(f"\n[System] All required attributes found: {self.checklist}")
                print("Compiling character stats... Please wait.")
                self.generate_and_save_character()
                break

            # 3. CONVERSATIONAL STEP (Dynamic State Reflection)
            active_persona = f"""
            You are the Character Creation Guide. Your job is to help the player build a D&D character.
            
            {DND_RULES_TEXT}
            
            Here is the current status of the character checklist:
            - name: {'COMPLETED ({})'.format(self.checklist['name']) if self.checklist['name'] else 'PENDING'}
            - race: {'COMPLETED ({})'.format(self.checklist['race']) if self.checklist['race'] else 'PENDING'}
            - character_class: {'COMPLETED ({})'.format(self.checklist['character_class']) if self.checklist['character_class'] else 'PENDING'}

            INSTRUCTIONS:
            - Focus on asking questions to resolve the PENDING traits. Ask about ONE trait at a time.
            - If 'name' is PENDING, ask them for their character's name, or offer 2-3 creative suggestions.
            - If 'race' is PENDING, guide them to select from the approved races list.
            - If 'character_class' is PENDING, guide them to select from the approved classes list.
            - CRITICAL BOUNDARY: You are STRICTLY FORBIDDEN from allowing any Race or Class outside of the approved lists. If the user insists on a custom option, you MUST politely reject their request and redirect them to the approved list.
            - Once Name, Race, and Class are all COMPLETED, use your final turn to describe their starting HP, AC, ability stats, and briefly ask about their background.
            """
            
            messages = PromptBuilder.compile_messages(
                system_persona=active_persona,
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
