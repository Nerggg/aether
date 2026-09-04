import json
import os
import re
import sqlite3
from pydantic import BaseModel, Field
from typing import List, Optional

from md_manager import MarkdownManager
from llm_manager import (
    LLMController, 
    PromptBuilder, 
    CampaignBlueprint, 
    StorylineBlueprint, 
    WorldMapBlueprint
)
from db import get_db_connection
from vector_db_manager import VectorDBManager


class CampaignChecklistParser(BaseModel):
    """Extraction schema to capture player-facing traits during conversation."""
    setting_description: Optional[str] = Field(None, description="Detailed setting/environment concept if decided, otherwise null.")
    primary_threat: Optional[str] = Field(None, description="The main villain or threat if decided, otherwise null.")
    starting_quest_hook: Optional[str] = Field(None, description="The starting quest objective if decided, otherwise null.")


class DerivedMetadataPayload(BaseModel):
    """Schema for silent extraction of stylistic details and campaign name options."""
    theme_vibe: str = Field(..., description="A 2-3 word stylistic sub-genre tag (e.g. 'Gothic Dark Fantasy').")
    starting_patron: str = Field(..., description="A friendly base of operations, helper NPC, or faction.")
    suggested_names: List[str] = Field(..., description="Exactly 3 atmospheric campaign name suggestions.")


class CampaignCreator:
    
    def __init__(self):
        self.md_manager = MarkdownManager()
        self.vector_db = VectorDBManager()
        self.llm = LLMController()
        self.chat_history = []

        self.checklist = {
            "setting_description": None,
            "primary_threat": None,
            "starting_quest_hook": None
        }

    @staticmethod
    def slugify(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"[-\s]+", "_", text)

    def run_brainstorm_loop(self) -> None:
        print("\n=======================================================")
        print("          AETHER: CAMPAIGN CREATION TERMINAL")
        print("=======================================================")
        print("Tell me about the D&D campaign you want to build. For example:")
        print("'A dark-fantasy swamp where a swamp witch resides.'")
        print("Once the environment, threat, and quest hook are established,")
        print("the setting will automatically build. Or type 'complete' to save.\n")

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() == "complete":
                print("\nProcessing finalized campaign assets... Please wait.")
                self.generate_and_save_campaign()
                break

            self.chat_history.append({"role": "user", "content": user_input})

            if len(self.chat_history) >= 2:
                try:
                    parser_system = """
                    You are a strict data extraction system. Analyze the recent conversation history and extract the campaign details into the required JSON schema format.
                    Only extract values that have been explicitly agreed upon or chosen. 
                    Default to null for any fields not explicitly described. Do not invent details.
                    """
                    parser_messages = [
                        {"role": "system", "content": parser_system}
                    ] + self.chat_history[-4:]
                    
                    parsed_response = self.llm.generate_structured_action(
                        messages=parser_messages,
                        response_schema=CampaignChecklistParser,
                        model=self.llm.referee_model,
                        temperature=0.0
                    )
                    
                    if parsed_response:
                        if parsed_response.setting_description: 
                            self.checklist["setting_description"] = parsed_response.setting_description
                        if parsed_response.primary_threat: 
                            self.checklist["primary_threat"] = parsed_response.primary_threat
                        if parsed_response.starting_quest_hook: 
                            self.checklist["starting_quest_hook"] = parsed_response.starting_quest_hook
                    
                    print(f"[Checklist Status] Setting: {'OK' if self.checklist['setting_description'] else 'PENDING'} | Threat: {'OK' if self.checklist['primary_threat'] else 'PENDING'} | Hook: {'OK' if self.checklist['starting_quest_hook'] else 'PENDING'}")
                except Exception as e:
                    print(f"[Debug] Parsing warning: {e}")

            if all(self.checklist.values()):
                print(f"\n[System] All core elements established: {self.checklist}")
                self.generate_and_save_campaign()
                break

            active_persona = f"""
            You are the World-Building Dungeon Master. Help the player establish exactly 3 campaign parameters:
            1. setting_description (the world's environment and style)
            2. primary_threat (the main threat, faction, or antagonist)
            3. starting_quest_hook (the opening call to adventure)

            Current Checklist Status:
            - setting_description: {'COMPLETED' if self.checklist['setting_description'] else 'PENDING'}
            - primary_threat: {'COMPLETED' if self.checklist['primary_threat'] else 'PENDING'}
            - starting_quest_hook: {'COMPLETED' if self.checklist['starting_quest_hook'] else 'PENDING'}

            INSTRUCTIONS:
            - Focus on PENDING parameters. Ask about ONE parameter at a time.
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
            
            response = self.llm.generate_narrative(messages, model=self.llm.model_name, temperature=0.7)
            print(f"\nDM: {response}")
            self.chat_history.append({"role": "assistant", "content": response})

    def generate_and_save_campaign(self) -> None:
        print("\n[1/4] Deriving stylistic attributes and campaign name suggestions...")
        
        derivation_prompt = """
        Analyze the campaign brainstorming details and output the structured DerivedMetadataPayload.
        Suggest exactly 3 highly atmospheric campaign names that match the style of the setting.
        """
        derivation_msg = PromptBuilder.compile_messages(derivation_prompt, [], self.chat_history)
        
        try:
            import ollama
            response = ollama.chat(
                model=self.llm.referee_model,
                messages=derivation_msg,
                format=DerivedMetadataPayload.model_json_schema(),
                options={"temperature": 0.8}
            )
            raw_json = response["message"]["content"]
            derived_meta = DerivedMetadataPayload.model_validate_json(raw_json)
        except Exception as e:
            print(f"[Warning] Silent derivation failed: {e}. Falling back to standard values.")
            derived_meta = DerivedMetadataPayload(
                theme_vibe="Heroic Fantasy",
                starting_patron="A local tavern owner",
                suggested_names=["An Unknown Journey", "Echoes of Adventure", "Tales of the Unknown"]
            )

        print("\nSelect a Campaign Name from the suggested options:")
        for idx, name in enumerate(derived_meta.suggested_names):
            print(f"{idx + 1}. {name}")
        print("4. Input a custom name")
        
        try:
            choice = int(input("\nSelect (1-4): ").strip())
            if choice in [1, 2, 3]:
                chosen_name = derived_meta.suggested_names[choice - 1]
            else:
                chosen_name = input("Enter custom name: ").strip()
        except Exception:
            chosen_name = derived_meta.suggested_names[0]

        campaign_slug = self.slugify(chosen_name)
        print(f"\nSetting finalized campaign name to: '{chosen_name}' (Slug: {campaign_slug})")

        print("\n[2/4] Executing sequential storyline Chain-of-Thought generation...")
        
        beginning_sys = f"You are a master RPG writer. Based on the setting '{self.checklist['setting_description']}', the threat '{self.checklist['primary_threat']}', and starting patron '{derived_meta.starting_patron}', write a highly descriptive narrative outline for the Beginning of this adventure."
        beginning_text = self.llm.generate_narrative([{"role": "system", "content": beginning_sys}], model=self.llm.model_name, temperature=0.7)
        
        middle_sys = f"You are a master RPG writer. Given the setting, threat, and the Beginning story beat:\n{beginning_text}\n\nWrite a highly descriptive narrative outline for the Middle part of this campaign detailing 2-3 specific complications, milestones, and travel obstacles."
        middle_text = self.llm.generate_narrative([{"role": "system", "content": middle_sys}], model=self.llm.model_name, temperature=0.7)
        
        print("Compiling final narrative structure and secret victory metrics...")
        end_sys = f"""
        Analyze the Beginning and Middle outlines and output the complete StorylineBlueprint JSON schema.
        Define the final confrontation scenario under 'end', and establish a precise, hidden 'victory_condition' metric (e.g. 'Defeat the swamp witch').
        
        Beginning Outline:
        {beginning_text}
        
        Middle Outline:
        {middle_text}
        """
        try:
            response = ollama.chat(
                model=self.llm.referee_model,
                messages=[{"role": "system", "content": end_sys}],
                format=StorylineBlueprint.model_json_schema(),
                options={"temperature": 0.5}
            )
            storyline_data = StorylineBlueprint.model_validate_json(response["message"]["content"])
        except Exception as e:
            print(f"[Warning] Storyline compilation failed: {e}. Defaulting structure.")
            storyline_data = StorylineBlueprint(
                beginning=beginning_text,
                middle=middle_text,
                end="The players confront the threat in a dramatic final battle.",
                victory_condition=f"Defeat {self.checklist['primary_threat']}"
            )

        print("\n[3/4] Designing structural location connection graph...")
        layout_sys = f"""
        Based on the setting '{self.checklist['setting_description']}' and the sequential storyline beginning, middle, and end, map out a connected layout of location nodes.
        You must determine the optimal number of locations dynamically based on the story beats.
        Return the structured WorldMapBlueprint JSON.
        
        Story beats:
        - Beginning: {storyline_data.beginning[:300]}...
        - Middle: {storyline_data.middle[:300]}...
        - End: {storyline_data.end[:300]}...
        """
        try:
            response = ollama.chat(
                model=self.llm.referee_model,
                messages=[{"role": "system", "content": layout_sys}],
                format=WorldMapBlueprint.model_json_schema(),
                options={"temperature": 0.5}
            )
            world_map = WorldMapBlueprint.model_validate_json(response["message"]["content"])
        except Exception as e:
            print(f"[Error] Failed to map world layout: {e}")
            return

        print("\n[4/4] Writing asset structures to disk and relational SQL databases...")
        self._write_assets_to_db_and_disk(
            campaign_slug=campaign_slug,
            campaign_name=chosen_name,
            theme_vibe=derived_meta.theme_vibe,
            starting_patron=derived_meta.starting_patron,
            storyline=storyline_data,
            world_map=world_map
        )

    def _write_assets_to_db_and_disk(
        self, 
        campaign_slug: str, 
        campaign_name: str, 
        theme_vibe: str, 
        starting_patron: str, 
        storyline: StorylineBlueprint, 
        world_map: WorldMapBlueprint
    ) -> None:
        
        campaign_metadata = {
            "campaign_name": campaign_name,
            "campaign_slug": campaign_slug,
            "theme_vibe": theme_vibe,
            "starting_patron": starting_patron,
            "primary_threat": self.checklist["primary_threat"],
            "starting_quest_hook": self.checklist["starting_quest_hook"],
            "starting_location": world_map.locations[0].id if world_map.locations else "global",
            "type": "campaign_info"
        }
        
        campaign_content = (
            f"# Campaign Setting: {campaign_name}\n\n"
            f"## Environmental Overview\n{self.checklist['setting_description']}\n\n"
            f"## Style Vibe\n{theme_vibe}\n\n"
            f"## Starting Patron\n{starting_patron}\n\n"
            f"## Campaign Storyline Outlines\n\n"
            f"### Beginning Beat\n{storyline.beginning}\n\n"
            f"### Middle Beat\n{storyline.middle}\n\n"
            f"### Climax Beat\n{storyline.end}\n"
        )
        
        campaign_filename = f"{campaign_slug}_info"
        self.md_manager.write_file(
            category="campaigns",
            filename=campaign_filename,
            metadata=campaign_metadata,
            content=campaign_content
        )
        self.vector_db.upsert_markdown_file(category="campaigns", filename=campaign_filename)

        conn = get_db_connection(campaign_slug)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO campaign_metadata (campaign_name, setting_description, primary_threat, starting_quest_hook, theme_vibe, starting_patron, victory_condition)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                campaign_name, 
                self.checklist["setting_description"], 
                self.checklist["primary_threat"], 
                self.checklist["starting_quest_hook"],
                theme_vibe, 
                starting_patron, 
                storyline.victory_condition
            ))

            for loc in world_map.locations:
                cursor.execute("""
                INSERT OR REPLACE INTO locations (id, name, brief_concept, is_generated)
                VALUES (?, ?, ?, 0)
                """, (loc.id, loc.name, loc.brief_concept))

            for loc in world_map.locations:
                for target_id in loc.connections:
                    cursor.execute("""
                    INSERT OR IGNORE INTO location_connections (from_location_id, to_location_id)
                    VALUES (?, ?)
                    """, (loc.id, target_id))

            conn.commit()
            
            print("\n=======================================================")
            print("          SUCCESS: CAMPAIGN ASSETS GENERATED")
            print("=======================================================")
            print(f"Isolated SQLite Updated: data/campaigns/{campaign_slug}/game.db")
            print(f"Narrative Profile Saved: data/campaigns/{campaign_filename}.md")
            print(f"Locations Generated:     {len(world_map.locations)} structural blueprint nodes loaded.")
            print("=======================================================\n")
            
        except sqlite3.Error as e:
            print(f"[Error] SQLite insertion failed during campaign serialization: {e}")
        finally:
            conn.close()


if __name__ == "__main__":
    creator = CampaignCreator()
    creator.run_brainstorm_loop()
