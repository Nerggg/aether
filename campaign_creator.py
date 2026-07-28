import json
import os
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import re
import ollama

from md_manager import MarkdownManager
from llm_manager import LLMController, PromptBuilder

# =====================================================================
# 1. Campaign Generation Schemas (Pydantic)
# =====================================================================

class LocationTemplate(BaseModel):
    id: str = Field(..., description="Lowercase slug identifier (e.g., 'crypt_of_shadows', 'ostia_village')")
    name: str = Field(..., description="The clear display name of the location.")
    danger_level: Literal["safe", "low", "medium", "high"] = Field(..., description="Expected physical threat level.")
    description: str = Field(..., description="An atmospheric narrative description of this room or area.")


class CampaignChecklistParser(BaseModel):
    """Extraction schema to programmatically capture world traits during the conversation."""
    campaign_name: Optional[str] = Field(None, description="The chosen name of the campaign setting if decided, otherwise null.")
    setting_description: Optional[str] = Field(None, description="Brief description of the setting style (e.g. gothic, pirate, swamp) if decided, otherwise null.")
    primary_threat: Optional[str] = Field(None, description="The main villain or threat (e.g. a witch, a dragon) if decided, otherwise null.")
    starting_quest_hook: Optional[str] = Field(None, description="The starting quest objective if decided, otherwise null.")


class CampaignTemplate(BaseModel):
    campaign_name: str = Field(..., description="The official name of the campaign.")
    setting_description: str = Field(..., description="A detailed summary of the world, themes, and climate.")
    primary_threat: str = Field(..., description="The main faction, monster, or antagonistic force.")
    starting_quest_hook: str = Field(..., description="The initial call to adventure that prompts player action.")
    locations: List[LocationTemplate] = Field(
        ..., 
        description="A list of exactly 3 starting locations for the adventuring party."
    )


# =====================================================================
# 2. Campaign Creator Engine
# =====================================================================

class CampaignCreator:
    
    def __init__(self):
        self.md_manager = MarkdownManager()
        self.llm = LLMController()
        self.chat_history = []

        # The state checklist
        self.checklist = {
            "campaign_name": None,
            "setting_description": None,
            "primary_threat": None,
            "starting_quest_hook": None
        }

    def run_brainstorm_loop(self) -> None:
        print("\n=======================================================")
        print("          AETHER: CAMPAIGN CREATION TERMINAL")
        print("=======================================================")
        print("Tell me about the D&D campaign you want to build. For example:")
        print("'A dark-fantasy Gothic horror world where vampires rule the cities,' or")
        print("'A high-seas pirate adventure where islands float in the sky.'")
        print("Once all world details are decided, the setting will automatically write.")
        print("Or you could type 'complete' to save and build the files at any point.\n")

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() == "complete":
                print("\nProcessing finalized campaign assets... Please wait.")
                self.generate_and_save_campaign()
                break

            # Append the latest turn to history so our parser has context
            self.chat_history.append({"role": "user", "content": user_input})

            # 1. CONTEXT-AWARE DETECTOR: Send recent chat history for extraction [2]
            try:
                parser_system = "You are a data extraction system. Analyze the recent conversation history and extract the campaign details into the required JSON schema format."
                # We compile the system prompt + the last 4 turns of context
                parser_messages = [{"role": "system", "content": parser_system}] + self.chat_history[-4:]
                
                detect_response = ollama.chat(
                    model=self.llm.model_name,
                    messages=parser_messages,
                    format=CampaignChecklistParser.model_json_schema(),
                    options={"temperature": 0.0} # Pure determinism
                )
                parsed_traits = CampaignChecklistParser.model_validate_json(detect_response["message"]["content"])
                
                # Update our python checklist
                if parsed_traits.campaign_name: self.checklist["campaign_name"] = parsed_traits.campaign_name
                if parsed_traits.setting_description: self.checklist["setting_description"] = parsed_traits.setting_description
                if parsed_traits.primary_threat: self.checklist["primary_threat"] = parsed_traits.primary_threat
                if parsed_traits.starting_quest_hook: self.checklist["starting_quest_hook"] = parsed_traits.starting_quest_hook
                
                print(f"[Checklist Status] Name: {self.checklist['campaign_name']} | Setting: {self.checklist['setting_description']} | Threat: {self.checklist['primary_threat']} | Hook: {self.checklist['starting_quest_hook']}")
            except Exception as e:
                pass # Fail silently on extraction errors and proceed

            # 2. AUTO-CLOSE CHECK: If all properties are successfully extracted, terminate loop [2]
            if all(self.checklist.values()):
                print(f"\n[System] All required campaign elements found: {self.checklist}")
                print("Processing finalized campaign assets... Please wait.")
                self.generate_and_save_campaign()
                break

            # 3. CONVERSATIONAL STEP (Dynamic State Reflection)
            # We inject the active checklist status directly into the interviewer prompt
            active_persona = f"""
            You are the World-Building Dungeon Master. Your goal is to guide the player through establishing exactly 4 campaign parameters:
            1. campaign_name
            2. setting_description (including environment, atmosphere, and general setting)
            3. primary_threat (the main villain, antagonistic faction, or monster)
            4. starting_quest_hook (the opening call to adventure)

            Here is the current status of the campaign checklist:
            - campaign_name: {'COMPLETED ({})'.format(self.checklist['campaign_name']) if self.checklist['campaign_name'] else 'PENDING'}
            - setting_description: {'COMPLETED ({})'.format(self.checklist['setting_description']) if self.checklist['setting_description'] else 'PENDING'}
            - primary_threat: {'COMPLETED ({})'.format(self.checklist['primary_threat']) if self.checklist['primary_threat'] else 'PENDING'}
            - starting_quest_hook: {'COMPLETED ({})'.format(self.checklist['starting_quest_hook']) if self.checklist['starting_quest_hook'] else 'PENDING'}

            INSTRUCTIONS:
            - Your current priority is to ask questions to fill the PENDING parameters. Focus on ONE parameter at a time.
            - If 'campaign_name' is PENDING, ask them what they want to name the setting, or offer 2-3 creative suggestions.
            - If 'setting_description' is PENDING, ask them about the general setting, the environment style (e.g. swamp, village, gothic city), or magic level.
            - If 'primary_threat' is PENDING, ask about the main villain, faction, or monster that is causing issues.
            - If 'starting_quest_hook' is PENDING, ask them how the party begins their adventure.
            - Once a parameter is resolved, acknowledge it and smoothly ask for the next PENDING parameter.
            - Speak in a highly collaborative, concise, and creative world-building tone.
            """
            
            messages = PromptBuilder.compile_messages(
                system_persona=active_persona,
                rag_context_chunks=[],
                chat_history=self.chat_history
            )
            
            response = self.llm.generate_narrative(messages)
            print(f"\nDM: {response}")
            self.chat_history.append({"role": "assistant", "content": response})

    def generate_and_save_campaign(self) -> None:
        compiler_prompt = """
        You are the System compiler. Analyze the previous brainstorming history and synthesize 
        the complete campaign details into the required JSON schema structure. 
        Ensure you extract/generate exactly 3 highly detailed starting locations matching the setting.
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
                format=CampaignTemplate.model_json_schema(),
                options={"temperature": 0.0}
            )
            
            raw_json = response["message"]["content"]
            campaign_data = CampaignTemplate.model_validate_json(raw_json)
            
            self._write_assets_to_disk(campaign_data)
            
        except Exception as e:
            print(f"\nError compiling campaign: {e}")
            print("Please try again or refine your brainstorming details.")

    @staticmethod
    def slugify(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"[-\s]+", "_", text)

    def _write_assets_to_disk(self, data: CampaignTemplate) -> None:
        campaign_slug = self.slugify(data.campaign_name)
        
        campaign_metadata = {
            "campaign_name": data.campaign_name,
            "campaign_slug": campaign_slug,
            "primary_threat": data.primary_threat,
            "starting_quest_hook": data.starting_quest_hook,
            "type": "campaign_info"
        }
        campaign_content = (
            f"# Campaign: {data.campaign_name}\n\n"
            f"## Setting Overview\n{data.setting_description}\n\n"
            f"## Main Threat\n{data.primary_threat}\n\n"
            f"## Call to Adventure\n{data.starting_quest_hook}"
        )
        
        campaign_filename = f"{campaign_slug}_info"
        self.md_manager.write_file(
            category="campaigns",
            filename=campaign_filename,
            metadata=campaign_metadata,
            content=campaign_content
        )

        for loc in data.locations:
            loc_slug = self.slugify(loc.name)
            unique_location_id = f"{campaign_slug}_{loc_slug}"
            
            loc_metadata = {
                "id": unique_location_id,
                "campaign_slug": campaign_slug,
                "name": loc.name,
                "danger_level": loc.danger_level,
                "type": "location_description"
            }
            loc_content = f"# {loc.name}\n\n{loc.description}"
            
            self.md_manager.write_file(
                category="locations",
                filename=unique_location_id,
                metadata=loc_metadata,
                content=loc_content
            )

        print("\n=======================================================")
        print("          SUCCESS: CAMPAIGN ASSETS GENERATED")
        print("=======================================================")
        print(f"Main Profile Created: data/campaigns/{campaign_filename}.md")
        for loc in data.locations:
            loc_slug = self.slugify(loc.name)
            print(f"Location Created:      data/locations/{campaign_slug}_{loc_slug}.md")
        print("=======================================================\n")

if __name__ == "__main__":
    creator = CampaignCreator()
    creator.run_brainstorm_loop()
