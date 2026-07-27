import json
import os
from pydantic import BaseModel, Field
from typing import List, Literal
import re

from md_manager import MarkdownManager
from llm_manager import LLMController, PromptBuilder

class LocationTemplate(BaseModel):
    id: str = Field(..., description="Lowercase slug identifier (e.g., 'crypt_of_shadows', 'ostia_village')")
    name: str = Field(..., description="The clear display name of the location.")
    danger_level: Literal["safe", "low", "medium", "high"] = Field(..., description="Expected physical threat level.")
    description: str = Field(..., description="An atmospheric narrative description of this room or area.")


class CampaignTemplate(BaseModel):
    campaign_name: str = Field(..., description="The official name of the campaign.")
    setting_description: str = Field(..., description="A detailed summary of the world, themes, and climate.")
    primary_threat: str = Field(..., description="The main faction, monster, or antagonistic force.")
    starting_quest_hook: str = Field(..., description="The initial call to adventure that prompts player action.")
    locations: List[LocationTemplate] = Field(
        ..., 
        description="A list of exactly 3 starting locations for the adventuring party."
    )

class CampaignCreator:
    
    def __init__(self):
        self.md_manager = MarkdownManager()
        self.llm = LLMController()
        self.chat_history = []
        
        self.brainstorm_persona = """
        You are the World-Building Dungeon Master. 
        Your goal is to help the player brainstorm a unique, exciting D&D campaign setting.
        Ask targeted questions about themes, magic level, factions, and starting environments.
        Be highly collaborative, keep your responses concise, and suggest creative ideas.
        
        When the player indicates they are happy with the concept (e.g. saying 'save', 'approve', or 'start'), 
        instruct them to type 'complete' to finalize the generation.
        """

    def run_brainstorm_loop(self) -> None:
        print("\n=======================================================")
        print("          AETHER: CAMPAIGN CREATION TERMINAL")
        print("=======================================================")
        print("Tell me about the D&D campaign you want to build. For example:")
        print("'A dark-fantasy Gothic horror world where vampires rule the cities,' or")
        print("'A high-seas pirate adventure where islands float in the sky.'")
        print("To save and build the files at any point, type 'complete'.\n")

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() == "complete":
                print("\nProcessing finalized campaign assets... Please wait.")
                self.generate_and_save_campaign()
                break

            self.chat_history.append({"role": "user", "content": user_input})
            
            messages = PromptBuilder.compile_messages(
                system_persona=self.brainstorm_persona,
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
            import ollama
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
