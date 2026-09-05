import json
import ollama
import random
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional, Literal, Union

class NPCSpawnDetails(BaseModel):
    name: str = Field(..., description="The unique name of the character introduced.")
    type: Literal["ally", "enemy", "merchant"] = Field(..., description="The role or classification of the character.")
    is_persistent: bool = Field(..., description="Whether they are a major/persistent character (allies, bosses, persistent merchants) or generic/ephemeral (goblins, random tavern patrons, generic guards).")
    brief_backstory: Optional[str] = Field(None, description="Detailed backstory and lore if persistent. Keep null or very brief for generic characters.")
    template_id: Optional[str] = Field(None, description="Monster template identifier (e.g. 'goblin', 'giant_rat') if applicable, else null.")

class GameActionPayload(BaseModel):
    action_type: Literal["damage", "heal", "skill_check", "move", "spawn_npc", "none"] = Field(
        ..., 
        description="The mechanical action type to perform on the database."
    )
    target_id: str = Field(
        ..., 
        description="The ID of the target. Use 'none' if inapplicable."
    )
    value: Union[int, str] = Field(
        ..., 
        description="The numeric amount or string identifier (e.g. skill name, item name, or destination ID)."
    )
    explanation: str = Field(
        ..., 
        description="A very brief reasoning for why this action was taken."
    )
    spawned_npc: Optional[NPCSpawnDetails] = Field(
        None,
        description="Populated ONLY when action_type is 'spawn_npc'."
    )

class CampaignBlueprint(BaseModel):
    campaign_name: str = Field(
        ..., 
        description="The finalized official name of the setting."
    )
    setting_description: str = Field(
        ..., 
        description="Atmospheric summary of the world environment, themes, and climate."
    )
    primary_threat: str = Field(
        ..., 
        description="The primary antagonistic force, faction, or threat."
    )
    starting_quest_hook: str = Field(
        ..., 
        description="The initial opening call to adventure presented to the player."
    )
    theme_vibe: str = Field(
        ..., 
        description="The derived stylistic sub-genre or tone tag (e.g., 'Gothic Dark Fantasy', 'High-Seas Exploration')."
    )
    starting_patron: str = Field(
        ..., 
        description="A friendly helper, local faction, or base of operations that anchors the beginning of the story."
    )


class StorylineBlueprint(BaseModel):
    beginning: str = Field(
        ..., 
        description="The opening scene, initial hook details, and local context."
    )
    middle: str = Field(
        ..., 
        description="A list of 2-3 major complications, milestones, or obstacles that occur during the journey."
    )
    end: str = Field(
        ..., 
        description="The final scenario outline, the climax, or confrontation setup."
    )
    victory_condition: str = Field(
        ..., 
        description="The hidden, precise mechanical criteria required to complete the campaign successfully (e.g., 'Slay the Swamp Witch')."
    )


class LocationBlueprint(BaseModel):
    id: str = Field(
        ..., 
        description="Unique, lowercase snake_case identifier slug (e.g., 'village_tavern', 'misty_bog_path')."
    )
    name: str = Field(
        ..., 
        description="The clear, human-readable display name of the location."
    )
    brief_concept: str = Field(
        ..., 
        description="A concise 1-2 sentence core concept defining the location's purpose (used later for JIT generation)."
    )
    connections: List[str] = Field(
        ..., 
        description="A list of location IDs that directly connect and are adjacent to this node."
    )


class WorldMapBlueprint(BaseModel):
    locations: List[LocationBlueprint] = Field(
        ..., 
        description="A dynamically sized array of location nodes representing the connected campaign map."
    )

class PromptBuilder:
    
    @staticmethod
    def compile_messages(
        system_persona: str, 
        rag_context_chunks: List[str], 
        chat_history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        formatted_context = ""
        if rag_context_chunks:
            formatted_context = "\n--- RELEVANT CAMPAIGN CONTEXT ---\n"
            formatted_context += "\n\n".join(rag_context_chunks)
            formatted_context += "\n---------------------------------\n"

        full_system_content = f"{system_persona.strip()}\n{formatted_context}\n"
        
        messages = [{"role": "system", "content": full_system_content}]
        
        for turn in chat_history:
            messages.append({
                "role": turn["role"], 
                "content": turn["content"]
            })
            
        return messages


class LLMController:
    
    def __init__(self, model_name: str = "llama3.2", referee_model: str = "qwen2.5-coder:3b"):
        self.model_name = model_name
        self.referee_model = referee_model
        print(f"LLM Controller initialized. DM: {self.model_name} | Referee: {self.referee_model}")

    def generate_narrative(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.7, seed: Optional[int] = None) -> str:
        target_model = model if model else self.model_name
        options = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
            
        try:
            response = ollama.chat(
                model=target_model,
                messages=messages,
                options=options
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"Error during narrative generation: {e}")
            return "The shadows whisper, but the words are lost..."

    def generate_narrative_stream(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.7, seed: Optional[int] = None):
        target_model = model if model else self.model_name
        options = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
            
        try:
            response = ollama.chat(
                model=target_model,
                messages=messages,
                options=options,
                stream=True
            )
            for chunk in response:
                yield chunk["message"]["content"]
        except Exception as e:
            print(f"Error during streaming narrative generation: {e}")
            yield "The shadows whisper, but the words are lost..."

    def generate_structured_action(
        self, 
        messages: List[Dict[str, str]], 
        response_schema = GameActionPayload, 
        model: Optional[str] = None, 
        temperature: float = 0.0, 
        seed: Optional[int] = None
    ) -> Any:

        target_model = model if model else self.referee_model
        options = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
            
        try:
            response = ollama.chat(
                model=target_model,
                messages=messages,
                format=response_schema.model_json_schema(),
                options=options
            )
            
            raw_content = response["message"]["content"]
            validated_payload = response_schema.model_validate_json(raw_content)
            return validated_payload
            
        except ValidationError as val_err:
            print(f"Pydantic Validation failed for schema output: {val_err}")
            return None
        except Exception as e:
            print(f"Error during structured action generation: {e}")
            return None

if __name__ == "__main__":
    llm = LLMController()

    print("\n--- Phase 1: Test Narrative Generation ---")
    system_prompt = "You are Barnaby, a short gnome merchant who runs an old trade cart."
    rag_context = [
        "Barnaby currently owns: 3 iron rations, 1 steel sword, 1 copper kettle.",
        "The town is currently experiencing a severe bread shortage."
    ]
    chat_history = [
        {"role": "user", "content": "What do you have for sale, Barnaby?"}
    ]

    compiled_prompt = PromptBuilder.compile_messages(system_prompt, rag_context, chat_history)
    
    npc_dialogue = llm.generate_narrative(compiled_prompt)
    print(f"Barnaby: {npc_dialogue}")


    print("\n--- Phase 2: Test Structured State Generation ---")
    referee_prompt = """
    You are the system referee. Analyze the player's action and decide if any numerical updates are required.
    Use the provided action schema to structure your answer.
    """
    referee_context = [
        "The player's active ID is 'player_warrior'.",
        "The warrior just drank a basic healing potion worth 2d4+2 health."
    ]
    referee_history = [
        {"role": "user", "content": "The warrior drinks the potion. I roll a 6 for total healing."}
    ]

    compiled_referee = PromptBuilder.compile_messages(referee_prompt, referee_context, referee_history)

    action_payload = llm.generate_structured_action(compiled_referee)

    if action_payload:
        print("\n--- Successfully Generated Structured Action ---")
        print(f"Action Type: {action_payload.action_type}")
        print(f"Target ID:   {action_payload.target_id}")
        print(f"Value:       {action_payload.value}")
        print(f"Explanation: {action_payload.explanation}")
    else:
        print("Failed to generate a valid, structured action schema.")
