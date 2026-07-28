import json
import ollama
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional, Literal, Union

class GameActionPayload(BaseModel):
    action_type: Literal["damage", "heal", "add_item", "remove_item", "skill_check", "none"] = Field(
        ..., 
        description="The mechanical action type to perform on the database."
    )
    target_id: str = Field(
        ..., 
        description="The ID of the target (e.g., 'player', 'actor_barnaby', 'chest_01'). Use 'none' if inapplicable."
    )
    value: Union[int, str] = Field(
        ..., 
        description="The numeric amount (e.g. damage/gold) or string identifier (e.g. item name) related to the action."
    )
    explanation: str = Field(
        ..., 
        description="A very brief reasoning for why this action was taken."
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
    
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        print(f"LLM Controller initialized using model: {self.model_name}")

    def generate_narrative(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.7}
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"Error during narrative generation: {e}")
            return "The shadows whisper, but the words are lost..."

    def generate_structured_action(self, messages: List[Dict[str, str]]) -> Optional[GameActionPayload]:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                format=GameActionPayload.model_json_schema(),
                options={"temperature": 0.0}
            )
            
            raw_content = response["message"]["content"]
            
            validated_payload = GameActionPayload.model_validate_json(raw_content)
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
