from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal
import inspect

class EnvironmentStateUpdate(BaseModel):
    ambient_changes: str = Field(
        ..., 
        description="Changes to lighting, weather, sound, or smell (e.g., 'The torches flicker and die, plunging the room into darkness.')"
    )
    object_states: str = Field(
        ..., 
        description="Changes to physical doors, chest locks, active fires, or traps (e.g., 'The wooden chest is now broken open.')"
    )
    danger_level: Literal["safe", "low", "medium", "high"] = Field(
        ..., 
        description="The modified level of physical risk in the environment."
    )


class CombatAction(BaseModel):
    target_id: str = Field(
        ..., 
        description="The ID of the opponent being targeted (e.g. 'player_warrior'). Use 'none' if fleeing/using an item on self."
    )
    action_type: Literal["melee_attack", "ranged_attack", "cast_spell", "flee", "use_item"] = Field(
        ..., 
        description="The general category of the combat action."
    )
    action_detail: str = Field(
        ..., 
        description="A short phrase detailing the action (e.g., 'Slashes with Rust-bitten Scimitar', 'Drinks Potion of Speed')."
    )


class DMAgent:
    
    def compile_prompt(self, location_lore: str, system_update: str = "") -> str:
        prompt = inspect.cleandoc("""
        You are the Dungeon Master (DM) for a local, text-based D&D game.
        Your duties are strictly:
        1. Describe the scenery, atmosphere, and environmental results of players' actions.
        2. Set the pacing of the narrative.
        3. Act as the third-person narrator for physical interactions.
        4. Integrate provided system/database updates seamlessly into your descriptions.

        CRITICAL CONSTRAINTS:
        - NEVER speak for or directly dialogue as NPCs or enemies. Let them speak for themselves in their turns.
        - NEVER make decisions for the player's character.
        - Describe events chronologically and naturally.
        """)
        
        if location_lore:
            prompt += f"\n\nActive Location Background:\n{location_lore}"
            
        if system_update:
            prompt += f"\n\nCRITICAL STATE UPDATE TO NARRATE:\n{system_update}\nYou must describe the physical outcome of this update in your text."
            
        return prompt.strip()


class EnvironmentAgent:
    
    def compile_prompt(self, player_action: str, current_states: str) -> str:
        return inspect.cleandoc(f"""
        You are the Physical Environment Engine for this game. 
        Your job is to analyze the player's action and update the state of the surrounding space, objects, and hazards.
        
        Current World State properties:
        {current_states}
        
        Player's proposed action:
        "{player_action}"
        
        Output strictly in the EnvironmentStateUpdate JSON format. Do not write dialogue.
        """)


class ActorAgent:
    
    def compile_narrative_prompt(self, actor_stats: Dict[str, Any], actor_lore: str) -> str:
        
        name = actor_stats.get("name", "An unknown figure")
        role = actor_stats.get("type", "NPC")
        hp = actor_stats.get("hp", 10)
        max_hp = actor_stats.get("max_hp", 10)
        
        prompt = inspect.cleandoc(f"""
        You are the Actor for the character '{name}'. 
        Role/Type: {role}
        Current Vitality: HP is {hp}/{max_hp}
        
        Your D&D Stats:
        STR: {actor_stats.get('strength', 10)} | DEX: {actor_stats.get('dexterity', 10)} | CON: {actor_stats.get('constitution', 10)}
        INT: {actor_stats.get('intelligence', 10)} | WIS: {actor_stats.get('wisdom', 10)} | CHA: {actor_stats.get('charisma', 10)}
        
        Background and Personality:
        {actor_lore}
        
        INSTRUCTIONS:
        - Speak and act in the first person as '{name}'.
        - Adhere strictly to your background, personality, and alignment.
        - If your HP is very low, let your dialogue and actions reflect your injury or fear.
        - Keep your responses relatively concise (under 100 words).
        """)
        return prompt.strip()

    def compile_combat_prompt(self, actor_stats: Dict[str, Any], opponent_descriptions: str, history: List[Dict[str, str]]) -> str:
        
        name = actor_stats.get("name", "The Combatant")
        hp = actor_stats.get("hp", 10)
        max_hp = actor_stats.get("max_hp", 10)
        
        prompt = f"""
        You are the Combat tactical brain for the actor '{name}'.
        Current Vitality: HP is {hp}/{max_hp}
        
        Your D&D Stats:
        STR: {actor_stats.get('strength', 10)} | DEX: {actor_stats.get('dexterity', 10)} | CON: {actor_stats.get('constitution', 10)}
        INT: {actor_stats.get('intelligence', 10)} | WIS: {actor_stats.get('wisdom', 10)} | CHA: {actor_stats.get('charisma', 10)}
        
        Active Combat Opponents in view:
        {opponent_descriptions}
        
        INSTRUCTIONS:
        - Analyze your stats, HP, and opponents to decide on your next action.
        - If you are a low-HP or cowardly enemy (like a goblin), prioritize fleeing or ranged attacks.
        - If you have high strength, prefer melee.
        - Output strictly in the CombatAction JSON format.
        """
        return prompt.strip()

if __name__ == "__main__":
    dm_agent = DMAgent()
    env_agent = EnvironmentAgent()
    actor_agent = ActorAgent()

    print("\n--- Test 1: Compile DM Agent Prompt with Spiked Trap Update ---")
    mock_location = "A damp, moss-covered corridor in the Ostian Undercroft."
    mock_hp_update = "SYSTEM UPDATE: player_warrior took 10 damage. HP is now 0/20. Fallen unconscious!"
    
    compiled_dm_prompt = dm_agent.compile_prompt(
        location_lore=mock_location, 
        system_update=mock_hp_update
    )
    print(compiled_dm_prompt)

    print("\n--- Test 2: Compile Environment Agent Prompt for Trap Sprung ---")
    mock_action = "I pull out a crowbar and jam it into the floor pressure plate."
    mock_ambient = "Old stone floor, torches on walls, pressure plate is active."
    
    compiled_env_prompt = env_agent.compile_prompt(
        player_action=mock_action,
        current_states=mock_ambient
    )
    print(compiled_env_prompt)

    print("\n--- Test 3: Compile Actor Agent Prompt for Injured Goblin ---")
    mock_goblin_stats = {
        "name": "Gruk the Rusty",
        "type": "enemy",
        "hp": 2,
        "max_hp": 8,
        "strength": 8,
        "dexterity": 14,
        "constitution": 10,
        "intelligence": 7,
        "wisdom": 8,
        "charisma": 5
    }
    mock_goblin_lore = "Gruk is a cowardly, rust-bitten goblin who is quick to beg for mercy if outmatched."
    
    compiled_actor_prompt = actor_agent.compile_narrative_prompt(
        actor_stats=mock_goblin_stats,
        actor_lore=mock_goblin_lore
    )
    print(compiled_actor_prompt)
