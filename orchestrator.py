import sqlite3
import os
import random
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import ollama

from md_manager import MarkdownManager
from vector_db_manager import VectorDBManager
from llm_manager import LLMController, PromptBuilder, GameActionPayload
from db import get_db_connection
from agents import DMAgent, ActorAgent


class TokenManager:
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, int(len(text) / 4))

    @classmethod
    def prune_history(cls, chat_history: List[Dict[str, str]], max_tokens: int = 4000) -> List[Dict[str, str]]:
        total_tokens = sum(cls.estimate_tokens(turn["content"]) for turn in chat_history)
        while total_tokens > max_tokens and len(chat_history) > 2:
            chat_history.pop(0)
            chat_history.pop(0)
            print("[Engine] Pruned oldest conversational turn to save context VRAM.")
            total_tokens = sum(cls.estimate_tokens(turn["content"]) for turn in chat_history)
        return chat_history

class GameOrchestrator:
    def __init__(self, campaign_slug: str, state: str = "NARRATIVE_PLAY"):
        self.campaign_slug = campaign_slug
        self.state = state

        self.md_manager = MarkdownManager()
        self.vector_db = VectorDBManager()
        self.llm = LLMController()
        
        self.dm_agent = DMAgent()
        self.actor_agent = ActorAgent()

        try:
            meta, _ = self.md_manager.read_file("campaigns", f"{self.campaign_slug}_info")
            self.current_location_slug = meta.get("starting_location", f"{self.campaign_slug}_start")
            print(f"[Orchestrator] Dynamic Start Room resolved: '{self.current_location_slug}'")
        except Exception:
            self.current_location_slug = f"{self.campaign_slug}_start"
        
        self.chat_history: List[Dict[str, str]] = []
        
        self.combat_queue: List[Dict[str, Any]] = []
        self.current_turn_index = 0

    def _execute_query(self, query: str, params: Tuple = ()) -> List[Tuple]:
        conn = get_db_connection(self.campaign_slug)
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            conn.commit()
            return cursor.fetchall()
        finally:
            conn.close()

    @staticmethod
    def dnd_roll(die_size: int, modifier: int = 0) -> int:
        return random.randint(1, die_size) + modifier

    def process_narrative_turn(self, user_input: str):
        self.chat_history.append({"role": "user", "content": user_input})
        
        self._ensure_location_generated(self.current_location_slug)
        
        movement_status = self._detect_and_handle_movement(user_input)
        
        mechanical_status = ""
        if movement_status:
            mechanical_status = movement_status
        else:
            ref_prompt = """
            You are the silent system referee. Analyze the player's last statement and decide if any numerical, 
            stat, or item inventory updates are required.
            Choose from these exact actions: 'damage', 'heal', 'add_item', 'remove_item', 'skill_check', 'none'.
            Output strictly in the required JSON schema format.
            """
            ref_context = [f"Active character ID: 'player_warrior'", f"Action: '{user_input}'"]
            ref_msg = PromptBuilder.compile_messages(ref_prompt, ref_context, [{"role": "user", "content": user_input}])
            
            action_payload = self.llm.generate_structured_action(
                messages=ref_msg,
                response_schema=GameActionPayload,
                model=self.llm.referee_model,
                temperature=0.0
            )
            
            if action_payload and action_payload.action_type != "none":
                mechanical_status = self.reconciler.reconcile(action_payload)
                print(f"[Engine Output] {mechanical_status}")

        vector = self.vector_db.embedder.get_embeddings([f"{self.current_location_slug} {user_input}"])[0]
        location_results = self.vector_db.search(query_vector=vector, category_filter="locations", limit=1)
        lore_results = self.vector_db.search(query_vector=vector, category_filter="lore", limit=1)

        rag_chunks = [
            r["document"] for r in location_results 
            if r["metadata"].get("meta_campaign_slug") == self.campaign_slug
        ]
        
        if not movement_status and action_payload and action_payload.action_type != "none":
            rules_query = f"{action_payload.action_type} {action_payload.value}"
            rules_results = self.vector_db.search(query=rules_query, category_filter="rules", limit=1)
            if rules_results:
                rag_chunks.append(f"DND RULE REFERENCE:\n{rules_results[0]['document']}")
                print(f"[RAG] Injected D&D Rule Context: {rules_results[0]['metadata']['source_file']}")

        if lore_results:
            rag_chunks.append(f"PROSE INSPIRATION:\n{lore_results[0]['document']}")
        
        dm_system = self.dm_agent.compile_prompt(
            location_lore="\n\n".join(rag_chunks),
            system_update=mechanical_status
        )
        dm_messages = PromptBuilder.compile_messages(dm_system, [], self.chat_history)
        
        full_response_chunks = []
        for chunk in self.llm.generate_narrative_stream(dm_messages):
            full_response_chunks.append(chunk)
            yield chunk
            
        narrative_response = "".join(full_response_chunks)
        self.chat_history.append({"role": "assistant", "content": narrative_response})
        self.chat_history = TokenManager.prune_history(self.chat_history)

    def set_state(self, new_state: str) -> None:
        self.state = new_state
        print(f"[Orchestrator State] Transitioned to -> {self.state}")

    def _get_connected_locations(self) -> List[Dict[str, str]]:
        """Queries SQLite to fetch all connected nodes adjacent to the current location."""
        conn = get_db_connection(self.campaign_slug)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, name, brief_concept 
            FROM locations 
            WHERE id IN (
                SELECT to_location_id FROM location_connections WHERE from_location_id = ?
                UNION
                SELECT from_location_id FROM location_connections WHERE to_location_id = ?
            )
        """, (self.current_location_slug, self.current_location_slug))
        rows = cursor.fetchall()
        conn.close()
        return [{"id": r[0], "name": r[1], "concept": r[2]} for r in rows]

    def _ensure_location_generated(self, location_id: str) -> None:
        """Verifies if a location's description exists. If is_generated is 0, executes JIT creation."""
        conn = get_db_connection(self.campaign_slug)
        cursor = conn.cursor()
        cursor.execute("SELECT name, brief_concept, is_generated FROM locations WHERE id = ?", (location_id,))
        res = cursor.fetchone()
        
        if not res:
            conn.close()
            return
            
        name, brief_concept, is_generated = res
        
        if is_generated == 0:
            print(f"\n[JIT Generator] Executing Just-In-Time generation for: '{name}' ({location_id})...")
            
            cursor.execute("SELECT theme_vibe FROM campaign_metadata LIMIT 1")
            vibe_res = cursor.fetchone()
            vibe = vibe_res[0] if vibe_res else "Heroic Fantasy"
            conn.close()
            
            jit_prompt = f"""
            You are a master of sensory fantasy descriptions.
            Generate a detailed, immersive, sensory-rich (sounds, smells, weather, lighting) description for a location called '{name}'.
            
            Core Concept: {brief_concept}
            Stylistic Theme: {vibe}
            
            INSTRUCTIONS:
            - Write in active, chronological, second-person style ('You see...', 'The air smells of...').
            - Keep the description between 100-150 words.
            - Focus only on describing physical atmosphere. Do not write dialogue or start encounters.
            """
            
            full_description = self.llm.generate_narrative(
                messages=[{"role": "system", "content": jit_prompt}],
                model=self.llm.model_name,
                temperature=0.7
            )
            
            loc_metadata = {
                "id": location_id,
                "campaign_slug": self.campaign_slug,
                "name": name,
                "type": "location_description"
            }
            loc_content = f"# {name}\n\n{full_description.strip()}"
            
            self.md_manager.write_file(
                category="locations",
                filename=location_id,
                metadata=loc_metadata,
                content=loc_content
            )
            
            self.vector_db.upsert_markdown_file(category="locations", filename=location_id)
            
            conn = get_db_connection(self.campaign_slug)
            cursor = conn.cursor()
            cursor.execute("UPDATE locations SET is_generated = 1 WHERE id = ?", (location_id,))
            conn.commit()
            print(f"[JIT Generator] Asset written and indexed: '{name}'\n")
            
        conn.close()

    def _detect_and_handle_movement(self, user_input: str) -> Optional[str]:
        """Intercepts movement intent. If connected, transitions the node; if unmapped, triggers redirection."""
        user_input_lower = user_input.lower()
        movement_triggers = ["go to", "walk to", "travel to", "head to", "move to", "enter", "leave", "exit"]
        
        is_moving = any(trigger in user_input_lower for trigger in movement_triggers)
        if not is_moving:
            directions = ["north", "south", "east", "west"]
            is_moving = any(f"go {d}" in user_input_lower or f"walk {d}" in user_input_lower for d in directions)
            
        if is_moving:
            connections = self._get_connected_locations()
            for conn in connections:
                clean_id = conn["id"].replace(f"{self.campaign_slug}_", "").replace("_", " ")
                if conn["id"].lower() in user_input_lower or conn["name"].lower() in user_input_lower or clean_id in user_input_lower:
                    old_loc = self.current_location_slug
                    self.current_location_slug = conn["id"]
                    
                    self._ensure_location_generated(conn["id"])
                    
                    return f"SYSTEM UPDATE: Player successfully moved from '{old_loc}' to '{conn['id']}' ({conn['name']}). You must narrate their departure, short journey, and physical arrival."
            
            return "SYSTEM UPDATE: Player attempted to travel to an unmapped area or leave the boundaries of the current location. You must narrate a natural physical or environmental obstacle blocking their path and redirect them back."
            
        return None

if __name__ == "__main__":
    campaign = "brindlemark_dragon_spine"
    
    # Simple check to verify initialization
    engine = GameOrchestrator(campaign_slug=campaign)
    print(f"Orchestrator successfully initialized for campaign: {engine.campaign_slug}")
