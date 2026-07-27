import sqlite3
import os
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel

from md_manager import MarkdownManager
from vector_db_manager import VectorDBManager
from llm_manager import LLMController, PromptBuilder, GameActionPayload

class TokenManager:
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, int(len(text) / 4))

    @classmethod
    def prune_history(cls, chat_history: List[Dict[str, str]], max_tokens: int = 4000) -> List[Dict[str, str]]:
        total_tokens = sum(cls.estimate_tokens(turn["content"]) for turn in chat_history)
        
        while total_tokens > max_tokens and len(chat_history) > 2:
            removed_1 = chat_history.pop(0)
            removed_2 = chat_history.pop(0)
            total_tokens -= (cls.estimate_tokens(removed_1["content"]) + cls.estimate_tokens(removed_2["content"]))
            print(f"Context ceiling reached. Pruned older conversational turns to save tokens.")
            
        return chat_history


class StateReconciler:
    
    def __init__(self, db_path: str = "aether_game.db"):
        self.db_path = db_path

    def _execute_query(self, query: str, params: Tuple = ()) -> List[Tuple]:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            conn.commit()
            return cursor.fetchall()
        finally:
            conn.close()

    def reconcile(self, payload: GameActionPayload) -> str:
        action = payload.action_type
        target = payload.target_id
        val = payload.value

        if action == "none":
            return ""

        print(f"Reconciling state change: Action={action}, Target={target}, Value={val}")

        if action == "damage":
            try:
                damage_amount = int(val)
                query = "SELECT hp, max_hp FROM characters WHERE id = ? OR name LIKE ?"
                res = self._execute_query(query, (target, f"%{target}%"))
                if not res:
                    return f"SYSTEM UPDATE ERROR: Target '{target}' not found in database."
                
                curr_hp, max_hp = res[0][0], res[0][1]
                new_hp = max(0, curr_hp - damage_amount)
                
                update_query = "UPDATE characters SET hp = ? WHERE id = ? OR name LIKE ?"
                self._execute_query(update_query, (new_hp, target, f"%{target}%"))
                
                status_msg = f"SYSTEM UPDATE: {target} took {damage_amount} damage. HP is now {new_hp}/{max_hp}."
                if new_hp == 0:
                    status_msg += f" {target} has fallen unconscious or died!"
                return status_msg
            except ValueError:
                return f"SYSTEM UPDATE ERROR: Invalid numeric value '{val}' for damage."

        elif action == "heal":
            try:
                heal_amount = int(val)
                query = "SELECT hp, max_hp FROM characters WHERE id = ? OR name LIKE ?"
                res = self._execute_query(query, (target, f"%{target}%"))
                if not res:
                    return f"SYSTEM UPDATE ERROR: Target '{target}' not found."
                
                curr_hp, max_hp = res[0][0], res[0][1]
                new_hp = min(max_hp, curr_hp + heal_amount)
                
                update_query = "UPDATE characters SET hp = ? WHERE id = ? OR name LIKE ?"
                self._execute_query(update_query, (new_hp, target, f"%{target}%"))
                
                return f"SYSTEM UPDATE: {target} healed by {heal_amount}. HP is now {new_hp}/{max_hp}."
            except ValueError:
                return f"SYSTEM UPDATE ERROR: Invalid numeric value '{val}' for healing."

        elif action == "add_item":
            item_res = self._execute_query("SELECT id FROM items WHERE name LIKE ?", (f"%{val}%",))
            if not item_res:
                self._execute_query("INSERT INTO items (name, value, weight, description) VALUES (?, 0, 1.0, 'A basic item.')", (str(val),))
                item_res = self._execute_query("SELECT id FROM items WHERE name = ?", (str(val),))
                
            item_id = item_res[0][0]
            
            owner_res = self._execute_query("SELECT id, type FROM characters WHERE id = ? OR name LIKE ?", (target, f"%{target}%"))
            if owner_res:
                owner_id, owner_type = owner_res[0][0], owner_res[0][1]
            else:
                owner_id, owner_type = 1, "container"
                
            inv_res = self._execute_query(
                "SELECT id, quantity FROM inventories WHERE owner_type = ? AND owner_id = ? AND item_id = ?",
                (owner_type, owner_id, item_id)
            )
            
            if inv_res:
                self._execute_query(
                    "UPDATE inventories SET quantity = quantity + 1 WHERE id = ?",
                    (inv_res[0][0],)
                )
            else:
                self._execute_query(
                    "INSERT INTO inventories (owner_type, owner_id, item_id, quantity) VALUES (?, ?, ?, 1)",
                    (owner_type, owner_id, item_id)
                )
                
            return f"SYSTEM UPDATE: Added 1x '{val}' to {target}'s inventory."

        return f"SYSTEM UPDATE: Logged event '{action}' on '{target}'."


class GameOrchestrator:
    
    def __init__(self, state: str = "NARRATIVE_PLAY"):
        self.state = state
        
        self.md_manager = MarkdownManager()
        self.vector_db = VectorDBManager()
        self.llm = LLMController()
        self.reconciler = StateReconciler()
        
        self.chat_history: List[Dict[str, str]] = []
        
    def set_state(self, new_state: str) -> None:
        self.state = new_state
        print(f"\n--- Orchestrator State Switched to: {self.state} ---")

    def process_turn(self, user_input: str) -> str:
        if self.state != "NARRATIVE_PLAY":
            return f"System is currently in '{self.state}' mode. Turn loop restricted."

        self.chat_history.append({"role": "user", "content": user_input})
        
        referee_system_prompt = """
        You are the silent system referee. Analyze the player's last statement and decide if any numerical, 
        stat, or item inventory updates are required.
        Choose from these exact actions: 'damage', 'heal', 'add_item', 'remove_item', 'skill_check', 'none'.
        Output strictly in the required JSON schema format.
        """
        referee_context = [
            "Active player character: 'player_warrior'.",
            f"Active player input: '{user_input}'"
        ]
        referee_messages = PromptBuilder.compile_messages(
            system_persona=referee_system_prompt,
            rag_context_chunks=referee_context,
            chat_history=[{"role": "user", "content": user_input}]
        )
        
        action_payload = self.llm.generate_structured_action(referee_messages)
        
        mechanical_status = ""
        if action_payload and action_payload.action_type != "none":
            mechanical_status = self.reconciler.reconcile(action_payload)
            if mechanical_status:
                print(f"[Engine Output] {mechanical_status}")

        rag_results = self.vector_db.search(query=user_input, limit=2)
        rag_chunks = [r["document"] for r in rag_results]
        
        dm_system_prompt = """
        You are the Dungeon Master (DM) for a locally hosted D&D game.
        Describe the physical outcomes of actions and speak for passive NPCs.
        Adhere strictly to the RAG context and any provided SYSTEM updates.
        Be descriptive and keep the pacing flowing.
        Integrate system updates naturally into the narrative prose. 
        Do not output raw technical text (like 'CRITICAL SYSTEM UPDATE') unless explicitly formatting a character stat sheet.
        """
        if mechanical_status:
            dm_system_prompt += f"\n\nCRITICAL SYSTEM UPDATE:\n{mechanical_status}\nApply this calculation strictly to your narration."

        dm_messages = PromptBuilder.compile_messages(
            system_persona=dm_system_prompt,
            rag_context_chunks=rag_chunks,
            chat_history=self.chat_history
        )

        narrative_response = self.llm.generate_narrative(dm_messages)
        
        self.chat_history.append({"role": "assistant", "content": narrative_response})
        self.chat_history = TokenManager.prune_history(self.chat_history, max_tokens=4000)

        return narrative_response

if __name__ == "__main__":
    conn = sqlite3.connect("aether_game.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM characters")
    cursor.execute("DELETE FROM items")
    cursor.execute("DELETE FROM inventories")
    
    cursor.execute("""
    INSERT INTO characters (id, name, type, strength, dexterity, constitution, intelligence, wisdom, charisma, hp, max_hp, ac)
    VALUES (1, 'player_warrior', 'player', 16, 12, 14, 10, 12, 10, 10, 20, 16)
    """)
    cursor.execute("INSERT INTO items (id, name, value, weight, description) VALUES (1, 'Healing Potion', 50, 0.5, 'Restores health.')")
    conn.commit()
    conn.close()

    orchestrator = GameOrchestrator(state="NARRATIVE_PLAY")

    print("\n--- Game Turn #1: Player takes damage ---")
    response_1 = orchestrator.process_turn("I walk down the dark corridor, but suddenly trip on a spiked floor trap!")
    print(f"\nDungeon Master:\n{response_1}")

    print("\n--- Game Turn #2: Player tries to heal ---")
    response_2 = orchestrator.process_turn("Ouch! I sit down, pull a Healing Potion out of my pack and quickly drink it. I heal myself for 8 hit points.")
    print(f"\nDungeon Master:\n{response_2}")
