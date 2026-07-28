
import sqlite3
import os
import random
from typing import List, Dict, Any, Tuple, Optional
import ollama

from md_manager import MarkdownManager
from vector_db_manager import VectorDBManager
from llm_manager import LLMController, PromptBuilder, GameActionPayload
from db import get_db_connection
from agents import DMAgent, EnvironmentAgent, ActorAgent, CombatAction, EnvironmentStateUpdate


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


class StateReconciler:
    
    def __init__(self, campaign_slug: str):
        self.campaign_slug = campaign_slug

    def _execute_query(self, query: str, params: Tuple = ()) -> List[Tuple]:
        conn = get_db_connection(self.campaign_slug)
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

        SKILL_MAP = {
            "stealth": "dexterity",
            "sleight of hand": "dexterity",
            "athletics": "strength",
            "history": "intelligence",
            "arcana": "intelligence",
            "insight": "wisdom",
            "perception": "wisdom",
            "persuasion": "charisma",
            "intimidation": "charisma"
        }

        print(f"[Reconciler] Action: {action} | Target: {target} | Val: {val}")

        if action == "skill_check":
            skill_name = str(val).lower().strip()
            ability = SKILL_MAP.get(skill_name, "dexterity")

            query = f"SELECT {ability} FROM characters WHERE id = ? OR name LIKE ?"
            res = self._execute_query(query, (target, f"%{target}%"))
            if not res:
                return f"SYSTEM REPORT: Target '{target}' not found. Roll failed."
            
            ability_score = res[0][0]
            modifier = (ability_score - 10) // 2
            
            d20_roll = random.randint(1, 20)
            total = d20_roll + modifier

            return f"SYSTEM REPORT: {target} attempted a '{skill_name}' check. Rolled {d20_roll} + {modifier} ({ability} mod) = {total}."

        elif action == "damage":
            try:
                damage_amount = int(val)
                res = self._execute_query("SELECT hp, max_hp FROM characters WHERE id = ? OR name LIKE ?", (target, f"%{target}%"))
                if not res:
                    return f"SYSTEM REPORT: Target '{target}' not found."
                curr_hp, max_hp = res[0][0], res[0][1]
                new_hp = max(0, curr_hp - damage_amount)
                self._execute_query("UPDATE characters SET hp = ? WHERE id = ? OR name LIKE ?", (new_hp, target, f"%{target}%"))
                
                status = f"SYSTEM REPORT: {target} took {damage_amount} damage. HP is now {new_hp}/{max_hp}."
                if new_hp == 0:
                    status += f" {target} has fallen unconscious or died!"
                return status
            except ValueError:
                return f"SYSTEM ERROR: Invalid damage value '{val}'."

        elif action == "heal":
            try:
                heal_amount = int(val)
                res = self._execute_query("SELECT hp, max_hp FROM characters WHERE id = ? OR name LIKE ?", (target, f"%{target}%"))
                if not res:
                    return f"SYSTEM REPORT: Target '{target}' not found."
                curr_hp, max_hp = res[0][0], res[0][1]
                new_hp = min(max_hp, curr_hp + heal_amount)
                self._execute_query("UPDATE characters SET hp = ? WHERE id = ? OR name LIKE ?", (new_hp, target, f"%{target}%"))
                return f"SYSTEM REPORT: {target} healed by {heal_amount}. HP is now {new_hp}/{max_hp}."
            except ValueError:
                return f"SYSTEM ERROR: Invalid heal value '{val}'."

        elif action == "add_item":
            item_res = self._execute_query("SELECT id FROM items WHERE name LIKE ?", (f"%{val}%",))
            if not item_res:
                self._execute_query("INSERT INTO items (name, value, weight, description) VALUES (?, 0, 1.0, 'A basic item.')", (str(val),))
                item_res = self._execute_query("SELECT id FROM items WHERE name = ?", (str(val),))
            item_id = item_res[0][0]
            
            owner_res = self._execute_query("SELECT id FROM characters WHERE id = ? OR name LIKE ?", (target, f"%{target}%"))
            if owner_res:
                owner_id, owner_type = owner_res[0][0], "actor"
            else:
                owner_id, owner_type = 1, "container"
                
            inv_res = self._execute_query(
                "SELECT id, quantity FROM inventories WHERE owner_type = ? AND owner_id = ? AND item_id = ?",
                (owner_type, owner_id, item_id)
            )
            if inv_res:
                self._execute_query("UPDATE inventories SET quantity = quantity + 1 WHERE id = ?", (inv_res[0][0],))
            else:
                self._execute_query("INSERT INTO inventories (owner_type, owner_id, item_id, quantity) VALUES (?, ?, ?, 1)", (owner_type, owner_id, item_id))
            return f"SYSTEM REPORT: Added 1x '{val}' to {target}'s inventory."

        return ""


class GameOrchestrator:
    def __init__(self, campaign_slug: str, state: str = "NARRATIVE_PLAY"):
        self.campaign_slug = campaign_slug
        self.state = state

        self.md_manager = MarkdownManager()
        self.vector_db = VectorDBManager()
        self.llm = LLMController()
        self.reconciler = StateReconciler(self.campaign_slug)
        
        self.dm_agent = DMAgent()
        self.env_agent = EnvironmentAgent()
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

    def process_narrative_turn(self, user_input: str) -> str:
        self.chat_history.append({"role": "user", "content": user_input})
        
        ref_prompt = """
        You are the silent system referee. Analyze the player's last statement and decide if any numerical, 
        stat, or item inventory updates are required.
        Choose from these exact actions: 'damage', 'heal', 'add_item', 'remove_item', 'skill_check', 'none'.
        
        CRITICAL CLASSIFICATION RULE:
        - Only classify an action as 'damage' or 'heal' if there is an explicit threat, attack, damage-dealing hazard, or consumption of a healing resource (like a potion).
        - NEVER classify standard environmental investigation, physical touching of scenery, or hand-waving as a spell or healing action.
        - If a player touches, inspects, or tests a physical object, classify it as 'skill_check' or 'none'.
        
        Output strictly in the required JSON schema format.
        """
        ref_context = [f"Active character ID: 'player_warrior'", f"Action: '{user_input}'"]
        ref_msg = PromptBuilder.compile_messages(ref_prompt, ref_context, [{"role": "user", "content": user_input}])
        action_payload = self.llm.generate_structured_action(ref_msg)
        
        mechanical_status = ""
        if action_payload and action_payload.action_type != "none":
            mechanical_status = self.reconciler.reconcile(action_payload)
            print(f"[Engine Output] {mechanical_status}")

        location_results = self.vector_db.search(query=user_input, category_filter="locations", limit=1)
        rag_chunks = [
            r["document"] for r in location_results 
            if r["metadata"].get("meta_campaign_slug") == self.campaign_slug
        ]
        
        if action_payload and action_payload.action_type != "none":
            rules_query = f"{action_payload.action_type} {action_payload.value}"
            rules_results = self.vector_db.search(query=rules_query, category_filter="rules", limit=1)
            if rules_results:
                rag_chunks.append(f"DND RULE REFERENCE:\n{rules_results[0]['document']}")
                print(f"[RAG] Injected D&D Rule Context: {rules_results[0]['metadata']['source_file']}")

        lore_results = self.vector_db.search(query=user_input, category_filter="lore", limit=1)
        if lore_results:
            rag_chunks.append(f"PROSE INSPIRATION (Style like this):\n{lore_results[0]['document']}")
            print(f"[RAG] Injected Lore Inspiration: {lore_results[0]['metadata']['source_file']}")
        
        dm_system = self.dm_agent.compile_prompt(
            location_lore="\n\n".join(rag_chunks),
            system_update=mechanical_status
        )
        dm_messages = PromptBuilder.compile_messages(dm_system, [], self.chat_history)
        narrative_response = self.llm.generate_narrative(dm_messages)
        
        self.chat_history.append({"role": "assistant", "content": narrative_response})
        self.chat_history = TokenManager.prune_history(self.chat_history)
        return narrative_response

    def initiate_combat(self) -> str:
        self.set_state("COMBAT_PLAY")
        self.chat_history.append({"role": "system", "content": "[COMBAT INITIATED! Enforcing turn order mechanics]"})
        
        self._execute_query("DELETE FROM combat_queue;")
        
        query = """
            SELECT id, name, type, dexterity, initiative_bonus 
            FROM characters 
            WHERE hp > 0 AND (location_id = ? OR type = 'player')
        """
        combatants = self._execute_query(query, (self.current_location_slug,))
        
        rolled_combatants = []
        for idx, (cid, name, char_type, dex, init_bonus) in enumerate(combatants):
            roll = self.dnd_roll(20, init_bonus)
            rolled_combatants.append({
                "id": cid,
                "name": name,
                "type": char_type,
                "roll": roll
            })
            
        rolled_combatants.sort(key=lambda x: x["roll"], reverse=True)
        self.combat_queue = rolled_combatants
        self.current_turn_index = 0
        
        for turn, actor in enumerate(rolled_combatants):
            self._execute_query(
                "INSERT INTO combat_queue (actor_id, roll_result, turn_order) VALUES (?, ?, ?)",
                (actor["id"], actor["roll"], turn)
            )

        queue_report = "\n=== INITIATIVE ORDER ===\n"
        for idx, actor in enumerate(rolled_combatants):
            queue_report += f"{idx+1}. {actor['name']} (Type: {actor['type']}) [Rolled: {actor['roll']}]\n"
        queue_report += "========================\n"
        print(queue_report)
        return queue_report

    def set_state(self, new_state: str) -> None:
        self.state = new_state
        print(f"[Orchestrator State] Transitioned to -> {self.state}")

    def process_combat_turn(self, player_action_text: Optional[str] = None) -> str:
        if self.state != "COMBAT_PLAY":
            return "Not currently in combat."

        active_actor = self.combat_queue[self.current_turn_index]
        print(f"\n[Turn: {self.current_turn_index + 1}] Active Combatant: {active_actor['name']}")

        res = self._execute_query("SELECT * FROM characters WHERE id = ?", (active_actor["id"],))
        stats_keys = ["id", "name", "type", "strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma", "hp", "max_hp", "ac", "initiative_bonus"]
        actor_stats = dict(zip(stats_keys, res[0]))

        if actor_stats["hp"] <= 0:
            print(f"[Combat] {actor_stats['name']} is incapacitated. Skipping turn.")
            self._advance_turn()
            return f"{actor_stats['name']} is unconscious and cannot act."

        mechanical_status = ""

        if actor_stats["type"] == "player":
            if not player_action_text:
                return "Awaiting player turn input..."
            
            self.chat_history.append({"role": "user", "content": f"[Combat Turn: {actor_stats['name']}] {player_action_text}"})
            
            str_mod = (actor_stats["strength"] - 10) // 2
            
            opponents = self._execute_query("SELECT id, name, ac, hp FROM characters WHERE type = 'enemy' AND hp > 0 LIMIT 1")
            if opponents:
                opp_id, opp_name, opp_ac, opp_hp = opponents[0]
                to_hit = self.dnd_roll(20, str_mod)
                print(f"[Roll] Player rolled {to_hit} to hit vs. {opp_name}'s AC of {opp_ac}.")
                
                if to_hit >= opp_ac:
                    damage = self.dnd_roll(8, str_mod)
                    new_hp = max(0, opp_hp - damage)
                    self._execute_query("UPDATE characters SET hp = ? WHERE id = ?", (new_hp, opp_id))
                    mechanical_status = f"SYSTEM COMBAT UPDATE: player_warrior hits {opp_name} for {damage} damage! {opp_name} HP is now {new_hp}."
                    if new_hp == 0:
                        mechanical_status += f" {opp_name} has fallen in battle!"
                else:
                    mechanical_status = f"SYSTEM COMBAT UPDATE: player_warrior swings and misses {opp_name}."
            else:
                mechanical_status = "SYSTEM COMBAT UPDATE: No active enemies left in view."

        else:
            print(f"[Combat AI] Thinking turn for NPC: {actor_stats['name']}...")
            opponents_desc = "player_warrior (HP: active, AC: 16)"
            actor_system = self.actor_agent.compile_combat_prompt(actor_stats, opponents_desc, [])
            
            try:
                response = ollama.chat(
                    model=self.llm.model_name,
                    messages=[{"role": "system", "content": actor_system}],
                    format=CombatAction.model_json_schema(),
                    options={"temperature": 0.0}
                )
                action_payload = CombatAction.model_validate_json(response["message"]["content"])
                print(f"[Combat AI Output] Choice: {action_payload.action_detail} on {action_payload.target_id}")
            except Exception as e:
                action_payload = CombatAction(target_id="player_warrior", action_type="melee_attack", action_detail="attacks with claw")

            if action_payload.action_type == "flee":
                flee_roll = self.dnd_roll(20, (actor_stats["dexterity"] - 10) // 2)
                if flee_roll >= 10:
                    mechanical_status = f"SYSTEM COMBAT UPDATE: {actor_stats['name']} successfully flees from combat, running into the misty woods!"
                    self.combat_queue.pop(self.current_turn_index)
                    self.current_turn_index = max(0, self.current_turn_index - 1)
                else:
                    mechanical_status = f"SYSTEM COMBAT UPDATE: {actor_stats['name']} attempts to flee, but player_warrior blocks their escape!"
            
            else:
                dex_mod = (actor_stats["dexterity"] - 10) // 2
                player_stats = self._execute_query("SELECT id, name, ac, hp FROM characters WHERE type = 'player' LIMIT 1")[0]
                p_id, p_name, p_ac, p_hp = player_stats
                
                to_hit = self.dnd_roll(20, dex_mod)
                print(f"[Roll] {actor_stats['name']} rolled {to_hit} to hit vs. Player's AC of {p_ac}.")
                
                if to_hit >= p_ac:
                    damage = self.dnd_roll(6, dex_mod)
                    new_hp = max(0, p_hp - damage)
                    self._execute_query("UPDATE characters SET hp = ? WHERE id = ?", (new_hp, p_id))
                    mechanical_status = f"SYSTEM COMBAT UPDATE: {actor_stats['name']} performs '{action_payload.action_detail}' hitting player for {damage} damage! Player HP is now {new_hp}."
                else:
                    mechanical_status = f"SYSTEM COMBAT UPDATE: {actor_stats['name']} misses player with '{action_payload.action_detail}'."

        dm_system = self.dm_agent.compile_prompt(location_lore="", system_update=mechanical_status)
        dm_system += f"\nActive Turn: It was {actor_stats['name']}'s turn. Narrate their action: '{mechanical_status}' in descriptive style."
        
        pruned_history = self._compile_pruned_combat_history()
        base_messages = PromptBuilder.compile_messages(dm_system, [], pruned_history)
        
        base_messages.append({
            "role": "user", 
            "content": f"[Referee] The mechanical action resolved as: '{mechanical_status}'. Dungeon Master, please narrate this turn's action in descriptive prose."
        })
        
        narrative_response = self.llm.generate_narrative(base_messages)
        narrative_response = narrative_response.replace("assistant\n", "").replace("assistant", "").strip()
        
        self.chat_history.append({
            "role": "assistant", 
            "content": f"[{actor_stats['name']}'s Turn] {narrative_response}",
            "mechanical_summary": mechanical_status
        })
        
        self._advance_turn()
        self._check_combat_end()
        
        return narrative_response

    def _advance_turn(self) -> None:
        self.current_turn_index = (self.current_turn_index + 1) % len(self.combat_queue)

    def _check_combat_end(self) -> None:
        enemies_alive = self._execute_query("SELECT COUNT(*) FROM characters WHERE type = 'enemy' AND hp > 0")[0][0]
        player_alive = self._execute_query("SELECT hp FROM characters WHERE type = 'player'")[0][0] > 0
        
        if enemies_alive == 0:
            print("\n[Combat Over] All threats cleared! Victorious!")
            self.set_state("NARRATIVE_PLAY")
            self.chat_history.append({"role": "system", "content": "[Combat Ended. All enemies cleared.]"})
        elif not player_alive:
            print("\n[Combat Over] Player character has fallen. Game Over.")
            self.set_state("NARRATIVE_PLAY")
            self.chat_history.append({"role": "system", "content": "[Combat Ended. The player has died.]"})

    def _compile_pruned_combat_history(self) -> List[Dict[str, str]]:
        pruned_history = []
        for turn in self.chat_history[-4:]:
            if "mechanical_summary" in turn and turn["mechanical_summary"]:
                pruned_history.append({
                    "role": turn["role"],
                    "content": f"Outcome of turn: {turn['mechanical_summary']}"
                })
            else:
                pruned_history.append({
                    "role": turn["role"],
                    "content": turn["content"]
                    
                })
        return pruned_history

if __name__ == "__main__":
    campaign = "brindlemark_dragon_spine"
    
    conn = get_db_connection(campaign)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM characters")
    
    cursor.execute("""
    INSERT INTO characters (id, name, type, strength, dexterity, constitution, intelligence, wisdom, charisma, hp, max_hp, ac, initiative_bonus)
    VALUES (1, 'player_warrior', 'player', 16, 12, 14, 10, 12, 10, 20, 20, 16, 1)
    """)
    cursor.execute("""
    INSERT INTO characters (id, name, type, strength, dexterity, constitution, intelligence, wisdom, charisma, hp, max_hp, ac, initiative_bonus)
    VALUES (2, 'Gruk the Rusty', 'enemy', 8, 14, 10, 7, 8, 5, 8, 8, 12, 2)
    """)
    conn.commit()
    conn.close()

    engine = GameOrchestrator(campaign_slug=campaign)

    print("\n=======================================================")
    print("          TEST PHASE 1: NARRATIVE LOOP")
    print("=======================================================")
    exp_reply = engine.process_narrative_turn("I enter the misty tree-line of the Whispering Woods, looking for shelter.")
    print(f"\nDM Narration:\n{exp_reply}")

    print("\n=======================================================")
    print("          TEST PHASE 2: INITIATING COMBAT")
    print("=======================================================")
    engine.initiate_combat()

    p_attack = "I draw my broadsword and swing it hard at Gruk the Rusty!"
    turn_1_narrative = engine.process_combat_turn(player_action_text=p_attack)
    print(f"\nDungeon Master Narration:\n{turn_1_narrative}")

    turn_2_narrative = engine.process_combat_turn()
    print(f"\nDungeon Master Narration:\n{turn_2_narrative}")
