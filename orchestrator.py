import sqlite3
import os
import random
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import ollama
import inspect
import re

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
        
        char_sheet_block = self._get_active_character_sheet()
        
        conn = get_db_connection(self.campaign_slug)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name, type, hp, max_hp, ac, is_persistent 
            FROM characters 
            WHERE location_id = ? AND type != 'player'
        """, (self.current_location_slug,))
        local_characters = cursor.fetchall()
        conn.close()
        
        local_presence_block = ""
        if local_characters:
            local_presence_block = "\n### CHARACTERS PRESENT AT CURRENT LOCATION:\n"
            for name, char_type, hp, max_hp, ac, is_p in local_characters:
                persist_str = "Persistent" if is_p else "Ephemeral"
                local_presence_block += f"- **{name}** ({char_type.upper()} | {persist_str}) | HP: {hp}/{max_hp} | AC: {ac}\n"
        
        vector = self.vector_db.embedder.get_embeddings([f"{self.current_location_slug} {user_input}"])[0]
        location_results = self.vector_db.search(query_vector=vector, category_filter="locations", limit=1)
        lore_results = self.vector_db.search(query_vector=vector, category_filter="lore", limit=1)

        rag_chunks = [
            r["document"] for r in location_results 
            if r["metadata"].get("meta_campaign_slug") == self.campaign_slug
        ]
        if lore_results:
            rag_chunks.append(f"PROSE INSPIRATION:\n{lore_results[0]['document']}")
            
        location_lore_combined = "\n\n".join(rag_chunks)
        if local_presence_block:
            location_lore_combined += f"\n{local_presence_block}"
            
        dm_system = self.dm_agent.compile_prompt(
            location_lore=location_lore_combined,
            system_update=movement_status if movement_status else ""
        )
        
        dm_system = f"{char_sheet_block}\n\n{dm_system}"
        
        dm_system += """\n
        CRITICAL RULES FOR MECHANICAL RESOLUTION:
        1. If the player attempts a risky, uncertain, or challenging action, you must NOT resolve the outcome. You must immediately halt generation and output this exact tag: [ROLL: skill_name] (choose from the 18 official skills or raw attributes, e.g. [ROLL: stealth] or [ROLL: athletics]).
        2. If the player gains or loses an item, or takes damage/healing, append the state update tag at the very end of your response:
           - [ADD_ITEM: item_name]
           - [REMOVE_ITEM: item_name]
           - [TAKE_DAMAGE: amount]
           - [HEAL_HP: amount]
        """

        dm_messages = PromptBuilder.compile_messages(dm_system, [], self.chat_history)
        
        yield from self._stream_and_intercept(dm_messages)

    def _stream_and_intercept(self, dm_messages: List[Dict[str, str]]):
        """Streams LLM generation, buffering and executing inline rolls and silent state updates."""
        buffer = ""
        in_tag = False
        full_response_text = ""
        
        for chunk in self.llm.generate_narrative_stream(dm_messages):
            for char in chunk:
                if char == "[":
                    in_tag = True
                    buffer += char
                elif char == "]":
                    buffer += char
                    in_tag = False
                    
                    tag_content = buffer.strip("[]")
                    
                    if tag_content.startswith("ROLL:"):
                        skill = tag_content.split(":", 1)[1].strip().lower()
                        print("\n")
                        
                        roll_result_str = self._execute_inline_roll(skill)
                        print(roll_result_str)
                        
                        self.chat_history.append({"role": "assistant", "content": full_response_text})
                        self.chat_history.append({
                            "role": "user", 
                            "content": f"[SYSTEM: Your d20 roll for {skill} was resolved. Roll outcome: {roll_result_str}. Describe the immediate consequence of this roll in character. Do not repeat this system message.]"
                        })
                        self.chat_history = TokenManager.prune_history(self.chat_history)
                        
                        char_sheet_block = self._get_active_character_sheet()
                        
                        vector = self.vector_db.embedder.get_embeddings([self.current_location_slug])[0]
                        location_results = self.vector_db.search(query_vector=vector, category_filter="locations", limit=1)
                        rag_chunks = [r["document"] for r in location_results if r["metadata"].get("meta_campaign_slug") == self.campaign_slug]
                        
                        dm_system = self.dm_agent.compile_prompt(location_lore="\n\n".join(rag_chunks))
                        dm_system = f"{char_sheet_block}\n\n{dm_system}"
                        dm_system += """\n
                        CRITICAL RULES FOR MECHANICAL RESOLUTION:
                        1. If the player attempts a risky, uncertain, or challenging action, you must NOT resolve the outcome. You must immediately halt generation and output this exact tag: [ROLL: skill_name].
                        2. If the player gains or loses an item, or takes damage/healing, append the state update tag at the very end of your response:
                           - [ADD_ITEM: item_name]
                           - [REMOVE_ITEM: item_name]
                           - [TAKE_DAMAGE: amount]
                           - [HEAL_HP: amount]
                        """
                        
                        new_messages = PromptBuilder.compile_messages(dm_system, [], self.chat_history)
                        yield from self._stream_and_intercept(new_messages)
                        return
                        
                    elif any(tag_content.startswith(act) for act in ["ADD_ITEM", "REMOVE_ITEM", "TAKE_DAMAGE", "HEAL_HP"]):
                        self._update_character_state(buffer)
                        buffer = ""
                    else:
                        yield buffer
                        full_response_text += buffer
                        buffer = ""
                else:
                    if in_tag:
                        buffer += char
                    else:
                        yield char
                        full_response_text += char
                        
        if buffer:
            yield buffer
            full_response_text += buffer
            
        self.chat_history.append({"role": "assistant", "content": full_response_text})
        self.chat_history = TokenManager.prune_history(self.chat_history)

        self._run_referee_state_reconciliation()

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
                    
                    db_conn = get_db_connection(self.campaign_slug)
                    cursor = db_conn.cursor()
                    try:
                        cursor.execute("""
                            DELETE FROM characters 
                            WHERE location_id = ? AND type != 'player' AND is_persistent = 0
                        """, (old_loc,))
                        db_conn.commit()
                        purged_count = cursor.rowcount
                        if purged_count > 0:
                            print(f"[Engine State] Purged {purged_count} ephemeral NPCs from old location '{old_loc}'")
                    except sqlite3.Error as e:
                        print(f"[Engine Error] Failed to purge ephemeral NPCs: {e}")
                    finally:
                        db_conn.close()

                    self.current_location_slug = conn["id"]
                    self._ensure_location_generated(conn["id"])
                    
                    return f"SYSTEM UPDATE: Player successfully moved from '{old_loc}' to '{conn['id']}' ({conn['name']}). You must narrate their departure, short journey, and physical arrival."
            
            return "SYSTEM UPDATE: Player attempted to travel to an unmapped area or leave the boundaries of the current location. You must narrate a natural physical or environmental obstacle blocking their path and redirect them back."
            
        return None

    def _get_player_name(self) -> str:
        """Retrieves the active player character's name from SQLite."""
        conn = get_db_connection(self.campaign_slug)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM characters WHERE type = 'player' LIMIT 1")
        res = cursor.fetchone()
        conn.close()
        return res[0] if res else "player_warrior"

    def slugify(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        return re.sub(r"[-\s]+", "_", text)

    def _get_active_character_sheet(self) -> str:
        """Queries SQLite and Markdown to compile the current character sheet and inventory."""
        conn = get_db_connection(self.campaign_slug)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name, strength, dexterity, constitution, intelligence, wisdom, charisma, hp, max_hp, ac, initiative_bonus 
            FROM characters 
            WHERE type = 'player' LIMIT 1
        """)
        char_row = cursor.fetchone()
        conn.close()
        
        if not char_row:
            return ""
            
        name, str_, dex, con, intel, wis, cha, hp, max_hp, ac, init_bonus = char_row
        char_slug = self.slugify(name)
        
        inventory_items = []
        try:
            metadata, _ = self.md_manager.read_file("actors", f"{self.campaign_slug}_{char_slug}")
            inventory_items = metadata.get("inventory", [])
            if isinstance(inventory_items, str):
                inventory_items = [i.strip() for i in inventory_items.split(",") if i.strip()]
        except Exception:
            inventory_items = ["Leather armor", "Short sword", "Stale bread"]
            
        inventory_str = ", ".join(inventory_items) if inventory_items else "Empty"
        
        sheet = f"""
        [Active Character Sheet]
        Name: {name}
        Class/Race: Elf Ranger
        HP: {hp}/{max_hp} | AC: {ac} | Initiative Bonus: +{init_bonus}
        Attributes: STR:{str_} | DEX:{dex} | CON:{con} | INT:{intel} | WIS:{wis} | CHA:{cha}
        Inventory: {inventory_str}
        """
        return inspect.cleandoc(sheet)

    def _update_character_state(self, tag: str) -> None:
        """Parses state tags like [ADD_ITEM: x] and updates SQLite and Markdown files on disk."""
        conn = get_db_connection(self.campaign_slug)
        cursor = conn.cursor()
        cursor.execute("SELECT name, hp, max_hp FROM characters WHERE type = 'player' LIMIT 1")
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return
            
        name, hp, max_hp = row
        char_slug = self.slugify(name)
        
        tag = tag.strip("[]")
        parts = tag.split(":", 1)
        if len(parts) < 2:
            conn.close()
            return
            
        action, value = parts[0].strip().upper(), parts[1].strip()
        
        if action == "ADD_ITEM":
            try:
                metadata, content = self.md_manager.read_file("actors", f"{self.campaign_slug}_{char_slug}")
                inventory = metadata.get("inventory", [])
                if isinstance(inventory, str):
                    inventory = [i.strip() for i in inventory.split(",") if i.strip()]
                if value not in inventory:
                    inventory.append(value)
                metadata["inventory"] = inventory
                self.md_manager.write_file("actors", f"{self.campaign_slug}_{char_slug}", metadata, content)
                print(f"\n[Engine State] Item added to inventory: {value}")
            except Exception as e:
                print(f"Error adding item: {e}")
                
        elif action == "REMOVE_ITEM":
            try:
                metadata, content = self.md_manager.read_file("actors", f"{self.campaign_slug}_{char_slug}")
                inventory = metadata.get("inventory", [])
                if isinstance(inventory, str):
                    inventory = [i.strip() for i in inventory.split(",") if i.strip()]
                
                inventory = [item for item in inventory if item.lower() != value.lower()]
                metadata["inventory"] = inventory
                self.md_manager.write_file("actors", f"{self.campaign_slug}_{char_slug}", metadata, content)
                print(f"\n[Engine State] Item removed from inventory: {value}")
            except Exception as e:
                print(f"Error removing item: {e}")
                
        elif action == "TAKE_DAMAGE":
            try:
                dmg = abs(int(value))
                new_hp = max(0, hp - dmg)
                cursor.execute("UPDATE characters SET hp = ? WHERE name = ?", (new_hp, name))
                conn.commit()
                print(f"\n[Engine State] Player took {dmg} damage. HP: {new_hp}/{max_hp}")
            except ValueError:
                pass
                
        elif action == "HEAL_HP":
            try:
                heal = abs(int(value))
                new_hp = min(max_hp, hp + heal)
                cursor.execute("UPDATE characters SET hp = ? WHERE name = ?", (new_hp, name))
                conn.commit()
                print(f"\n[Engine State] Player healed for {heal} HP. HP: {new_hp}/{max_hp}")
            except ValueError:
                pass
                
        conn.close()

    def _execute_inline_roll(self, skill: str) -> str:
        """Executes a D&D d20 roll in Python, applying modifiers from the player's database sheet."""
        player_name = self._get_player_name()
        
        SKILL_MAP = {
            "athletics": "strength", "acrobatics": "dexterity", "sleight of hand": "dexterity", "stealth": "dexterity",
            "arcana": "intelligence", "history": "intelligence", "investigation": "intelligence", "nature": "intelligence", "religion": "intelligence",
            "animal handling": "wisdom", "insight": "wisdom", "medicine": "wisdom", "perception": "wisdom", "survival": "wisdom",
            "deception": "charisma", "intimidation": "charisma", "performance": "charisma", "persuasion": "charisma"
        }
        
        ability = SKILL_MAP.get(skill, skill)
        allowed_columns = ["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]
        if ability not in allowed_columns:
            ability = "dexterity"
            
        conn = get_db_connection(self.campaign_slug)
        cursor = conn.cursor()
        cursor.execute(f"SELECT {ability} FROM characters WHERE name = ?", (player_name,))
        res = cursor.fetchone()
        conn.close()
        
        score = res[0] if res else 10
        modifier = (score - 10) // 2
        
        import time
        print(f"\n[Engine] Attempting roll for '{skill}' ({ability})... ", end="", flush=True)
        time.sleep(0.4)
        for _ in range(3):
            print(".", end="", flush=True)
            time.sleep(0.3)
            
        d20 = random.randint(1, 20)
        total = d20 + modifier
        
        return f"Got {d20} + {modifier} ({ability} modifier) = {total}!"

    def _run_referee_state_reconciliation(self) -> None:
        """Invokes the Referee Agent silently to evaluate if a new actor was dynamically met."""
        referee_system = """
        You are the system referee. Analyze the recent conversational turn (user action and the DM description) to decide if a new character was introduced, met, or spawned in the scene.
        
        CRITICAL RULES:
        1. If a new character is introduced, set action_type to 'spawn_npc' and populate 'spawned_npc'.
        2. Set is_persistent to true ONLY for major recurring allies, named bosses, or significant quest-givers.
        3. Set is_persistent to false for generic, non-crucial NPCs (shopkeepers, guards, low-importance enemies like generic goblins).
        4. If no character is introduced, set action_type to 'none'.
        """
        
        referee_history = self.chat_history[-2:]
        compiled_referee = PromptBuilder.compile_messages(referee_system, [], referee_history)
        
        payload = self.llm.generate_structured_action(
            messages=compiled_referee,
            response_schema=GameActionPayload,
            temperature=0.0
        )
        
        if payload and payload.action_type == "spawn_npc" and payload.spawned_npc:
            print(f"\n[Referee] Detected dynamic NPC introduction: '{payload.spawned_npc.name}' ({payload.spawned_npc.type})")
            self._handle_dynamic_npc_spawn(payload.spawned_npc)

    def _handle_dynamic_npc_spawn(self, npc_details) -> None:
        """Saves met NPCs. Persistent ones get Markdown + SQLite profiles, generic ones are SQLite-only."""
        name = npc_details.name
        char_type = npc_details.type
        is_persistent = 1 if npc_details.is_persistent else 0
        backstory = npc_details.brief_backstory or f"A generic {char_type} met in the campaign."
        template_id = npc_details.template_id
        
        if is_persistent == 1:
            strength, dexterity, constitution = 14, 12, 13
            intelligence, wisdom, charisma = 10, 12, 14
            hp, max_hp, ac = 12, 12, 13
        else:
            strength, dexterity, constitution = 10, 10, 10
            intelligence, wisdom, charisma = 10, 10, 10
            hp, max_hp, ac = 6, 6, 10
            
        if template_id:
            if "goblin" in template_id.lower():
                strength, dexterity, constitution = 8, 14, 10
                intelligence, wisdom, charisma = 10, 8, 8
                hp, max_hp, ac = 7, 7, 12
            elif "giant_rat" in template_id.lower():
                strength, dexterity, constitution = 7, 15, 11
                intelligence, wisdom, charisma = 2, 10, 4
                hp, max_hp, ac = 7, 7, 12

        initiative_bonus = (dexterity - 10) // 2
        
        conn = get_db_connection(self.campaign_slug)
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO characters (name, type, location_id, strength, dexterity, constitution, intelligence, wisdom, charisma, hp, max_hp, ac, initiative_bonus, is_persistent, template_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, char_type, self.current_location_slug,
                strength, dexterity, constitution, intelligence, wisdom, charisma,
                hp, hp, ac, initiative_bonus, is_persistent, template_id
            ))
            conn.commit()
            print(f"[Engine State] Registered NPC '{name}' in SQLite under '{self.current_location_slug}'")
        except sqlite3.Error as e:
            print(f"[Engine Error] SQLite dynamic insertion failed: {e}")
        finally:
            conn.close()
            
        if is_persistent == 1:
            char_slug = self.slugify(name)
            filename = f"{self.campaign_slug}_{char_slug}"
            
            actor_metadata = {
                "id": char_slug,
                "campaign_slug": self.campaign_slug,
                "name": name,
                "type": char_type,
                "is_persistent": True
            }
            
            actor_content = (
                f"# Actor Sheet: {name}\n\n"
                f"**Role:** {char_type.capitalize()} | **Location:** {self.current_location_slug}\n\n"
                f"## Backstory & Background Lore\n{backstory}\n\n"
                f"## Determined Stats\n"
                f"- STR: {strength} | DEX: {dexterity} | CON: {constitution}\n"
                f"- INT: {intelligence} | WIS: {wisdom} | CHA: {charisma}\n\n"
                f"## Combat Profile\n"
                f"- **HP:** {hp}/{max_hp}\n"
                f"- **AC:** {ac}\n"
            )
            
            try:
                self.md_manager.write_file(
                    category="actors",
                    filename=filename,
                    metadata=actor_metadata,
                    content=actor_content
                )
                self.vector_db.upsert_markdown_file(category="actors", filename=filename)
                print(f"[Engine State] Wrote detailed persistent character Markdown profile: '{filename}.md'")
            except Exception as e:
                print(f"[Engine Error] Failed to write persistent Markdown profile: {e}")

if __name__ == "__main__":
    campaign = "brindlemark_dragon_spine"
    
    engine = GameOrchestrator(campaign_slug=campaign)
    print(f"Orchestrator successfully initialized for campaign: {engine.campaign_slug}")
