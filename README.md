# Aether: Stateful LLM Orchestrator for Locally Hosted Tabletop RPGs

Aether is a local, stateful multi-agent orchestrator designed to run D&D campaigns using a single lightweight local LLM (such as Llama 3.2 3B). The project establishes a robust system boundary by combining structured relational databases (SQLite) for numeric game states (such as health, inventory, and initiative orders) with semantic vector databases (ChromaDB) for narrative context and rules retrieval.

---

## Technical Architecture Overview

Aether avoids the common pitfalls of naive LLM wrappers by separating **narrative generation** (which requires high creative temperature) from **mechanical state reconciliation** (which requires absolute deterministic parsing). 

```
               [ User Input ]
                     │
                     ▼
          ┌───────────────────┐
          │   Orchestrator    │◄───► [ Vector DB / RAG ] (ChromaDB / mxbai-embed-large)
          │  (State Router)   │      - Locations, rules, lore
          └─────────┬─────────┘
                    │ (Generates Narrative & State Payload)
                    ▼
          ┌───────────────────┐
          │ State Reconciler  │◄───► [ Relational DB ] (SQLite Campaign-Specific Slot)
          │  (JSON Parser)    │      - Characters, inventories, combat queue
          └─────────┬─────────┘
                    │
                    ▼
           [ DM Agent Narrative ]
```

### Key Technical Achievements

* **Relational State Continuity:** The architecture uses a dynamic save-slot paradigm where each campaign lives in its own isolated filesystem directory (e.g., `data/campaigns/{campaign_slug}/`) containing its own private SQLite database (`game.db`). This ensures data isolation, prevents cross-campaign data pollution, and allows users to backup, share, or delete entire campaign states cleanly by moving directories.
* **Resource-Optimized Hybrid RAG:** Running multiple models locally often leads to GPU VRAM contention. Aether handles this by routing both text generation (`llama3.2`) and embedding vector generation (`mxbai-embed-large`) through a single local Ollama server process, which schedules resources efficiently. It applies semantic RAG with metadata filtering, querying only specific subsets (such as `rules`, `locations`, or `lore`) based on the current game phase, which keeps the token footprint within a highly responsive 4,000-token window.
* **Grammar-Constrained Asset Generation:** Aether leverages Ollama's native JSON mode and Pydantic schema validation to force the local model's GPU token output loop to strictly conform to structural schemas. This guarantees deterministic formatting during campaign synthesis, character creation, and combat evaluations thus eliminating JSON parsing errors or schema mismatches even at high temperatures.
* **Decoupled Agent Persona Engine:** Instead of trying to force a single prompt to manage the entire world, Aether splits responsibilities into three logical, context-swapping agents:
  1. *Dungeon Master (DM) Agent:* Handles chronological world narration, descriptions, and rules refereeing, but is strictly forbidden from speaking for characters.
  2. *Environment Agent:* Evaluates physical object interactions, room attributes, and weather changes, returning structured updates.
  3. *Actor Agent (NPCs/Enemies):* Swaps in character sheet statistics from SQLite and distinct behavioral profiles from Markdown to execute roleplay dialogue or tactical combat actions.

---

## Directory Structure

```text
.
├── db.py                 # Dynamic SQLite save-slot manager
├── md_manager.py         # Directory and Markdown manager with YAML front-matter parsing
├── vector_db_manager.py  # ChromaDB ingestion, recursive directory indexing, and mxbai-embed RAG
├── llm_manager.py        # Ollama API wrapper, Pydantic schemas, and structured JSON parsing
├── agents.py             # Decoupled prompt templates and compilers for DM, Environment, and Actor agents
├── campaign_creator.py   # Interactive campaign creator with dynamic checklist tracking
├── character_creator.py  # D&D character generator with rules constraints and SQLite registration
├── orchestrator.py       # Core game controller managing Narrative Mode, Combat Mode, and turn tracking
├── main.py               # Master terminal-based game launcher
└── README.md             # Documentation
```

---

## How to Run the Project

Follow these steps to set up, index, and launch the game locally:

### 1. Prerequisiutes & System Setup
Ensure you have the local Ollama server running on your machine.
Download the required LLM and embedding models:
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### 2. Install Python Dependencies
Create a virtual environment and install the required libraries:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install sentence-transformers chromadb ollama pydantic pyyaml
```

### 3. Clone and Import the D&D Rulebook (optional)
Aether supports bulk-indexing of open-source rulebooks. Clone the official, community-converted D&D 5e System Reference Document (SRD) directly into your rules directory:
```bash
mkdir -p data/rules
cd data/rules
git clone https://github.com/downfallx/dnd-5e-srd-markdown
# Clean up git files to keep directories clean
cd dnd-5e-srd-markdown && rm -rf .git
cd ../../..
```

### 4. Bulk Index the Rules and Lore Database
Run the vector database manager to recursively scan the newly cloned rulebook subdirectories (such as `spells`, `monsters`, and `mechanics`), split them by headings, and load them into ChromaDB:
```bash
python3 vector_db_manager.py
```

### 5. Launch the Master Game Launcher
Launch the console interface to start your campaign creation, build your protagonist, and begin active play:
```bash
python3 main.py
```

#### Inside the Game Terminal:
* **Conversational World-Building:** Use the interactive creation terminals to consult with the AI. Type `complete` (or let the dynamic checklist auto-complete) to generate and index your custom setting files on disk and SQLite.
* **Check Status (/ Commands):** At any point during active narrative exploration, you can bypass the LLM and query the database directly for exact stat values by typing `/stats` or `/inventory` in the console.
* **Combat Mode Initiation:** Typing words like `attack` or `combat` will prompt the system to scan your current coordinates. If active enemies are present in SQLite at your location, the engine automatically transitions to turn-based Combat Mode.

---

## Roadmap for Future Development

The following architectural updates are designed to further elevate Aether's systems design:

### 1. Contextual Feasibility: Detecting and Rejecting Illogical Player Inputs
* **The Goal:** Prevent players from performing actions that make no logical or physical sense in their active surroundings (such as trying to *"attack an undead skeleton"* while celebrating at a peaceful festival inside a village tavern).
* **The System Implementation:** 
  1. Expand the existing `GameActionPayload` Pydantic schema in `llm_manager.py` to include `is_contextually_feasible: bool` and an optional `rejection_reason: str`.
  2. The Referee Agent will analyze the user input against the recent history. If the action is contextually impossible, it sets `is_contextually_feasible` to `False` and details the failure.
  3. The State Router in `orchestrator.py` intercepts this flag, bypasses SQLite updates entirely, and passes the rejection reason directly to the DM Agent.
  4. The DM Agent narrates the player's physical or social failure (e.g., describing the awkward silence of the tavern patrons as the player draws their sword against imaginary phantoms).

### 2. State-Persistence: Conversational History Serialization & Custom Openings
* **The Goal:** Fully preserve active conversation history across application restarts, preventing the DM from "forgetting" the active scene and defaulting back to a generic opening.
* **The System Implementation:**
  * **Short-Term Memory Disk Serialization:** Every time a turn resolves, serialize `self.chat_history` directly to `data/campaigns/{campaign_slug}/chat_history.json` on disk.
  * **On-Demand Deserialization:** When `GameOrchestrator` initializes, search for this save file. If present, deserialize and load it into RAM.
  * **Dynamic Campaign-Specific Intros:** Completely eliminate the generic `"I open my eyes..."` startup prompt in `main.py`. If `chat_history.json` is empty, read the campaign profile (`campaign_info.md`) to extract the custom `setting_description` and `starting_quest_hook`. Have the DM Agent generate a fully custom, lore-accurate introductory sequence on the first run.

### 3. Rules & Lore Ingestion Segregation
* **The Goal:** Implement a dynamic indexer call inside `vector_db_manager.py` for `/data/lore/`. 
* **The System Implementation:** Configure the recursive directory crawler (`index_directory`) to crawl a dedicated `/data/lore/` folder. This allows developers to drag-and-drop fantasy novels (like *The Lord of the Rings* converted to markdown) directly into the database. By separating the rules database (`data/rules/`) from the narrative database (`data/lore/`), the orchestrator can target its RAG queries using `category_filters`, keeping mechanical rules and colorful prose inspirations entirely segregated to prevent token context pollution.
