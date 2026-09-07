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
          │   Orchestrator    │◄───► [ Vector DB / RAG ] (ChromaDB + bge-small-en-v1.5)
          │  (State Router)   │
          └─────────┬─────────┘
                    │ (Generates Narrative & State Updates)
                    ▼
          ┌───────────────────┐
          │ State Reconciler  │◄───► [ Relational DB ] (SQLite)
          │  (JSON Parser)    │
          └─────────┬─────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ Markdown Manager  │◄───► [ File System Storage ] (.md + YAML Front-Matter)
          │ (Profile Tracker) │
          └─────────┬─────────┘
                    │
                    ▼
           [ DM Agent Narrative ]
```

### Key Technical Achievements

* **Resource-Optimized Hybrid RAG:** Running multiple models locally can cause high VRAM contention. Aether solves this by dividing tasks: text generation is handled by `llama3.2` via Ollama, while embeddings are processed entirely *in-process* using `sentence-transformers` with the lightweight `BAAI/bge-small-en-v1.5` model. This split execution prevents embedding searches from blocking Ollama's generation queue and allows fast semantic lookup within a tight token footprint.
* **State-Split Storage Architecture:** Instead of forcing highly qualitative or highly structured data into one unfitted storage type, Aether divides states by their native structures:
  1. *Relational Layer (SQLite):* Handles quantitative game states that require strict constraints (such as base attributes, health pools, armor class, and physical location coordinates).
  2. *Document Layer (Markdown with YAML):* Handles flexible, unstructured data (such as rich character backgrounds, active inventories, and descriptive room properties). This prevents database schema bloat while keeping profiles easily editable.
* **Decoupled Persona Engine:** Rather than overloading a single prompt to manage the entire simulation, Aether isolates responsibilities:
  1. *Dungeon Master (DM) Agent:* Generates chronological narrative descriptions and scenery but is strictly forbidden from speaking directly for actors.
  2. *Actor Agent:* Swaps in character stats from SQLite and unique behavioral profiles from Markdown to execute immersive, in-character dialogues and tactical actions.
  3. *State Router:* A procedural state machine that coordinates spatial movement, JIT location descriptions, and validates gameplay integrity.

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
pip install -r requirements.txt
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
