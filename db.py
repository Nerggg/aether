import os
import sqlite3

def get_db_connection(campaign_slug: str, base_dir: str = "data") -> sqlite3.Connection:
    db_dir = os.path.join(base_dir, "campaigns", campaign_slug)
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "game.db")
    
    db_is_new = not os.path.exists(db_path)
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    
    if db_is_new:
        schema = """
        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('player', 'ally', 'enemy', 'merchant')),
            strength INTEGER NOT NULL CHECK(strength >= 1 AND strength <= 30),
            dexterity INTEGER NOT NULL CHECK(dexterity >= 1 AND dexterity <= 30),
            constitution INTEGER NOT NULL CHECK(constitution >= 1 AND constitution <= 30),
            intelligence INTEGER NOT NULL CHECK(intelligence >= 1 AND intelligence <= 30),
            wisdom INTEGER NOT NULL CHECK(wisdom >= 1 AND wisdom <= 30),
            charisma INTEGER NOT NULL CHECK(charisma >= 1 AND charisma <= 30),
            hp INTEGER NOT NULL CHECK(hp >= 0),
            max_hp INTEGER NOT NULL CHECK(max_hp > 0),
            ac INTEGER NOT NULL CHECK(ac >= 0),
            initiative_bonus INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            value INTEGER NOT NULL CHECK(value >= 0),
            weight REAL NOT NULL CHECK(weight >= 0.0),
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS inventories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_type TEXT NOT NULL CHECK(owner_type IN ('player', 'actor', 'container')),
            owner_id INTEGER NOT NULL,
            item_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1 CHECK(quantity > 0),
            FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS combat_queue (
            actor_id INTEGER PRIMARY KEY,
            roll_result INTEGER NOT NULL CHECK(roll_result >= 1),
            turn_order INTEGER NOT NULL UNIQUE,
            FOREIGN KEY (actor_id) REFERENCES characters(id) ON DELETE CASCADE
        );
        """
        try:
            conn.executescript(schema)
            conn.commit()
            print(f"Dynamic save-slot database successfully initialized at: {os.path.abspath(db_path)}")
        except sqlite3.Error as e:
            print(f"Failed to initialize save-slot database: {e}")
            raise e
            
    return conn
