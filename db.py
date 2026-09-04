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
        CREATE TABLE IF NOT EXISTS campaign_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_name TEXT NOT NULL,
            setting_description TEXT NOT NULL,
            primary_threat TEXT NOT NULL,
            starting_quest_hook TEXT NOT NULL,
            theme_vibe TEXT,
            starting_patron TEXT,
            victory_condition TEXT
        );

        CREATE TABLE IF NOT EXISTS locations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            brief_concept TEXT NOT NULL,
            is_generated INTEGER NOT NULL DEFAULT 0 CHECK(is_generated IN (0, 1))
        );

        CREATE TABLE IF NOT EXISTS location_connections (
            from_location_id TEXT NOT NULL,
            to_location_id TEXT NOT NULL,
            PRIMARY KEY (from_location_id, to_location_id),
            FOREIGN KEY (from_location_id) REFERENCES locations(id) ON DELETE CASCADE,
            FOREIGN KEY (to_location_id) REFERENCES locations(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('player', 'ally', 'enemy', 'merchant')),
            location_id TEXT NOT NULL DEFAULT 'global',
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
        """
        try:
            conn.executescript(schema)
            conn.commit()
            print(f"Dynamic save-slot database successfully initialized at: {os.path.abspath(db_path)}")
        except sqlite3.Error as e:
            print(f"Failed to initialize save-slot database: {e}")
            raise e
            
    return conn
