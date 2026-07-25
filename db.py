import sqlite3
import os

DB_NAME = "aether_game.db"

def init_db():
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
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA foreign_keys = ON;")
        
        cursor.executescript(schema)
        conn.commit()
        
        print(f"Database successfully initialized: {os.path.abspath(DB_NAME)}")
        
    except sqlite3.Error as e:
        print(f"An error occurred while initializing the database: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    init_db()
