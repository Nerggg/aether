"""SQLite persistence for Aether.

One campaign = one folder = one ``game.db`` file:

    <data_dir>/campaigns/<slug>/game.db

All game state lives here: campaign info, the world map, characters,
inventory, the append-only transcript, summaries, story beats and the
small amount of loop state needed to resume a game.

Multi-step updates should be wrapped in ``with game.transaction():`` so a
crash can never leave a half-saved turn.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from aether.config import get_config
from aether.models import (
    BEAT_CONDITIONS,
    BEAT_STATUSES,
    CHARACTER_STATUSES,
    CHARACTER_TYPES,
    MESSAGE_ROLES,
    CampaignInfo,
    Character,
    GameState,
    Location,
    Message,
    StoryBeat,
    Summary,
)

SCHEMA_VERSION = 1
DB_FILENAME = "game.db"
_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


class CampaignExistsError(Exception):
    pass


class SchemaVersionError(Exception):
    pass


def slugify(text: str) -> str:
    """Turn any title into a safe lowercase slug (letters, digits, underscores)."""
    text = re.sub(r"[^\w\s-]", "", text.lower().strip())
    text = re.sub(r"[-\s]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text).strip("_")
    return text or "campaign"


def _check_in(column: str, values: Sequence[str]) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"CHECK({column} IN ({quoted}))"


def _stat(column: str) -> str:
    return f"{column} INTEGER NOT NULL DEFAULT 10 CHECK({column} BETWEEN 1 AND 30)"


SCHEMA = f"""
CREATE TABLE locations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    brief_concept TEXT NOT NULL,
    description TEXT                      -- NULL until just-in-time generation
);

CREATE TABLE location_connections (
    from_location_id TEXT NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    to_location_id   TEXT NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
    PRIMARY KEY (from_location_id, to_location_id),
    CHECK(from_location_id <> to_location_id)
);

CREATE TABLE campaign (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    theme_vibe TEXT NOT NULL DEFAULT '',
    starting_patron TEXT NOT NULL DEFAULT '',
    setting_description TEXT NOT NULL,
    primary_threat TEXT NOT NULL,
    starting_quest_hook TEXT NOT NULL,
    storyline_beginning TEXT NOT NULL DEFAULT '',
    storyline_middle TEXT NOT NULL DEFAULT '',
    storyline_end TEXT NOT NULL DEFAULT '',
    victory_condition TEXT NOT NULL DEFAULT '',
    starting_location_id TEXT REFERENCES locations(id),
    embedding_model TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL {_check_in('type', CHARACTER_TYPES)},
    location_id TEXT NOT NULL REFERENCES locations(id),
    race TEXT,
    char_class TEXT,
    {_stat('strength')},
    {_stat('dexterity')},
    {_stat('constitution')},
    {_stat('intelligence')},
    {_stat('wisdom')},
    {_stat('charisma')},
    hp INTEGER NOT NULL CHECK(hp >= 0),
    max_hp INTEGER NOT NULL CHECK(max_hp > 0),
    status TEXT NOT NULL DEFAULT 'alive' {_check_in('status', CHARACTER_STATUSES)},
    is_persistent INTEGER NOT NULL DEFAULT 1 CHECK(is_persistent IN (0, 1)),
    template_id TEXT,
    backstory TEXT NOT NULL DEFAULT '',
    CHECK(hp <= max_hp)
);
CREATE UNIQUE INDEX one_player ON characters(type) WHERE type = 'player';
CREATE INDEX characters_by_location ON characters(location_id);

CREATE TABLE inventory (
    character_id INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    item_name TEXT NOT NULL COLLATE NOCASE,
    quantity INTEGER NOT NULL DEFAULT 1 CHECK(quantity > 0),
    PRIMARY KEY (character_id, item_name)
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL {_check_in('role', MESSAGE_ROLES)},
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    covers_through_message_id INTEGER NOT NULL REFERENCES messages(id),
    narrative TEXT NOT NULL,
    key_npcs TEXT NOT NULL DEFAULT '[]',        -- JSON list
    open_threads TEXT NOT NULL DEFAULT '[]',    -- JSON list
    promises_debts TEXT NOT NULL DEFAULT '[]',  -- JSON list
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE story_beats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position INTEGER NOT NULL UNIQUE,
    title TEXT NOT NULL,
    objective TEXT NOT NULL,
    condition_type TEXT NOT NULL DEFAULT 'referee' {_check_in('condition_type', BEAT_CONDITIONS)},
    condition_value TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending' {_check_in('status', BEAT_STATUSES)},
    completed_at_turn INTEGER
);
CREATE UNIQUE INDEX one_active_beat ON story_beats(status) WHERE status = 'active';

CREATE TABLE npc_notes (
    character_id INTEGER PRIMARY KEY REFERENCES characters(id) ON DELETE CASCADE,
    notes TEXT NOT NULL,
    updated_turn INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE game_state (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    current_turn INTEGER NOT NULL DEFAULT 0,
    current_beat_id INTEGER REFERENCES story_beats(id) ON DELETE SET NULL,
    turns_since_beat_progress INTEGER NOT NULL DEFAULT 0,
    pending_options TEXT,                       -- JSON list, NULL when no menu is showing
    last_played_at TEXT
);
"""


def _connect(path: Path) -> sqlite3.Connection:
    # isolation_level=None -> we control transactions explicitly.
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _loads_list(text: Optional[str]) -> List[Any]:
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


# --- row -> dataclass conversion -------------------------------------------

def _campaign(row: sqlite3.Row) -> CampaignInfo:
    data = dict(row)
    data.pop("id")
    return CampaignInfo(**data)


def _location(row: sqlite3.Row) -> Location:
    return Location(**dict(row))


def _character(row: sqlite3.Row) -> Character:
    data = dict(row)
    data["is_persistent"] = bool(data["is_persistent"])
    return Character(**data)


def _message(row: sqlite3.Row) -> Message:
    return Message(**dict(row))


def _summary(row: sqlite3.Row) -> Summary:
    data = dict(row)
    for key in ("key_npcs", "open_threads", "promises_debts"):
        data[key] = _loads_list(data[key])
    return Summary(**data)


def _beat(row: sqlite3.Row) -> StoryBeat:
    return StoryBeat(**dict(row))


def _state(row: sqlite3.Row) -> GameState:
    data = dict(row)
    data.pop("id")
    raw = data["pending_options"]
    data["pending_options"] = _loads_list(raw) if raw else None
    return GameState(**data)


class GameDB:
    """Access layer for one campaign's database."""

    def __init__(self, conn: sqlite3.Connection, slug: str, path: Path):
        self.conn = conn
        self.slug = slug
        self.path = path
        self._tx_depth = 0

    # ------------------------------------------------------------------
    # Opening, creating, listing
    # ------------------------------------------------------------------

    @staticmethod
    def _campaigns_dir(campaigns_dir: Optional[Path]) -> Path:
        return Path(campaigns_dir) if campaigns_dir else get_config().paths.campaigns_dir

    @staticmethod
    def validate_slug(slug: str) -> str:
        if not _SLUG_RE.match(slug or ""):
            raise ValueError(
                f"Invalid campaign slug {slug!r}: use lowercase letters, digits and underscores only."
            )
        return slug

    @classmethod
    def db_path(cls, slug: str, campaigns_dir: Optional[Path] = None) -> Path:
        cls.validate_slug(slug)
        return cls._campaigns_dir(campaigns_dir) / slug / DB_FILENAME

    @classmethod
    def exists(cls, slug: str, campaigns_dir: Optional[Path] = None) -> bool:
        return cls.db_path(slug, campaigns_dir).exists()

    @classmethod
    def unique_slug(cls, base: str, campaigns_dir: Optional[Path] = None) -> str:
        """Return `base`, or `base_2`, `base_3`... if that slug is already taken."""
        base = slugify(base)
        candidate, n = base, 2
        while cls.exists(candidate, campaigns_dir):
            candidate = f"{base}_{n}"
            n += 1
        return candidate

    @classmethod
    def list_campaigns(cls, campaigns_dir: Optional[Path] = None) -> List[str]:
        """Slugs of every saved campaign, most recently modified first."""
        root = cls._campaigns_dir(campaigns_dir)
        if not root.exists():
            return []
        found = [
            (p.stat().st_mtime, p.parent.name)
            for p in root.glob(f"*/{DB_FILENAME}")
            if _SLUG_RE.match(p.parent.name)
        ]
        return [slug for _, slug in sorted(found, reverse=True)]

    @classmethod
    def create(cls, slug: str, campaigns_dir: Optional[Path] = None) -> "GameDB":
        """Create a new, empty campaign database."""
        path = cls.db_path(slug, campaigns_dir)
        if path.exists():
            raise CampaignExistsError(f"Campaign '{slug}' already exists at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        conn = _connect(path)
        try:
            conn.executescript(SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.execute("INSERT INTO game_state (id) VALUES (1)")
        except Exception:
            conn.close()
            path.unlink(missing_ok=True)
            raise
        return cls(conn, slug, path)

    @classmethod
    def open(cls, slug: str, campaigns_dir: Optional[Path] = None) -> "GameDB":
        """Open an existing campaign database."""
        path = cls.db_path(slug, campaigns_dir)
        if not path.exists():
            raise FileNotFoundError(f"No campaign '{slug}' at {path}")
        conn = _connect(path)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            conn.close()
            raise SchemaVersionError(
                f"Campaign '{slug}' uses save format v{version}, but this version of "
                f"Aether needs v{SCHEMA_VERSION}."
            )
        return cls(conn, slug, path)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "GameDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """All-or-nothing block. Nested use is safe (inner blocks use savepoints)."""
        depth = self._tx_depth
        if depth == 0:
            self.conn.execute("BEGIN IMMEDIATE")
        else:
            self.conn.execute(f"SAVEPOINT sp_{depth}")
        self._tx_depth += 1
        try:
            yield
        except BaseException:
            self._tx_depth -= 1
            if depth == 0:
                self.conn.execute("ROLLBACK")
            else:
                self.conn.execute(f"ROLLBACK TO sp_{depth}")
                self.conn.execute(f"RELEASE sp_{depth}")
            raise
        else:
            self._tx_depth -= 1
            if depth == 0:
                self.conn.execute("COMMIT")
            else:
                self.conn.execute(f"RELEASE sp_{depth}")

    # ------------------------------------------------------------------
    # Campaign
    # ------------------------------------------------------------------

    def set_campaign(self, info: CampaignInfo) -> None:
        """Insert or replace the single campaign row."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO campaign (
                id, name, slug, theme_vibe, starting_patron, setting_description,
                primary_threat, starting_quest_hook, storyline_beginning,
                storyline_middle, storyline_end, victory_condition,
                starting_location_id, embedding_model
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                info.name, info.slug, info.theme_vibe, info.starting_patron,
                info.setting_description, info.primary_threat, info.starting_quest_hook,
                info.storyline_beginning, info.storyline_middle, info.storyline_end,
                info.victory_condition, info.starting_location_id, info.embedding_model,
            ),
        )

    def get_campaign(self) -> Optional[CampaignInfo]:
        row = self.conn.execute("SELECT * FROM campaign WHERE id = 1").fetchone()
        return _campaign(row) if row else None

    def set_embedding_model(self, model_name: str) -> None:
        self.conn.execute("UPDATE campaign SET embedding_model = ? WHERE id = 1", (model_name,))

    # ------------------------------------------------------------------
    # Locations
    # ------------------------------------------------------------------

    def add_location(self, location_id: str, name: str, brief_concept: str) -> None:
        self.conn.execute(
            "INSERT INTO locations (id, name, brief_concept) VALUES (?, ?, ?)",
            (location_id, name, brief_concept),
        )

    def add_connection(self, from_id: str, to_id: str) -> None:
        """Link two locations. Connections work in both directions."""
        if from_id == to_id:
            raise ValueError(f"A location cannot connect to itself ('{from_id}')")
        self.conn.execute(
            "INSERT OR IGNORE INTO location_connections (from_location_id, to_location_id) VALUES (?, ?)",
            (from_id, to_id),
        )

    def get_location(self, location_id: str) -> Optional[Location]:
        row = self.conn.execute("SELECT * FROM locations WHERE id = ?", (location_id,)).fetchone()
        return _location(row) if row else None

    def list_locations(self) -> List[Location]:
        rows = self.conn.execute("SELECT * FROM locations ORDER BY rowid").fetchall()
        return [_location(r) for r in rows]

    def set_location_description(self, location_id: str, description: str) -> None:
        cur = self.conn.execute(
            "UPDATE locations SET description = ? WHERE id = ?", (description, location_id)
        )
        if cur.rowcount == 0:
            raise LookupError(f"Unknown location '{location_id}'")

    def get_connected_locations(self, location_id: str) -> List[Location]:
        rows = self.conn.execute(
            """
            SELECT * FROM locations WHERE id IN (
                SELECT to_location_id FROM location_connections WHERE from_location_id = ?
                UNION
                SELECT from_location_id FROM location_connections WHERE to_location_id = ?
            ) ORDER BY rowid
            """,
            (location_id, location_id),
        ).fetchall()
        return [_location(r) for r in rows]

    # ------------------------------------------------------------------
    # Characters and inventory
    # ------------------------------------------------------------------

    def add_character(self, c: Character) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO characters (
                name, type, location_id, race, char_class,
                strength, dexterity, constitution, intelligence, wisdom, charisma,
                hp, max_hp, status, is_persistent, template_id, backstory
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                c.name, c.type, c.location_id, c.race, c.char_class,
                c.strength, c.dexterity, c.constitution, c.intelligence, c.wisdom, c.charisma,
                c.hp, c.max_hp, c.status, int(c.is_persistent), c.template_id, c.backstory,
            ),
        )
        return cur.lastrowid

    def get_character(self, character_id: int) -> Optional[Character]:
        row = self.conn.execute("SELECT * FROM characters WHERE id = ?", (character_id,)).fetchone()
        return _character(row) if row else None

    def get_player(self) -> Optional[Character]:
        row = self.conn.execute("SELECT * FROM characters WHERE type = 'player'").fetchone()
        return _character(row) if row else None

    def has_player(self) -> bool:
        return self.get_player() is not None

    def list_characters_at(self, location_id: str, include_inactive: bool = False) -> List[Character]:
        """NPCs at a location (never the player). Dead or departed ones are hidden by default."""
        sql = "SELECT * FROM characters WHERE location_id = ? AND type <> 'player'"
        if not include_inactive:
            sql += " AND status = 'alive'"
        rows = self.conn.execute(sql + " ORDER BY id", (location_id,)).fetchall()
        return [_character(r) for r in rows]

    def move_character(self, character_id: int, location_id: str) -> None:
        cur = self.conn.execute(
            "UPDATE characters SET location_id = ? WHERE id = ?", (location_id, character_id)
        )
        if cur.rowcount == 0:
            raise LookupError(f"Unknown character id {character_id}")

    def change_hp(self, character_id: int, delta: int) -> int:
        """Add `delta` (negative for damage) to HP, clamped to 0..max_hp. Returns the new HP."""
        with self.transaction():
            row = self.conn.execute(
                "SELECT hp, max_hp FROM characters WHERE id = ?", (character_id,)
            ).fetchone()
            if row is None:
                raise LookupError(f"Unknown character id {character_id}")
            new_hp = max(0, min(row["max_hp"], row["hp"] + delta))
            self.conn.execute("UPDATE characters SET hp = ? WHERE id = ?", (new_hp, character_id))
            return new_hp

    def set_character_status(self, character_id: int, status: str) -> None:
        cur = self.conn.execute(
            "UPDATE characters SET status = ? WHERE id = ?", (status, character_id)
        )
        if cur.rowcount == 0:
            raise LookupError(f"Unknown character id {character_id}")

    def purge_ephemeral_npcs(self, location_id: str) -> int:
        """Delete non-persistent NPCs at a location (used when the player leaves it)."""
        cur = self.conn.execute(
            "DELETE FROM characters WHERE location_id = ? AND type <> 'player' AND is_persistent = 0",
            (location_id,),
        )
        return cur.rowcount

    def add_item(self, character_id: int, item_name: str, quantity: int = 1) -> None:
        item_name = item_name.strip()
        if not item_name:
            raise ValueError("Item name must not be empty")
        self.conn.execute(
            """
            INSERT INTO inventory (character_id, item_name, quantity) VALUES (?, ?, ?)
            ON CONFLICT(character_id, item_name) DO UPDATE SET quantity = quantity + excluded.quantity
            """,
            (character_id, item_name, quantity),
        )

    def remove_item(self, character_id: int, item_name: str, quantity: int = 1) -> bool:
        """Remove items. Returns False (and changes nothing) if the character doesn't have enough."""
        with self.transaction():
            row = self.conn.execute(
                "SELECT quantity FROM inventory WHERE character_id = ? AND item_name = ?",
                (character_id, item_name.strip()),
            ).fetchone()
            if row is None or row["quantity"] < quantity:
                return False
            if row["quantity"] == quantity:
                self.conn.execute(
                    "DELETE FROM inventory WHERE character_id = ? AND item_name = ?",
                    (character_id, item_name.strip()),
                )
            else:
                self.conn.execute(
                    "UPDATE inventory SET quantity = quantity - ? WHERE character_id = ? AND item_name = ?",
                    (quantity, character_id, item_name.strip()),
                )
            return True

    def has_item(self, character_id: int, item_name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM inventory WHERE character_id = ? AND item_name = ?",
            (character_id, item_name.strip()),
        ).fetchone()
        return row is not None

    def get_inventory(self, character_id: int) -> List[Tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT item_name, quantity FROM inventory WHERE character_id = ? ORDER BY rowid",
            (character_id,),
        ).fetchall()
        return [(r["item_name"], r["quantity"]) for r in rows]

    # ------------------------------------------------------------------
    # NPC notes
    # ------------------------------------------------------------------

    def set_npc_notes(self, character_id: int, notes: str, turn: Optional[int] = None) -> None:
        if turn is None:
            turn = self.get_state().current_turn
        self.conn.execute(
            """
            INSERT INTO npc_notes (character_id, notes, updated_turn) VALUES (?, ?, ?)
            ON CONFLICT(character_id) DO UPDATE SET notes = excluded.notes, updated_turn = excluded.updated_turn
            """,
            (character_id, notes, turn),
        )

    def get_npc_notes(self, character_id: int) -> Optional[str]:
        row = self.conn.execute(
            "SELECT notes FROM npc_notes WHERE character_id = ?", (character_id,)
        ).fetchone()
        return row["notes"] if row else None

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str, turn_number: Optional[int] = None) -> int:
        """Append to the transcript and return the new message id."""
        with self.transaction():
            if turn_number is None:
                turn_number = self.get_state().current_turn
            cur = self.conn.execute(
                "INSERT INTO messages (turn_number, role, content) VALUES (?, ?, ?)",
                (turn_number, role, content),
            )
            self.conn.execute("UPDATE game_state SET last_played_at = CURRENT_TIMESTAMP WHERE id = 1")
            return cur.lastrowid

    def get_messages(self, after_id: int = 0, up_to_id: Optional[int] = None) -> List[Message]:
        """Messages with id > after_id (and <= up_to_id if given), oldest first."""
        sql, params = "SELECT * FROM messages WHERE id > ?", [after_id]
        if up_to_id is not None:
            sql += " AND id <= ?"
            params.append(up_to_id)
        rows = self.conn.execute(sql + " ORDER BY id", params).fetchall()
        return [_message(r) for r in rows]

    def get_recent_messages(self, n: int, include_system: bool = True) -> List[Message]:
        """The last `n` messages, oldest first."""
        sql = "SELECT * FROM messages"
        if not include_system:
            sql += " WHERE role <> 'system'"
        rows = self.conn.execute(sql + " ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [_message(r) for r in reversed(rows)]

    def get_last_message(self) -> Optional[Message]:
        recent = self.get_recent_messages(1)
        return recent[0] if recent else None

    def count_messages(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    def needs_dm_reply(self) -> bool:
        """True if the player has acted but the DM never answered (e.g. a crash mid-turn)."""
        row = self.conn.execute(
            "SELECT role FROM messages WHERE role <> 'system' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row is not None and row["role"] == "player"

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def add_summary(self, summary: Summary) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO summaries (covers_through_message_id, narrative, key_npcs, open_threads, promises_debts)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                summary.covers_through_message_id,
                summary.narrative,
                json.dumps(summary.key_npcs),
                json.dumps(summary.open_threads),
                json.dumps(summary.promises_debts),
            ),
        )
        return cur.lastrowid

    def get_latest_summary(self) -> Optional[Summary]:
        row = self.conn.execute("SELECT * FROM summaries ORDER BY id DESC LIMIT 1").fetchone()
        return _summary(row) if row else None

    def get_uncompacted_messages(self) -> List[Message]:
        """Messages not yet covered by the latest summary (the raw 'tail')."""
        summary = self.get_latest_summary()
        return self.get_messages(after_id=summary.covers_through_message_id if summary else 0)

    # ------------------------------------------------------------------
    # Story beats
    # ------------------------------------------------------------------

    def add_beat(self, beat: StoryBeat) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO story_beats (position, title, objective, condition_type, condition_value)
            VALUES (?, ?, ?, ?, ?)
            """,
            (beat.position, beat.title, beat.objective, beat.condition_type, beat.condition_value),
        )
        return cur.lastrowid

    def get_beats(self) -> List[StoryBeat]:
        rows = self.conn.execute("SELECT * FROM story_beats ORDER BY position").fetchall()
        return [_beat(r) for r in rows]

    def get_current_beat(self) -> Optional[StoryBeat]:
        row = self.conn.execute("SELECT * FROM story_beats WHERE status = 'active'").fetchone()
        return _beat(row) if row else None

    def activate_first_beat(self) -> Optional[StoryBeat]:
        """Start the story: make the first pending beat active (no-op if one already is)."""
        with self.transaction():
            current = self.get_current_beat()
            if current:
                return current
            row = self.conn.execute(
                "SELECT * FROM story_beats WHERE status = 'pending' ORDER BY position LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            self.conn.execute("UPDATE story_beats SET status = 'active' WHERE id = ?", (row["id"],))
            self.conn.execute("UPDATE game_state SET current_beat_id = ? WHERE id = 1", (row["id"],))
            return self.get_current_beat()

    def complete_current_beat(self) -> Optional[StoryBeat]:
        """Mark the active beat done, activate the next one and reset the drift clock.

        Returns the newly active beat, or None if that was the last one.
        """
        with self.transaction():
            current = self.get_current_beat()
            if current is None:
                raise LookupError("There is no active beat to complete")
            turn = self.get_state().current_turn
            self.conn.execute(
                "UPDATE story_beats SET status = 'done', completed_at_turn = ? WHERE id = ?",
                (turn, current.id),
            )
            nxt = self.conn.execute(
                "SELECT id FROM story_beats WHERE status = 'pending' ORDER BY position LIMIT 1"
            ).fetchone()
            if nxt:
                self.conn.execute("UPDATE story_beats SET status = 'active' WHERE id = ?", (nxt["id"],))
            self.conn.execute(
                "UPDATE game_state SET current_beat_id = ?, turns_since_beat_progress = 0 WHERE id = 1",
                (nxt["id"] if nxt else None,),
            )
            return self.get_current_beat()

    def is_campaign_complete(self) -> bool:
        beats = self.get_beats()
        return bool(beats) and all(b.status == "done" for b in beats)

    # ------------------------------------------------------------------
    # Game state
    # ------------------------------------------------------------------

    def get_state(self) -> GameState:
        row = self.conn.execute("SELECT * FROM game_state WHERE id = 1").fetchone()
        return _state(row)

    def advance_turn(self) -> int:
        """Start a new player turn: bumps the turn counter and the drift clock."""
        with self.transaction():
            self.conn.execute(
                """
                UPDATE game_state
                SET current_turn = current_turn + 1,
                    turns_since_beat_progress = turns_since_beat_progress + 1
                WHERE id = 1
                """
            )
            return self.get_state().current_turn

    def set_pending_options(self, options: Optional[List[Dict[str, Any]]]) -> None:
        """Save the menu currently shown to the player (None clears it)."""
        payload = json.dumps(options) if options is not None else None
        self.conn.execute("UPDATE game_state SET pending_options = ? WHERE id = 1", (payload,))
