import os
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

class MarkdownManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.directories = {
            "rules": self.base_dir / "rules",
            "lore": self.base_dir / "lore",
            "campaigns": self.base_dir / "campaigns",
            "locations": self.base_dir / "locations",
            "items": self.base_dir / "items",
            "actors": self.base_dir / "actors",
            "history": self.base_dir / "history",
            "chronicle": self.base_dir / "history" / "chronicle_logs"
        }
        self.setup_directories()

    def setup_directories(self) -> None:
        for path in self.directories.values():
            path.mkdir(parents=True, exist_ok=True)
        print(f"Filesystem directories initialized at: {self.base_dir.resolve()}")

    def _get_file_path(self, category: str, filename: str) -> Path:
        if category not in self.directories:
            raise ValueError(f"Unknown category '{category}'. Choose from {list(self.directories.keys())}")
        
        if not filename.endswith(".md"):
            filename += ".md"
            
        return self.directories[category] / filename

    def write_file(self, category: str, filename: str, metadata: Dict[str, Any], content: str) -> None:
        file_path = self._get_file_path(category, filename)
        
        yaml_header = yaml.dump(metadata, default_flow_style=False).strip()
        
        full_markdown = f"---\n{yaml_header}\n---\n\n{content.strip()}\n"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)
        print(f"Successfully wrote file: {file_path}")

    def read_file(self, category: str, filename: str) -> Tuple[Dict[str, Any], str]:
        file_path = self._get_file_path(category, filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        if raw_text.startswith("---"):
            parts = raw_text.split("---", 2)
            if len(parts) >= 3:
                try:
                    metadata = yaml.safe_load(parts[1]) or {}
                    content = parts[2].strip()
                    return metadata, content
                except yaml.YAMLError as e:
                    print(f"Error parsing YAML front matter in {file_path}: {e}")
                    
        return {}, raw_text.strip()

    def append_to_chronicle(self, session_number: int, log_entry: str) -> None:
        filename = f"session_{session_number:02d}.md"
        file_path = self._get_file_path("chronicle", filename)
        
        if not file_path.exists():
            metadata = {
                "session_id": session_number,
                "type": "campaign_chronicle",
                "status": "active"
            }
            self.write_file("chronicle", filename, metadata, f"# Session {session_number} Chronicle\n\n")

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{log_entry.strip()}\n")
        print(f"Appended entry to chronicle: {file_path}")

if __name__ == "__main__":
    manager = MarkdownManager()

    location_metadata = {
        "id": "whispering_woods_entry",
        "name": "Whispering Woods Entry",
        "danger_level": "low",
        "connected_locations": ["town_of_ostia", "deep_woods"]
    }
    location_body = """
# Entry to the Whispering Woods
The misty tree-line towers before you. Moss hangs heavily from ancient boughs, and a quiet, unnatural stillness settles over the path ahead. The air is damp, smelling faintly of pine and rotting foliage.
"""
    manager.write_file("locations", "whispering_woods_entry", location_metadata, location_body)

    actor_metadata = {
        "id": 12,
        "name": "Barnaby Ostia",
        "role": "merchant",
        "status": "alive",
        "alignment": "neutral"
    }
    actor_body = """
# Barnaby the Ostian Merchant
A slightly eccentric, short-statured gnome trader who wears a dusty patchwork coat. He carries his entire inventory on the back of an old mule. He treats trade transactions with transactional seriousness, always looking to maximize profit, but is fond of telling rumors for a silver piece.
"""
    manager.write_file("actors", "barnaby_merchant", actor_metadata, actor_body)

    try:
        meta, content = manager.read_file("actors", "barnaby_merchant")
        print("\n--- Verification Read ---")
        print(f"Parsed Name from Metadata: {meta.get('name')}")
        print(f"Parsed Status: {meta.get('status')}")
    except Exception as e:
        print(f"Read failed: {e}")
