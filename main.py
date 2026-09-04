import sys
import os

from campaign_creator import CampaignCreator
from character_creator import CharacterCreator
from orchestrator import GameOrchestrator
from db import get_db_connection


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    clear_screen()
    print("=======================================================")
    print("                WELCOME TO AETHER D&D                  ")
    print("=======================================================")
    print("A Stateful LLM Orchestrator for Locally Hosted Games\n")
    print("1. Start a New Campaign")
    print("2. Continue an Existing Campaign (Load Save-Slot)")
    print("3. Exit")
    
    choice = input("\nSelect (1-3): ").strip()
    
    campaign_slug = ""
    
    if choice == "3":
        print("Exiting Aether. Safe travels, adventurer!")
        sys.exit(0)

    elif choice == "2":
        campaigns_dir = "data/campaigns"
        if not os.path.exists(campaigns_dir) or not os.listdir(campaigns_dir):
            print("\n[System] No campaigns found on disk. Please start a new campaign first.")
            input("Press Enter to return...")
            return main()
            
        files = [f.replace("_info.md", "") for f in os.listdir(campaigns_dir) if f.endswith("_info.md")]
        
        print("\n=== AVAILABLE SAVE SLOTS ===")
        for idx, slug in enumerate(files):
            print(f"{idx+1}. {slug}")
        print("=============================")
        
        try:
            slot_choice = int(input(f"Select campaign (1-{len(files)}): ").strip())
            campaign_slug = files[slot_choice - 1]
        except (ValueError, IndexError):
            print("Invalid selection.")
            input("Press Enter to try again...")
            return main()
            
    else:
        clear_screen()
        campaign_engine = CampaignCreator()
        campaign_engine.run_brainstorm_loop()
        
        campaigns_dir = "data/campaigns"
        files = [f for f in os.listdir(campaigns_dir) if f.endswith("_info.md")]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(campaigns_dir, x)), reverse=True)
        campaign_slug = files[0].replace("_info.md", "")

    print(f"\n[Master] Loading Save Slot: {campaign_slug}")
    
    conn = get_db_connection(campaign_slug)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM characters WHERE type = 'player'")
    has_char = cursor.fetchone()[0] > 0
    conn.close()

    if not has_char:
        print("\n[System] No player character registered in this campaign database.")
        input("Press Enter to run Character Creation...")
        clear_screen()
        char_engine = CharacterCreator(campaign_slug=campaign_slug)
        char_engine.run_creation_loop()

    input("Press Enter to launch the main game loop...")

    clear_screen()
    game = GameOrchestrator(campaign_slug=campaign_slug, state="NARRATIVE_PLAY")
    
    print("\n=======================================================")
    print("                 AETHER GAME TERMINAL                  ")
    print("=======================================================")
    print(f"Campaign: {campaign_slug}")
    print("Type your actions or dialogue below. To exit, type 'exit'.")
    print("=======================================================\n")
    
    print("\nDungeon Master:")
    for chunk in game.process_narrative_turn(
        "I open my eyes and look at my surroundings, ready to begin the adventure."
    ):
        print(chunk, end="", flush=True)
    print("\n")

    while True:
        try:
            player_input = input("\nYou: ").strip()
            if not player_input:
                continue
                
            if player_input.lower() == "exit":
                print("Saving game slot and exiting. Safe travels!")
                break

            if player_input.startswith("/"):
                command = player_input.lower().strip()
                if command == "/stats" or command == "/inventory":
                    print("\n[System] Character sheet tools are being migrated to the new flat-text system.")
                    continue
                else:
                    print("\n[System] Unknown slash command.")
                    continue

            print("\nDungeon Master:")
            for chunk in game.process_narrative_turn(player_input):
                print(chunk, end="", flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            print("\nGame interrupted. Progress saved in save slot directory. Exiting.")
            break

if __name__ == "__main__":
    main()
