import sys
import os

# Import our custom components
from campaign_creator import CampaignCreator
from character_creator import CharacterCreator
from orchestrator import GameOrchestrator


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    clear_screen()
    print("=======================================================")
    print("                WELCOME TO AETHER D&D                  ")
    print("=======================================================")
    print("A Stateful LLM Orchestrator for Locally Hosted Games\n")
    print("Choose an option:")
    print("1. Start a New Campaign (Brainstorm & Build)")
    print("2. Exit")
    
    choice = input("\nSelect (1-2): ").strip()
    if choice != "1":
        print("Exiting Aether. Safe travels, adventurer!")
        sys.exit(0)

    # -------------------------------------------------------------
    # PHASE 1: Campaign Creation
    # -------------------------------------------------------------
    clear_screen()
    campaign_engine = CampaignCreator()
    campaign_engine.run_brainstorm_loop()
    
    # Extract the generated campaign slug
    # We dynamically slugify the name the player just generated
    campaign_name = campaign_engine.chat_history[-2]["content"]  # Grab the last assistant output or name
    # Alternatively, we can find the written campaign file inside data/campaigns/
    campaigns_dir = "data/campaigns"
    files = [f for f in os.listdir(campaigns_dir) if f.endswith("_info.md")]
    if not files:
        print("Error: No campaign file was generated. Exiting.")
        sys.exit(1)
        
    # Sort by modification time to grab the most recently written campaign
    files.sort(key=lambda x: os.path.getmtime(os.path.join(campaigns_dir, x)), reverse=True)
    active_campaign_file = files[0]
    campaign_slug = active_campaign_file.replace("_info.md", "")

    print(f"\n[Master] Active Campaign Slug Resolved: {campaign_slug}")
    input("Press Enter to proceed to Character Creation...")

    # -------------------------------------------------------------
    # PHASE 2: Character Creation
    # -------------------------------------------------------------
    clear_screen()
    char_engine = CharacterCreator(campaign_slug=campaign_slug)
    char_engine.run_creation_loop()

    input("Press Enter to launch the main game loop...")

    # -------------------------------------------------------------
    # PHASE 3: The Main Game Loop
    # -------------------------------------------------------------
    clear_screen()
    game = GameOrchestrator(campaign_slug=campaign_slug, state="NARRATIVE_PLAY")
    
    print("\n=======================================================")
    print("                 AETHER GAME TERMINAL                  ")
    print("=======================================================")
    print(f"Campaign: {campaign_slug}")
    print("Type your actions or dialogue below. To exit, type 'exit'.")
    print("=======================================================\n")
    
    # Trigger an initial DM narrative introduction
    intro_narration = game.process_narrative_turn(
        "I open my eyes and look at my surroundings, ready to begin the adventure."
    )
    print(f"\nDungeon Master:\n{intro_narration}\n")

    while True:
        try:
            player_input = input("\nYou: ").strip()
            if not player_input:
                continue
                
            if player_input.lower() == "exit":
                print("Saving game slot and exiting. Safe travels!")
                break
                
            # If the player types "attack" or "combat", let's trigger combat mode!
            if "attack" in player_input.lower() or "combat" in player_input.lower():
                print("\n[Engine] Hostilities detected! Transitioning to Combat Mode...")
                game.initiate_combat()
                
                # Run combat loop until ended
                while game.state == "COMBAT_PLAY":
                    current_actor = game.combat_queue[game.current_turn_index]
                    if current_actor["type"] == "player":
                        p_turn_input = input(f"\n[Your Turn - {current_actor['name']}] Action: ").strip()
                        if p_turn_input.lower() == "exit":
                            break
                        combat_reply = game.process_combat_turn(player_action_text=p_turn_input)
                    else:
                        input(f"\n[NPC Turn - {current_actor['name']}] Press Enter to resolve AI turn...")
                        combat_reply = game.process_combat_turn()
                        
                    print(f"\nDungeon Master:\n{combat_reply}\n")
                continue

            # Standard narrative loop
            reply = game.process_narrative_turn(player_input)
            print(f"\nDungeon Master:\n{reply}\n")
            
        except KeyboardInterrupt:
            print("\nGame interrupted. Progress saved in save slot directory. Exiting.")
            break


if __name__ == "__main__":
    main()
