#!/usr/bin/env python3
"""
Test script to verify PettingZoo Atari setup for NeatRL
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_pettingzoo_imports():
    """Test that all required PettingZoo modules can be imported"""
    print("Testing PettingZoo imports...")
    
    try:
        import pettingzoo.atari.pong_v3
        print("✓ pettingzoo.atari.pong_v3 imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pettingzoo.atari.pong_v3: {e}")
        return False
    
    try:
        import pettingzoo.atari.boxing_v2
        print("✓ pettingzoo.atari.boxing_v2 imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pettingzoo.atari.boxing_v2: {e}")
        return False
    
    try:
        import supersuit as ss
        print("✓ supersuit imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import supersuit: {e}")
        return False
    
    return True

def test_environment_creation():
    """Test that environments can be created and reset"""
    print("\nTesting environment creation...")
    
    try:
        import pettingzoo.atari.pong_v3
        import supersuit as ss
        
        # Create environment with wrappers
        env = pettingzoo.atari.pong_v3.env(render_mode='rgb_array')
        env = ss.max_observation_v0(env, 2)
        env = ss.frame_skip_v0(env, 4)
        env = ss.color_reduction_v0(env, mode='B')
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)
        env = ss.agent_indicator_v0(env, type_only=False)
        
        # Reset environment
        env.reset()
        print("✓ Pong v3 environment created and reset successfully")
        print(f"  - Possible agents: {env.possible_agents}")
        print(f"  - Action space: {env.action_space(env.possible_agents[0])}")
        print(f"  - Observation space: {env.observation_space(env.possible_agents[0])}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False

def test_environment_list():
    """Test that all supported environments can be imported"""
    print("\nTesting supported environments...")
    
    supported_envs = [
        "pong_v3", "boxing_v2", "tennis_v3", "ice_hockey_v2", "double_dunk_v3", 
        "wizard_of_wor_v3", "space_invaders_v2", "basketball_pong_v3", "volleyball_pong_v3",
        "foozpong_v3", "quadrapong_v4", "joust_v3", "mario_bros_v3", "maze_craze_v3",
        "othello_v3", "video_checkers_v4", "warlords_v3", "combat_plane_v2", "combat_tank_v2",
        "flag_capture_v2", "surround_v2", "space_war_v2", "entombed_competitive_v3", "entombed_cooperative_v3"
    ]
    
    success_count = 0
    for env_name in supported_envs:
        try:
            # Dynamic import
            module = __import__(f"pettingzoo.atari.{env_name}", fromlist=['env'])
            env = module.env(render_mode='rgb_array')
            env.reset()
            env.close()
            print(f"✓ {env_name} imported and reset successfully")
            success_count += 1
        except Exception as e:
            print(f"✗ {env_name} failed: {e}")
    
    print(f"\nSuccessfully tested {success_count}/{len(supported_envs)} environments")
    return success_count == len(supported_envs)

def test_ale_play_script():
    """Test that the ale_play_script.py can be imported"""
    print("\nTesting ale_play_script.py import...")
    
    try:
        from ale_play_script import play_pettingzoo_game, get_keyboard_mappings, get_environment_info
        print("✓ ale_play_script.py imported successfully")
        print("✓ play_pettingzoo_game function available")
        print("✓ get_keyboard_mappings function available")
        print("✓ get_environment_info function available")
        return True
    except ImportError as e:
        print(f"✗ Failed to import ale_play_script.py: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("NeatRL PettingZoo Atari Setup Test")
    print("=" * 60)
    
    tests = [
        ("PettingZoo Imports", test_pettingzoo_imports),
        ("Environment Creation", test_environment_creation),
        ("Environment List", test_environment_list),
        ("ALE Play Script", test_ale_play_script),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PettingZoo Atari setup is ready for NeatRL.")
        return 0
    else:
        print("❌ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
