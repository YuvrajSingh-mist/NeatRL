#!/usr/bin/env python3
"""
Test script for NeatRL system
Tests RL Hub and RL Arena functionality
"""

import requests
import json
import time
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
TEST_MODEL_PATH = "example_agents/dqn_arena_model.py"

def test_rl_hub():
    """Test RL Hub functionality"""
    print("🧪 Testing RL Hub...")
    
    # Test model upload
    try:
        with open(TEST_MODEL_PATH, 'rb') as f:
            files = {'file': ('dqn_arena_model.py', f, 'text/plain')}
            data = {
                'model_name': 'Test DQN Model',
                'description': 'A test DQN model for NeatRL Arena',
                'algorithm': 'DQN',
                'env_id': 'CartPole-v1',
                'user_id': 'test_user',
                'is_public': True
            }
            
            response = requests.post(f"{API_URL}/api/rlhub/upload", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                model_id = result['id']
                print(f"✅ Model uploaded successfully: {model_id}")
                return model_id
            else:
                print(f"❌ Model upload failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        return None

def test_list_models():
    """Test listing models"""
    print("📋 Testing model listing...")
    
    try:
        response = requests.get(f"{API_URL}/api/rlhub/models")
        
        if response.status_code == 200:
            result = response.json()
            models = result.get('models', [])
            print(f"✅ Found {len(models)} models")
            return models
        else:
            print(f"❌ Failed to list models: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []

def test_create_battle(player_model_id):
    """Test creating a battle"""
    print("⚔️ Testing battle creation...")
    
    try:
        data = {
            'player_model_id': player_model_id,
            'env_id': 'CartPole-v1'
        }
        
        response = requests.post(f"{API_URL}/api/rlarena/battle", json=data)
        
        if response.status_code == 200:
            result = response.json()
            battle_id = result['battle_id']
            print(f"✅ Battle created successfully: {battle_id}")
            return battle_id
        else:
            print(f"❌ Battle creation failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating battle: {e}")
        return None

def test_battle_status(battle_id):
    """Test checking battle status"""
    print(f"🔍 Testing battle status for {battle_id}...")
    
    try:
        response = requests.get(f"{API_URL}/api/rlarena/battles/{battle_id}")
        
        if response.status_code == 200:
            result = response.json()
            status = result['status']
            print(f"✅ Battle status: {status}")
            return status
        else:
            print(f"❌ Failed to get battle status: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error checking battle status: {e}")
        return None

def test_rankings():
    """Test arena rankings"""
    print("🏆 Testing arena rankings...")
    
    try:
        response = requests.get(f"{API_URL}/api/rlarena/rankings")
        
        if response.status_code == 200:
            result = response.json()
            rankings = result.get('rankings', [])
            print(f"✅ Found {len(rankings)} ranked models")
            return rankings
        else:
            print(f"❌ Failed to get rankings: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Error getting rankings: {e}")
        return []

def test_health():
    """Test API health"""
    print("🏥 Testing API health...")
    
    try:
        response = requests.get(f"{API_URL}/health")
        
        if response.status_code == 200:
            result = response.json()
            status = result.get('status')
            print(f"✅ API health: {status}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking API health: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting NeatRL System Tests")
    print("=" * 50)
    
    # Test API health
    if not test_health():
        print("❌ API is not healthy. Please start the system first.")
        return
    
    print()
    
    # Test RL Hub
    model_id = test_rl_hub()
    if not model_id:
        print("❌ RL Hub test failed")
        return
    
    print()
    
    # Test listing models
    models = test_list_models()
    if not models:
        print("❌ Model listing test failed")
        return
    
    print()
    
    # Test creating battle
    battle_id = test_create_battle(model_id)
    if not battle_id:
        print("❌ Battle creation test failed")
        return
    
    print()
    
    # Test battle status
    status = test_battle_status(battle_id)
    if status is None:
        print("❌ Battle status test failed")
        return
    
    print()
    
    # Test rankings
    rankings = test_rankings()
    if rankings is None:
        print("❌ Rankings test failed")
        return
    
    print()
    print("🎉 All tests completed successfully!")
    print("=" * 50)
    print("NeatRL system is working correctly!")
    print()
    print("Next steps:")
    print("1. Open http://localhost:7860 to access the Gradio frontend")
    print("2. Upload models to the RL Hub")
    print("3. Create battles in the RL Arena")
    print("4. View rankings and results")

if __name__ == "__main__":
    main()
