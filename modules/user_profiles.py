import os
import json
from pydantic import BaseModel
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class UserProfile(BaseModel):
    user_id: str
    name: str
    goals: List[str] = []
    chakra_focus: Optional[str] = None
    healing_preferences: List[str] = []
    preferred_tone: Optional[str] = "G"

def create_user_profile(user_id: str, name: str, goals: List[str] = None, chakra_focus: str = None,
                       healing_preferences: List[str] = None, preferred_tone: str = "G"):
    """Create or update a user profile in data/user_profiles/{user_id}_profile.json."""
    profile = UserProfile(
        user_id=user_id,
        name=name,
        goals=goals or [],
        chakra_focus=chakra_focus,
        healing_preferences=healing_preferences or [],
        preferred_tone=preferred_tone
    )

    os.makedirs("data/user_profiles", exist_ok=True)
    profile_file = f"data/user_profiles/{user_id}_profile.json"

    try:
        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(profile.dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Created profile for user {user_id}")
        return {"status": "success", "message": f"Profile created for {user_id}"}
    except Exception as e:
        logger.error(f"Failed to create profile for {user_id}: {str(e)}")
        return {"status": "error", "message": f"Failed to create profile: {str(e)}"}

def get_user_profile(user_id: str):
    """Fetch a user profile."""
    profile_file = f"data/user_profiles/{user_id}_profile.json"
    try:
        with open(profile_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Fetched profile for user {user_id}")
            return UserProfile(**data)
    except:
        logger.warning(f"No profile found for user {user_id}")
        return None