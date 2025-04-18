import json
import os
from modules.user_profiles import get_user_profile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def suggest_mantra(emotion: str, goal: str, chakra: str = None, user_id: str = None, astrological_context: str = None):
    """Suggest a mantra, factoring in user profile preferences."""
    try:
        with open("data/vedic_knowledge/vedic_knowledge.json", "r", encoding="utf-8") as f:
            vedas = json.load(f)
        logger.info("Loaded vedic_knowledge.json successfully")
    except Exception as e:
        logger.error(f"Failed to load vedic_knowledge.json: {str(e)}")
        return {
            "mantra": "Om Namah Shivaya",
            "frequency": 432.0,
            "tone": "G",
            "repetitions": 108,
            "text": "Default mantra",
            "translation": "Universal chant"
        }

    user_chakra = chakra
    user_tone = "G"
    user_emotion = emotion.lower() if emotion else ""
    user_goal = goal.lower() if goal else ""
    if user_id:
        profile = get_user_profile(user_id)
        if profile:
            user_chakra = user_chakra or profile.chakra_focus
            user_tone = profile.preferred_tone
            user_emotion = user_emotion or (profile.goals[0].lower() if profile.goals else "")
            user_goal = user_goal or (profile.goals[1].lower() if len(profile.goals) > 1 else "")
            logger.info(f"User profile for {user_id}: chakra={user_chakra}, tone={user_tone}, emotion={user_emotion}, goal={user_goal}")

    best_match = None
    best_score = -1
    for text in vedas["texts"]:
        for section in text["sections"]:
            for verse in section["verses"]:
                verse_emotions = [e.lower() for e in verse.get("emotion", [])]
                verse_goals = [g.lower() for g in verse.get("goal", [])]
                verse_chakra = verse.get("chakra", "").lower()
                
                score = 0
                if user_emotion and any(user_emotion in ve for ve in verse_emotions):
                    score += 2
                elif user_emotion and any(user_emotion[:3] in ve for ve in verse_emotions):  # Partial match
                    score += 1
                if user_goal and any(user_goal in vg for vg in verse_goals):
                    score += 2
                elif user_goal and any(user_goal[:3] in vg for vg in verse_goals):  # Partial match
                    score += 1
                if user_chakra and user_chakra.lower() == verse_chakra:
                    score += 3
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        "mantra": verse["mantra"] or "Om Namah Shivaya",
                        "frequency": verse["frequency"],
                        "tone": user_tone if user_tone else verse["tone"],
                        "repetitions": verse["repetitions"],
                        "text": verse["text"],
                        "translation": verse["translation"]
                    }
                    logger.info(f"Found match with score {score}: {best_match['mantra']}")

    if best_match:
        logger.info(f"Returning best match: {best_match['mantra']}")
        return best_match
    
    logger.warning("No match found, returning default mantra")
    return {
        "mantra": "Om Namah Shivaya",
        "frequency": 432.0,
        "tone": user_tone,
        "repetitions": 108,
        "text": "Default mantra",
        "translation": "Universal chant"
    }