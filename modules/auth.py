import json
import os
from typing import Dict
from fastapi import HTTPException
import secrets

class Auth:
    def __init__(self, user_dir: str = "data/users"):
        self.user_dir = user_dir
        os.makedirs(user_dir, exist_ok=True)

    def register(self, user_id: str, password: str) -> Dict:
        user_file = os.path.join(self.user_dir, f"{user_id}.json")
        if os.path.exists(user_file):
            raise HTTPException(status_code=400, detail="User already exists")

        token = secrets.token_hex(16)
        user_data = {
            "user_id": user_id,
            "password": password,
            "token": token
        }
        with open(user_file, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2)
        return {"user_id": user_id, "token": token}

    def login(self, user_id: str, password: str) -> Dict:
        user_file = os.path.join(self.user_dir, f"{user_id}.json")
        if not os.path.exists(user_file):
            raise HTTPException(status_code=404, detail="User not found")

        with open(user_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)

        if user_data["password"] != password:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return {"user_id": user_id, "token": user_data["token"]}

    def verify_token(self, user_id: str, token: str) -> bool:
        user_file = os.path.join(self.user_dir, f"{user_id}.json")
        if not os.path.exists(user_file):
            return False

        with open(user_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)

        return user_data["token"] == token
