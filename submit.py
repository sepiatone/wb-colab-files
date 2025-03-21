import requests
import os
from dotenv import load_dotenv
from hashlib import md5


SUPABASE_URL = "https://udnmlcykctnahtnzmmmx.supabase.co"

def generate_hash(mid, eid, iid, uid):
    return md5("-".join(map(str, [mid, eid, iid, uid])).encode()).hexdigest()

def test_submit(iid, is_checked=True, mid=4, eid=1):
    load_dotenv()
    jwt_token = os.getenv("JWT")
    uid = os.getenv("UID")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not jwt_token or not uid:
        print("❌ Please login first.")
        return
    else:
        auth_headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        }

        payload = {
            "id": generate_hash(mid, eid, iid, uid),
            "module_id": mid,
            "exercise_id": eid,
            "item_id": iid,
            "user_id": uid,
            "is_checked": is_checked,
        }

        url = f"{SUPABASE_URL}/rest/v1/item_completion"
        response = requests.post(url, json=payload, headers=auth_headers)
        
        if response.status_code not in [200, 201]:
            print("Insert failed:", response.json())