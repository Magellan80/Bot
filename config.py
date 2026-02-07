import json
import os

SETTINGS_FILE = "settings.json"

# ============================================================
#   TELEGRAM BOT TOKEN (вставлен ровно как прислано)
# ============================================================

BOT_TOKEN = "8015271649:AAGaq86xSKBD1e8Av7lYCYfrNsXI_4bDsaQ"

# ID администратора — поставь свой Telegram user_id
ADMIN_ID = 1312559799

# Ключи Bybit (если нужны)
BYBIT_API_KEY = ""
BYBIT_API_SECRET = ""

# ============================================================
#   РЕЖИМЫ A/B/C
# ============================================================

MODES = {
    "A": {"name": "Aggressive"},
    "B": {"name": "Balanced"},
    "C": {"name": "Conservative"},
}

# ============================================================
#   ПАРАМЕТРЫ ПО УМОЛЧАНИЮ
# ============================================================

DEFAULT_MIN_SCORE = 45

# ============================================================
#   SETTINGS.JSON
# ============================================================

def load_settings() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_settings(data: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[config] Ошибка сохранения settings.json: {e}")

# ============================================================
#   РЕЖИМ A/B/C
# ============================================================

def get_current_mode():
    settings = load_settings()
    mode = settings.get("mode", "B")
    if mode not in ("A", "B", "C"):
        mode = "B"
    return mode, {"mode": mode}

def set_mode(mode: str):
    if mode not in ("A", "B", "C"):
        raise ValueError("Режим должен быть A, B или C")
    settings = load_settings()
    settings["mode"] = mode
    save_settings(settings)
