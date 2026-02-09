import json
import os
from dotenv import load_dotenv

# Загружаем .env
load_dotenv()

# ====== ТЕЛЕГРАМ ======
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")

ADMIN_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
if ADMIN_ID == 0:
    raise RuntimeError("TELEGRAM_CHAT_ID is not set in .env")

# ====== BYBIT API ======
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# В режиме скринера Bybit ключи могут быть пустыми — это нормально
# Поэтому НЕ выбрасываем ошибку
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    print("⚠ WARNING: BYBIT_API_KEY / BYBIT_API_SECRET not set — trading disabled")

# ====== ФАЙЛ НАСТРОЕК ======
SETTINGS_FILE = "settings.json"

# ====== РЕЖИМЫ A / B / C ======
MODES = {
    "A": {
        "name": "Вариант A — умеренно строгий",
        "pump_5m": 7.0,
        "volume_spike": 4.0,
        "up_bars": 5,
        "change_24h": 10.0,
        "volume_usdt": 5_000_000
    },
    "B": {
        "name": "Вариант B — очень строгий",
        "pump_5m": 10.0,
        "volume_spike": 5.0,
        "up_bars": 6,
        "change_24h": 15.0,
        "volume_usdt": 10_000_000
    },
    "C": {
        "name": "Вариант C — супер-строгий",
        "pump_5m": 15.0,
        "volume_spike": 6.0,
        "up_bars": 7,
        "change_24h": 20.0,
        "volume_usdt": 20_000_000
    }
}

# ====== ЧУВСТВИТЕЛЬНОСТЬ ПО УМОЛЧАНИЮ ======
DEFAULT_MIN_SCORE = 40


def load_settings():
    defaults = {
        "mode": "A",
        "min_score": DEFAULT_MIN_SCORE,
        "strictness_level": "strict",
        "reversal_requires_state": True,
        "reversal_state_ttl_sec": 7200,
        "reversal_min_score_bonus": 10,
        "reversal_min_delay_bars": 3,
        "max_concurrency": 20,
        "orderbook_max_spread_pct": 0.5,
        "orderbook_min_total_vol": 500.0,
        "orderbook_depth_n": 10,
    }
    if not os.path.exists(SETTINGS_FILE):
        return defaults
    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        defaults.update(data)
        if "strictness_level" not in data and "strict_mode" in data:
            defaults["strictness_level"] = "strict" if data["strict_mode"] else "soft"
    return defaults


def save_settings(settings: dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


def get_current_mode():
    settings = load_settings()
    mode_key = settings.get("mode", "A")
    if mode_key not in MODES:
        mode_key = "A"
    return mode_key, MODES[mode_key]


def get_current_min_score():
    settings = load_settings()
    return int(settings.get("min_score", DEFAULT_MIN_SCORE))
