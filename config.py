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
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    print("⚠ WARNING: BYBIT_API_KEY / BYBIT_API_SECRET not set — trading disabled")

# ====== ФАЙЛ НАСТРОЕК ======
SETTINGS_FILE = "settings.json"

# ====== КЭШ НАСТРОЕК ======
_settings_cache = None

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


# ============================================================
#   SETTINGS (С КЭШЕМ)
# ============================================================

def load_settings(force_reload: bool = False):
    global _settings_cache

    defaults = {
        "mode": "A",
        "min_score": DEFAULT_MIN_SCORE,
        "strictness_level": "strict",
        "reversal_requires_state": True,
        "reversal_state_ttl_sec": 7200,
        "reversal_min_score_bonus": 10,
        "reversal_min_delay_bars": 3,
        "bot_mode": "SCREENER",
    }

    # Если кэш есть и не требуется принудительная перезагрузка
    if _settings_cache is not None and not force_reload:
        return _settings_cache

    if not os.path.exists(SETTINGS_FILE):
        _settings_cache = defaults
        return _settings_cache

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            defaults.update(data)

    except Exception as e:
        print(f"[CONFIG] Error loading settings: {e}")

    _settings_cache = defaults
    return _settings_cache


def save_settings(settings: dict):
    global _settings_cache

    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[CONFIG] Error saving settings: {e}")
        return

    # Обновляем кэш
    _settings_cache = settings


def get_current_mode():
    settings = load_settings()
    mode_key = settings.get("mode", "A")

    if mode_key not in MODES:
        mode_key = "A"

    return mode_key, MODES[mode_key]


def get_current_min_score():
    settings = load_settings()
    return int(settings.get("min_score", DEFAULT_MIN_SCORE))
