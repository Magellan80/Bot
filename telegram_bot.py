import asyncio
from contextlib import suppress

from aiogram import Bot, Dispatcher, types
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter, TelegramServerError
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery

from config import (
    BOT_TOKEN,
    ADMIN_ID,
    DEFAULT_MIN_SCORE,
    load_settings,
    save_settings,
    MODES,
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
)

from screener import scanner_loop
from trading_engine import (
    TradingEngine,
    sync_loop,
    kill_switch_loop,
)
from broker_bybit_futures import BrokerBybitFutures
from bybit_ws import BybitWebSocket


# ============================================================
#   GLOBALS
# ============================================================

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

engine: TradingEngine | None = None

scanner_task: asyncio.Task | None = None
price_task: asyncio.Task | None = None
sync_task: asyncio.Task | None = None
kill_task: asyncio.Task | None = None
ws_task: asyncio.Task | None = None

ws_client: BybitWebSocket | None = None

trading_supervisor_task: asyncio.Task | None = None
scanner_watchdog_task: asyncio.Task | None = None


# ============================================================
#   FALLBACK PRICE MONITOR LOOP (REST)
#   — резерв к WebSocket, вариант B
# ============================================================

async def price_monitor_loop(engine: TradingEngine, interval: int = 5):
    """
    Резервный источник цен:
    - раз в interval секунд опрашивает цены по всем открытым символам через REST
    - обновляет движок через on_price_update
    - обновляет heartbeat, чтобы kill-switch видел активность
    """
    while True:
        try:
            symbols = list(engine.positions.keys())
            for symbol in symbols:
                try:
                    price = await engine.broker.get_last_price(symbol)
                except Exception:
                    engine._register_rest_error("price_monitor_loop:get_last_price")
                    continue

                if price is None:
                    continue

                try:
                    await engine.on_price_update(symbol, price)
                    engine._register_price_heartbeat()
                except Exception:
                    engine._register_rest_error("price_monitor_loop:on_price_update")
                    continue
        except Exception:
            # глобальная защита цикла
            engine._register_rest_error("price_monitor_loop:loop_error")

        await asyncio.sleep(interval)


# ============================================================
#   SAFE HELPERS
# ============================================================

async def safe_answer(call: CallbackQuery, text: str | None = None):
    with suppress(Exception):
        await call.answer(text)


async def safe_edit(message, text, reply_markup=None):
    try:
        if message.text == text and message.reply_markup == reply_markup:
            return
        await message.edit_text(text, reply_markup=reply_markup)
    except Exception as e:
        if "message is not modified" in str(e):
            return


async def safe_send_message(chat_id: int, text: str):
    """
    Безопасная отправка сообщений с обработкой сетевых ошибок и retry-after.
    """
    delay = 1
    while True:
        try:
            await bot.send_message(chat_id, text)
            return
        except TelegramRetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except (TelegramNetworkError, TelegramServerError) as e:
            print(f"[safe_send_message] network/server error: {e}, retry in {delay}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        except Exception as e:
            print(f"[safe_send_message] fatal error: {e}")
            return


async def safe_send_photo(chat_id: int, photo):
    delay = 1
    while True:
        try:
            await bot.send_photo(chat_id, photo)
            return
        except TelegramRetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except (TelegramNetworkError, TelegramServerError) as e:
            print(f"[safe_send_photo] network/server error: {e}, retry in {delay}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        except Exception as e:
            print(f"[safe_send_photo] fatal error: {e}")
            return


# ============================================================
#   INLINE MENUS
# ============================================================

def sensitivity_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="25 (Aggressive)", callback_data="sens_25"),
            InlineKeyboardButton(text="40 (Normal)", callback_data="sens_40"),
            InlineKeyboardButton(text="55 (Conservative)", callback_data="sens_55"),
        ],
        [InlineKeyboardButton(text="⬅ Назад", callback_data="back_main")]
    ])


def mode_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="A", callback_data="mode_A"),
            InlineKeyboardButton(text="B", callback_data="mode_B"),
            InlineKeyboardButton(text="C", callback_data="mode_C"),
        ],
        [InlineKeyboardButton(text="⬅ Назад", callback_data="back_main")]
    ])


def bot_mode_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="СКРИНЕР", callback_data="botmode_SCREENER"),
        ],
        [
            InlineKeyboardButton(text="СКРИНЕР + ТОРГОВЛЯ", callback_data="botmode_TRADING"),
        ],
        [InlineKeyboardButton(text="⬅ Назад", callback_data="back_main")],
    ])


# ============================================================
#   MAIN MENU
# ============================================================

def main_menu():
    settings = load_settings()
    mode_key = settings.get("mode", "A")
    min_score = settings.get("min_score", DEFAULT_MIN_SCORE)
    mode_name = MODES[mode_key]["name"]

    bot_mode_key = settings.get("bot_mode", "SCREENER")
    bot_mode_name = "СКРИНЕР + ТОРГОВЛЯ" if bot_mode_key == "TRADING" else "СКРИНЕР"

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="▶ Запустить сканер", callback_data="start_scanner")],
        [InlineKeyboardButton(text="⏹ Остановить сканер", callback_data="stop_scanner")],
        [InlineKeyboardButton(text="🎛 Режим A/B/C", callback_data="mode_menu")],
        [InlineKeyboardButton(text="⚙ Чувствительность", callback_data="sensitivity_menu")],
        [InlineKeyboardButton(text="🤖 Режим работы бота", callback_data="bot_mode_menu")],
        [InlineKeyboardButton(text="📊 Статус сканера", callback_data="scanner_status")],
    ])

    text = (
        f"Главное меню:\n\n"
        f"Текущий режим сигналов: {mode_name} ({mode_key})\n"
        f"Текущая чувствительность: {min_score}\n"
        f"Режим работы бота: {bot_mode_name}\n"
    )
    return text, kb


# ============================================================
#   WATCHDOG — авто‑рестарт сканера
# ============================================================

async def scanner_watchdog():
    global scanner_task, engine, price_task, sync_task, ws_task, ws_client, kill_task

    while True:
        await asyncio.sleep(10)

        if scanner_task is None:
            continue

        if scanner_task.done():
            try:
                err = scanner_task.exception()
            except Exception:
                err = None

            await safe_send_message(
                ADMIN_ID,
                f"⚠️ Сканер остановился и будет перезапущен.\nПричина: {repr(err)}"
            )

            settings = load_settings()
            min_score = int(settings.get("min_score", DEFAULT_MIN_SCORE))
            bot_mode = settings.get("bot_mode", "SCREENER")

            async def send_text(text):
                await safe_send_message(ADMIN_ID, text)

            async def send_photo(photo):
                await safe_send_photo(ADMIN_ID, photo)

            # TRADING MODE
            if bot_mode == "TRADING":
                if engine is None:
                    broker = BrokerBybitFutures(BYBIT_API_KEY, BYBIT_API_SECRET)
                    engine = TradingEngine(broker)

                # резервный REST‑монитор цен
                if price_task is None or price_task.done():
                    price_task = asyncio.create_task(price_monitor_loop(engine, interval=5))

                # sync SL/TP
                if sync_task is None or sync_task.done():
                    sync_task = asyncio.create_task(sync_loop(engine, interval=15))

                # kill-switch
                if kill_task is None or kill_task.done():
                    kill_task = asyncio.create_task(
                        kill_switch_loop(engine, max_silence_sec=60, max_rest_errors=20)
                    )

                # WebSocket — основной источник цен
                if ws_task and not ws_task.done():
                    ws_task.cancel()
                ws_client = BybitWebSocket(BYBIT_API_KEY, BYBIT_API_SECRET, engine)
                ws_task = asyncio.create_task(ws_client.run())

                scanner_task = asyncio.create_task(
                    scanner_loop(send_text, send_photo, min_score, engine=engine)
                )

            # SCREENER MODE
            else:
                if ws_task and not ws_task.done():
                    ws_task.cancel()
                ws_client = None

                scanner_task = asyncio.create_task(
                    scanner_loop(send_text, send_photo, min_score)
                )


# ============================================================
#   TRADING TASKS SUPERVISOR (self‑healing)
# ============================================================

async def trading_tasks_supervisor():
    """
    Следит за задачами торговли (price_monitor, sync, kill_switch, ws) и перезапускает их при падении.
    Работает только в режиме TRADING.
    """
    global engine, price_task, sync_task, kill_task, ws_task, ws_client

    while True:
        await asyncio.sleep(5)

        settings = load_settings()
        bot_mode = settings.get("bot_mode", "SCREENER")
        if bot_mode != "TRADING":
            # в режиме SCREENER ничего не делаем, но аккуратно гасим задачи
            if price_task and price_task.done():
                price_task = None
            if sync_task and sync_task.done():
                sync_task = None
            if kill_task and kill_task.done():
                kill_task = None
            if ws_task and ws_task.done():
                ws_task = None
            continue

        if engine is None:
            broker = BrokerBybitFutures(BYBIT_API_KEY, BYBIT_API_SECRET)
            engine = TradingEngine(broker)

        # price monitor
        if price_task is None or price_task.done():
            price_task = asyncio.create_task(price_monitor_loop(engine, interval=5))

        # sync loop
        if sync_task is None or sync_task.done():
            sync_task = asyncio.create_task(sync_loop(engine, interval=15))

        # kill-switch
        if kill_task is None or kill_task.done():
            kill_task = asyncio.create_task(
                kill_switch_loop(engine, max_silence_sec=60, max_rest_errors=20)
            )

        # WebSocket
        if ws_task is None or ws_task.done():
            ws_client = BybitWebSocket(BYBIT_API_KEY, BYBIT_API_SECRET, engine)
            ws_task = asyncio.create_task(ws_client.run())


# ============================================================
#   COMMANDS
# ============================================================

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        return await message.answer("У вас нет доступа")

    text, kb = main_menu()
    await message.answer(text, reply_markup=kb)


# ============================================================
#   CALLBACKS
# ============================================================

@dp.callback_query(lambda c: c.data == "back_main")
async def cb_back_main(call: CallbackQuery):
    await safe_answer(call)
    text, kb = main_menu()
    await safe_edit(call.message, text, reply_markup=kb)


@dp.callback_query(lambda c: c.data == "sensitivity_menu")
async def cb_sensitivity_menu(call: CallbackQuery):
    await safe_answer(call)
    await safe_edit(call.message, "Выбери чувствительность:", reply_markup=sensitivity_menu())


@dp.callback_query(lambda c: c.data.startswith("sens_"))
async def cb_set_sensitivity(call: CallbackQuery):
    await safe_answer(call)

    value = int(call.data.split("_")[1])
    settings = load_settings()
    settings["min_score"] = value
    save_settings(settings)

    text, kb = main_menu()
    await safe_edit(call.message, f"Чувствительность установлена: {value}\n\n" + text, reply_markup=kb)


@dp.callback_query(lambda c: c.data == "mode_menu")
async def cb_mode_menu(call: CallbackQuery):
    await safe_answer(call)
    await safe_edit(call.message, "Выбери режим A/B/C:", reply_markup=mode_menu())


@dp.callback_query(lambda c: c.data.startswith("mode_"))
async def cb_set_mode(call: CallbackQuery):
    await safe_answer(call)

    mode_key = call.data.split("_")[1]
    if mode_key not in MODES:
        return

    settings = load_settings()
    settings["mode"] = mode_key
    save_settings(settings)

    mode_name = MODES[mode_key]["name"]

    text, kb = main_menu()
    await safe_edit(
        call.message,
        f"Режим сигналов переключён на: {mode_name} ({mode_key})\n\n" + text,
        reply_markup=kb
    )


@dp.callback_query(lambda c: c.data == "bot_mode_menu")
async def cb_bot_mode_menu(call: CallbackQuery):
    await safe_answer(call)

    global scanner_task
    if scanner_task and not scanner_task.done():
        text, kb = main_menu()
        return await safe_edit(
            call.message,
            "Нельзя менять режим работы бота, пока сканер запущен.\n\nСначала останови сканер.\n\n" + text,
            reply_markup=kb
        )

    await safe_edit(call.message, "Выбери режим работы бота:", reply_markup=bot_mode_menu())


@dp.callback_query(lambda c: c.data.startswith("botmode_"))
async def cb_set_bot_mode(call: CallbackQuery):
    await safe_answer(call)

    global scanner_task, ws_task, ws_client, price_task, sync_task, kill_task, engine

    if scanner_task and not scanner_task.done():
        text, kb = main_menu()
        return await safe_edit(
            call.message,
            "Нельзя менять режим работы бота, пока сканер запущен.\n\nСначала останови сканер.\n\n" + text,
            reply_markup=kb
        )

    mode = call.data.split("_")[1]

    settings = load_settings()
    settings["bot_mode"] = mode
    save_settings(settings)

    # при смене режима — останавливаем все фоновые задачи торговли
    if price_task and not price_task.done():
        price_task.cancel()
        price_task = None

    if sync_task and not sync_task.done():
        sync_task.cancel()
        sync_task = None

    if kill_task and not kill_task.done():
        kill_task.cancel()
        kill_task = None

    if ws_task and not ws_task.done():
        ws_task.cancel()
        ws_task = None
    ws_client = None

    # движок можно оставить, чтобы не терять состояние, но в SCREENER он не используется
    msg = "Режим работы бота: СКРИНЕР + ТОРГОВЛЯ" if mode == "TRADING" else "Режим работы бота: СКРИНЕР"

    text, kb = main_menu()
    await safe_edit(call.message, msg + "\n\n" + text, reply_markup=kb)


@dp.callback_query(lambda c: c.data == "start_scanner")
async def cb_start_scanner(call: CallbackQuery):
    global scanner_task, engine, price_task, sync_task, ws_task, ws_client, kill_task

    await safe_answer(call)

    if scanner_task and not scanner_task.done():
        text, kb = main_menu()
        return await safe_edit(
            call.message,
            "Сканер уже работает.\n\n" + text,
            reply_markup=kb
        )

    settings = load_settings()
    min_score = int(settings.get("min_score", DEFAULT_MIN_SCORE))
    bot_mode = settings.get("bot_mode", "SCREENER")

    async def send_text(text):
        await safe_send_message(ADMIN_ID, text)

    async def send_photo(photo):
        await safe_send_photo(ADMIN_ID, photo)

    # TRADING MODE
    if bot_mode == "TRADING":
        if engine is None:
            broker = BrokerBybitFutures(BYBIT_API_KEY, BYBIT_API_SECRET)
            engine = TradingEngine(broker)

        # резервный REST‑монитор цен
        if price_task is None or price_task.done():
            price_task = asyncio.create_task(price_monitor_loop(engine, interval=5))

        # sync SL/TP
        if sync_task is None or sync_task.done():
            sync_task = asyncio.create_task(sync_loop(engine, interval=15))

        # kill-switch
        if kill_task is None or kill_task.done():
            kill_task = asyncio.create_task(
                kill_switch_loop(engine, max_silence_sec=60, max_rest_errors=20)
            )

        # WebSocket — основной источник цен
        if ws_task and not ws_task.done():
            ws_task.cancel()
        ws_client = BybitWebSocket(BYBIT_API_KEY, BYBIT_API_SECRET, engine)
        ws_task = asyncio.create_task(ws_client.run())

        scanner_task = asyncio.create_task(
            scanner_loop(send_text, send_photo, min_score, engine=engine)
        )
        status_line = "Сканер запущен в режиме: СКРИНЕР + ТОРГОВЛЯ."

    # SCREENER MODE
    else:
        if ws_task and not ws_task.done():
            ws_task.cancel()
        ws_client = None

        scanner_task = asyncio.create_task(
            scanner_loop(send_text, send_photo, min_score)
        )
        status_line = "Сканер запущен в режиме: СКРИНЕР (без торговли)."

    text, kb = main_menu()
    await safe_edit(call.message, status_line + "\n\n" + text, reply_markup=kb)


@dp.callback_query(lambda c: c.data == "stop_scanner")
async def cb_stop_scanner(call: CallbackQuery):
    global scanner_task, ws_task, ws_client

    await safe_answer(call, "Останавливаю...")

    if scanner_task and not scanner_task.done():
        scanner_task.cancel()
        scanner_task = None

    if ws_task and not ws_task.done():
        ws_task.cancel()
        ws_task = None
    ws_client = None

    text, kb = main_menu()
    await safe_edit(
        call.message,
        "Сканер остановлен.\n\n" + text,
        reply_markup=kb
    )


@dp.callback_query(lambda c: c.data == "scanner_status")
async def cb_scanner_status(call: CallbackQuery):
    await safe_answer(call)

    status = "🟢 Работает" if scanner_task and not scanner_task.done() else "🔴 Остановлен"
    settings = load_settings()
    mode_key = settings.get("mode", "A")
    mode_name = MODES[mode_key]["name"]
    min_score = settings.get("min_score", DEFAULT_MIN_SCORE)
    bot_mode = settings.get("bot_mode", "SCREENER")
    bot_mode_name = "СКРИНЕР + ТОРГОВЛЯ" if bot_mode == "TRADING" else "СКРИНЕР"

    text, kb = main_menu()
    await safe_edit(
        call.message,
        f"Статус сканера: {status}\n"
        f"Текущий режим сигналов: {mode_name} ({mode_key})\n"
        f"Текущая чувствительность: {min_score}\n"
        f"Режим работы бота: {bot_mode_name}\n\n" + text,
        reply_markup=kb
    )


# ============================================================
#   SELF‑HEALING POLLING WRAPPER
# ============================================================

async def run_polling_forever():
    """
    Self‑healing контур вокруг dp.start_polling:
    - перезапускает polling при сетевых ошибках
    - использует экспоненциальный backoff
    """
    delay = 1
    while True:
        try:
            print("Запуск Telegram polling...")
            await dp.start_polling(bot)
        except TelegramRetryAfter as e:
            print(f"[polling] RetryAfter: {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
        except (TelegramNetworkError, TelegramServerError) as e:
            print(f"[polling] network/server error: {e}, retry in {delay}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        except Exception as e:
            print(f"[polling] fatal error: {e}, retry in {delay}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        else:
            # если polling завершился без ошибок — сбрасываем delay
            delay = 1


# ============================================================
#   MAIN
# ============================================================

async def main():
    global engine, trading_supervisor_task, scanner_watchdog_task

    print("Telegram бот запущен. Режимы: СКРИНЕР / СКРИНЕР + ТОРГОВЛЯ.")

    settings = load_settings()
    bot_mode = settings.get("bot_mode", "SCREENER")

    # Если режим TRADING — создаём движок сразу
    if bot_mode == "TRADING":
        broker = BrokerBybitFutures(BYBIT_API_KEY, BYBIT_API_SECRET)
        engine = TradingEngine(broker)

    # watchdog сканера
    scanner_watchdog_task = asyncio.create_task(scanner_watchdog())

    # supervisor для торговых задач (self‑healing)
    trading_supervisor_task = asyncio.create_task(trading_tasks_supervisor())

    # Telegram polling с self‑healing
    await run_polling_forever()


if __name__ == "__main__":
    asyncio.run(main())
