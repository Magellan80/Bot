import requests
import time


def download(symbol: str, interval: str, total_needed: int):

    print(f"\nDownloading {symbol} {interval} ...")

    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None

    # защита от зависания
    max_retries = 5

    while len(all_data) < total_needed:

        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": 1000
        }

        if end_time:
            params["endTime"] = end_time

        retries = 0

        while True:
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                retries += 1
                print(f"Retry {retries}/{max_retries} for {symbol} {interval} due to error: {e}")
                time.sleep(0.5)

                if retries >= max_retries:
                    print("Max retries reached. Stopping.")
                    return []

        # Binance может вернуть пустой массив → конец истории
        if not data:
            print("No more data returned by Binance.")
            break

        # защита от некорректного формата
        if not isinstance(data, list) or not isinstance(data[0], list):
            print("Unexpected data format from Binance.")
            break

        # добавляем новые свечи в начало
        all_data = data + all_data

        # обновляем end_time для следующего запроса
        end_time = data[0][0] - 1

        print(f"{interval}: {len(all_data)} candles")

        # защита от rate-limit
        time.sleep(0.15)

    # преобразуем в удобный формат
    candles = []
    for k in all_data:
        try:
            candles.append({
                "timestamp": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4])
            })
        except Exception:
            # пропускаем битые свечи
            continue

    print(f"{interval} DONE. Total: {len(candles)} candles.")
    return candles
