import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Telegram Bot 설정
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # print( url )

    params = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message
    }

    print()
    print("--- Send Message")
    print(message)
    print("--- --- ---")
    print()

    response = requests.post(url, params=params)
    return response.json()
