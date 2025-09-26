import os
import json
import logging
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import httpx

# --- Настройка ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI(title="Универсальный API Шлюз для Gemini и Telegram")

# --- Конфигурация из .env ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    INTERNAL_SECRET = os.getenv("INTERNAL_SECRET")

    if not all([GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, INTERNAL_SECRET]):
        raise ValueError("Не установлены все необходимые переменные: GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, INTERNAL_SECRET")
    
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("API Gemini и токен бота успешно настроены на сервере.")
except Exception as e:
    logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА при настройке: {e}")
    # Приложение не сможет работать без этих ключей, можно было бы и завершить, но FastAPI продолжит.

# --- Модели данных для FastAPI ---
class RecognizeDocsRequest(BaseModel):
    images_base64: list[str]
    country: str

class NotifyRequest(BaseModel):
    user_id: int
    text: str
    
class ParseDealRequest(BaseModel):
    description: str

class BuyoutPlanRequest(BaseModel):
    deal_description: str

# ==============================================================================
# ВСЯ ЛОГИКА GEMINI С ТВОИМИ ОРИГИНАЛЬНЫМИ, ПРАВИЛЬНЫМИ ПРОМПТАМИ
# ==============================================================================

def recognize_documents_with_gemini(images: list, country: str) -> dict | None:
    """Распознает документы с использованием твоих оригинальных промптов."""
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        model = genai.GenerativeModel('gemini-2.5-flash', safety_settings=safety_settings)

        if country == 'РФ':
            # ТВОЙ ИДЕАЛЬНЫЙ ПРОМПТ ДЛЯ ПАСПОРТА РФ
            prompt = """
            Проанализируй эти три изображения: основной разворот паспорта РФ, страница с пропиской и селфи с паспортом.
            Извлеки все данные и верни их в виде ОДНОГО плоского JSON объекта.
            Ключи: "Фамилия", "Имя", "Отчество", "Дата рождения", "Серия и номер паспорта", "Кем выдан", "Дата выдачи", "Адрес регистрации".
            Если поле не найдено, значение должно быть пустой строкой.
            Ответ должен быть только чистым JSON.
            """
        else:
            # ТВОЙ ИДЕАЛЬНЫЙ ПРОМПТ ДЛЯ ИНОСТРАННЫХ ДОКУМЕНТОВ
            prompt = """
            Проанализируй эти четыре изображения: паспорт иностранного гражданина, регистрация в РФ, патент и селфи с паспортом.
            Извлеки все данные и верни их в виде ОДНОГО плоского JSON объекта.
            Ключи: "ФИО", "Гражданство", "Дата рождения", "Номер паспорта", "Адрес регистрации в РФ", "Номер патента".
            Если поле не найдено, значение должно быть пустой строкой.
            Ответ должен быть только чистым JSON.
            """
        
        response = model.generate_content([prompt] + images, request_options={"timeout": 120})
        response.resolve()
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        logger.info(f"Ответ от Gemini (документы): {cleaned_text}")
        return json.loads(cleaned_text)

    except Exception as e:
        logger.error(f"Ошибка Gemini (документы): {e}", exc_info=True)
        return None

def parse_custom_deal_with_gemini(description: str) -> dict | None:
    """Распознает комплектующие из описания сделки."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
        Проанализируй описание комплекта: "{description}".
        Извлеки название/модель велосипеда, его серийный номер (VIN),
        а также количество, емкость (Ah) и серийные номера аккумуляторов.
        Верни результат СТРОГО в формате JSON.
        Ключи: "model_name", "bike_number", "batteries".
        "batteries" должен быть списком словарей, каждый с ключами "capacity" и "number".
        Если что-то не найдено, значение должно быть null.
        """
        response = model.generate_content(prompt)
        response.resolve()
        cleaned_text = response.text.strip().replace("`", "").lstrip("json").strip()
        logger.info(f"Ответ от Gemini (комплект): {cleaned_text}")
        return json.loads(cleaned_text)
    except Exception as e:
        logger.error(f"Ошибка Gemini (комплект): {e}", exc_info=True)
        return None

def get_buyout_plans_with_gemini(deal_description: str) -> dict | None:
    """Генерирует планы выкупа по описанию комплекта."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        # ТВОЙ ИДЕАЛЬНЫЙ ПРОМПТ ДЛЯ ПЛАНОВ
        prompt = f"""
        Проанализируй описание комплекта для курьера: "{deal_description}".
        Это может быть электровелосипед, аккумуляторы или и то, и другое.
        Примерная рыночная стоимость такого комплекта около 80,000 - 120,000 рублей.
        Твоя задача - сгенерировать 3-4 варианта плана рассрочки (выкупа) для этого комплекта.
        Верни результат СТРОГО в формате JSON. Ответ должен быть только чистым JSON объектом,
        без лишних слов, комментариев или ```json ``` оберток.
        Ключами в JSON должны быть короткие идентификаторы (например, "plan_1", "plan_2"),
        а значениями - словари с ключами:
        - "label": Короткое и понятное описание для кнопки (например, "3 мес / 16000 ₽").
        - "full_label": Полное описание для подтверждения (например, "3 месяца: 7 платежей по 16 000 ₽").
        - "first_payment": Сумма первого взноса (число).
        - "total_payments": Общее количество платежей (число).
        - "period_days": Периодичность платежей в днях (30 для месяца, 14 для 2 недель).
        Пример твоего идеального ответа:
        {{
            "plan_1": {{"label": "3 мес / 16000 ₽", "full_label": "3 месяца: 7 платежей по 16 000 ₽ (раз в 2 недели)", "first_payment": 16000, "total_payments": 7, "period_days": 14}},
            "plan_2": {{"label": "4 мес / 21000 ₽", "full_label": "4 месяца: 5 платежей по 21 000 ₽ (раз в месяц)", "first_payment": 21000, "total_payments": 5, "period_days": 30}}
        }}
        """
        response = model.generate_content(prompt)
        response.resolve()
        cleaned_text = response.text.strip().replace("`", "").lstrip("json").strip()
        logger.info(f"Ответ от Gemini (планы выкупа): {cleaned_text}")
        return json.loads(cleaned_text)
    except Exception as e:
        logger.error(f"Ошибка Gemini (планы выкупа): {e}", exc_info=True)
        return None

# ==============================================================================
# API ЭНДПОИНТЫ
# ==============================================================================

@app.post("/recognize-documents")
async def api_recognize_documents(request: RecognizeDocsRequest):
    logger.info(f"Входящий запрос /recognize-documents для страны: {request.country}")
    images = [Image.open(BytesIO(base64.b64decode(b64))) for b64 in request.images_base64]
    data = recognize_documents_with_gemini(images, request.country)
    if data: return data
    raise HTTPException(status_code=500, detail="Ошибка распознавания документов на стороне Gemini.")

@app.post("/parse-deal")
async def api_parse_deal(request: ParseDealRequest):
    logger.info(f"Входящий запрос /parse-deal: {request.description}")
    data = parse_custom_deal_with_gemini(request.description)
    if data: return data
    raise HTTPException(status_code=400, detail="Не удалось распознать описание комплекта.")

@app.post("/generate-buyout-plans")
async def api_generate_buyout_plans(request: BuyoutPlanRequest):
    logger.info(f"Входящий запрос /generate-buyout-plans: {request.deal_description}")
    data = get_buyout_plans_with_gemini(request.deal_description)
    if data: return data
    raise HTTPException(status_code=400, detail="Не удалось сгенерировать планы выкупа.")

@app.post("/notify")
async def notify_user(request: NotifyRequest, http_request: Request):
    logger.info(f"Входящий запрос /notify для user_id: {request.user_id}")
    if http_request.headers.get('x-internal-secret') != INTERNAL_SECRET:
        logger.warning(f"Попытка неавторизованного доступа к /notify с IP: {http_request.client.host}")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    telegram_api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(telegram_api_url, json={"chat_id": request.user_id, "text": request.text, "parse_mode": "Markdown"})
            response.raise_for_status()
            logger.info(f"Уведомление успешно отправлено пользователю {request.user_id}")
            return {"success": True}
        except httpx.HTTPStatusError as e:
            error_info = e.response.json()
            logger.error(f"Ошибка от Telegram API для user {request.user_id}: {error_info}")
            raise HTTPException(status_code=400, detail=f"Telegram API error: {error_info.get('description')}")

@app.get("/")
async def root():
    return {"status": "Универсальный API-шлюз работает"}