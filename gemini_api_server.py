import os
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# --- Настройка ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загружаем переменные окружения (API ключ) из файла .env
load_dotenv()

# Создаем приложение FastAPI
app = FastAPI(
    title="Gemini API Gateway",
    description="Централизованный сервер для обработки запросов к Gemini API"
)

# --- Настройка Gemini API ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("Ключ GEMINI_API_KEY не найден в переменных окружения (.env)")
    
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API успешно настроен.")
except Exception as e:
    logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА при настройке Gemini API: {e}")
    # Если мы не можем запустить Gemini, серверу нет смысла работать.
    # В реальном продакшене можно добавить более сложную обработку.

# --- Модели данных для входящих запросов (Pydantic) ---
# Это обеспечивает автоматическую валидацию данных, которые приходят на сервер.

class ParseDealRequest(BaseModel):
    description: str

class BuyoutPlanRequest(BaseModel):
    deal_description: str
    plan_description: str

# ==============================================================================
# КОПИРУЕМ ВАШИ ФУНКЦИИ ОБРАБОТКИ GEMINI ПРЯМО СЮДА
# Ничего менять в них не нужно.
# ==============================================================================

def parse_custom_deal_with_gemini(description: str) -> dict | None:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""
        Проанализируй описание комплекта для курьера: "{description}".
        Твоя задача - извлечь название/модель велосипеда, его серийный номер (VIN),
        а также количество, емкость (Ah) и серийные номера аккумуляторов.
        Верни результат СТРОГО в формате JSON, без лишних слов или ```json```.
        Ключи: "model_name", "bike_number", "batteries".
        "batteries" должен быть списком словарей, каждый с ключами "capacity" и "number".
        Если что-то не найдено, значение должно быть null.

        Пример ответа:
        {{
          "model_name": "Монстр-Гибрид",
          "bike_number": "12345",
          "batteries": [
            {{ "capacity": "30Ah", "number": "312123" }},
            {{ "capacity": "30Ah", "number": "312312" }}
          ]
        }}
        """
        response = model.generate_content(prompt, stream=False)
        response.resolve()

        cleaned_response_text = response.text.strip().replace("`", "")
        if cleaned_response_text.lower().startswith("json"):
            cleaned_response_text = cleaned_response_text[4:].strip()

        logger.info(f"Получены данные о комплекте от Gemini: {cleaned_response_text}")
        data = json.loads(cleaned_response_text)
        
        return data if isinstance(data, dict) else None

    except Exception as e:
        logger.error(f"Ошибка при парсинге комплекта с Gemini: {e}", exc_info=True)
        return None

def get_buyout_plans_with_gemini(deal_description: str, plan_description: str) -> dict | None:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""
        Проанализируй описание комплекта: "{deal_description}" и желаемый план рассрочки: "{plan_description}".
        Рыночная стоимость комплекта 80,000-120,000 руб.
        Твоя задача - сгенерировать ОДИН структурированный план рассрочки.
        Верни результат СТРОГО в формате JSON. Ответ должен быть только чистым JSON объектом.
        Ключи: "label", "full_label", "first_payment", "total_payments", "period_days".

        Пример твоего идеального ответа:
        {{
            "plan_1": {{
                "label": "5 мес / 10000 ₽",
                "full_label": "5 месяцев: 10 платежей по 10 000 ₽ (раз в 2 недели)",
                "first_payment": 10000,
                "total_payments": 10,
                "period_days": 14
            }}
        }}
        """
        response = model.generate_content(prompt, stream=False)
        response.resolve()

        cleaned_response_text = response.text.strip().replace("`", "")
        if cleaned_response_text.lower().startswith("json"):
            cleaned_response_text = cleaned_response_text[4:].strip()

        logger.info(f"Получены планы выкупа от Gemini: {cleaned_response_text}")
        data = json.loads(cleaned_response_text)

        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            return data
        else:
            logger.error(f"Ответ от Gemini (планы выкупа) имеет неверную структуру: {data}")
            return None

    except Exception as e:
        logger.error(f"Ошибка при генерации планов выкупа с Gemini: {e}", exc_info=True)
        return None

# ==============================================================================
# СОЗДАЕМ API ЭНДПОИНТЫ (точки входа для ваших ботов)
# ==============================================================================

@app.post("/parse-deal")
async def api_parse_deal(request: ParseDealRequest):
    """
    Эндпоинт для парсинга описания кастомной сделки.
    Принимает JSON вида: {"description": "Текст описания"}
    """
    logger.info(f"Получен запрос на /parse-deal с описанием: {request.description}")
    parsed_data = parse_custom_deal_with_gemini(request.description)
    
    if parsed_data:
        return parsed_data
    else:
        # Возвращаем ошибку, если Gemini не смог обработать запрос
        raise HTTPException(
            status_code=400, 
            detail="Не удалось распознать описание. Пожалуйста, попробуйте сформулировать иначе."
        )

@app.post("/get-buyout-plans")
async def api_get_buyout_plans(request: BuyoutPlanRequest):
    """
    Эндпоинт для генерации планов выкупа.
    Принимает JSON вида: {"deal_description": "...", "plan_description": "..."}
    """
    logger.info(f"Получен запрос на /get-buyout-plans для: {request.deal_description}")
    plans = get_buyout_plans_with_gemini(request.deal_description, request.plan_description)

    if plans:
        return plans
    else:
        raise HTTPException(
            status_code=400,
            detail="Не удалось сгенерировать план выкупа. Попробуйте описать его по-другому."
        )

@app.get("/")
async def root():
    return {"status": "Gemini API Gateway is running"}