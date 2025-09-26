import os
import json
import logging
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# --- Настройка ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI(title="Gemini Universal API Gateway")

# --- Настройка Gemini API ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("Ключ GEMINI_API_KEY не найден в .env")
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API успешно настроен на сервере.")
except Exception as e:
    logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА при настройке Gemini API: {e}")

# --- Модели данных для входящих запросов ---

class RecognizeDocsRequest(BaseModel):
    images_base64: list[str]
    country: str

# НОВАЯ МОДЕЛЬ ДЛЯ ПЛАНОВ ВЫКУПА
class BuyoutPlanRequest(BaseModel):
    deal_description: str

# ==============================================================================
# КОПИРУЕМ ВАШИ ФУНКЦИИ GEMINI СЮДА
# ==============================================================================

# Ваша функция для генерации планов выкупа (с исправленной моделью)
def get_buyout_plans_with_gemini(deal_description: str) -> dict | None:
    try:
        # ИСПОЛЬЗУЕМ НАДЕЖНОЕ НАЗВАНИЕ МОДЕЛИ
        model = genai.GenerativeModel('gemini-2.5-flash-lite')

        prompt = f"""
        Проанализируй описание комплекта для курьера: "{deal_description}".
        Это может быть электровелосипед, аккумуляторы или и то, и другое.
        Примерная рыночная стоимость такого комплекта около 80,000 - 120,000 рублей.

        Твоя задача - сгенерировать 3-4 варианта плана рассрочки (выкупа) для этого комплекта.
        Верни результат СТРОГО в формате JSON. Ответ должен быть только чистым JSON объектом,
        без лишних слов, комментариев или ```json ``` оберток.

        Ключами в JSON должны быть короткие идентификаторы (например, "plan_1", "plan_2"),
        а значениями - словари со следующими ключами:
        - "label": Короткое и понятное описание для кнопки (например, "3 мес / 16000 ₽").
        - "full_label": Полное описание для подтверждения (например, "3 месяца: 7 платежей по 16 000 ₽").
        - "first_payment": Сумма первого взноса (число).
        - "total_payments": Общее количество платежей (число).
        - "period_days": Периодичность платежей в днях (30 для месяца, 14 для 2 недель).

        Пример твоего идеального ответа:
        {{
            "plan_1": {{
                "label": "3 мес / 16000 ₽",
                "full_label": "3 месяца: 7 платежей по 16 000 ₽ (раз в 2 недели)",
                "first_payment": 16000,
                "total_payments": 7,
                "period_days": 14
            }},
            "plan_2": {{
                "label": "4 мес / 21000 ₽",
                "full_label": "4 месяца: 5 платежей по 21 000 ₽ (раз в месяц)",
                "first_payment": 21000,
                "total_payments": 5,
                "period_days": 30
            }}
        }}
        """

        response = model.generate_content(prompt, stream=False)
        response.resolve()
        
        cleaned_response_text = response.text.strip().replace("`", "").lstrip("json").strip()

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
# API ЭНДПОИНТЫ
# ==============================================================================

# Старый эндпоинт для документов (без изменений)
@app.post("/recognize-documents")
async def recognize_documents(request: RecognizeDocsRequest):
    # ... (код этой функции остается без изменений)
    logger.info(f"Получен запрос на распознавание для страны: {request.country}. Количество изображений: {len(request.images_base64)}")
    if not request.images_base64:
        raise HTTPException(status_code=400, detail="Не переданы изображения для распознавания.")
    try:
        images_for_gemini = []
        for b64_string in request.images_base64:
            image_bytes = base64.b64decode(b64_string)
            image = Image.open(BytesIO(image_bytes))
            images_for_gemini.append(image)
        if request.country == 'РФ':
            prompt = """
            Проанализируй эти изображения: основной разворот паспорта РФ и страница с пропиской.
            Извлеки все данные и верни их в виде ОДНОГО плоского JSON объекта.
            Ключи: "Фамилия", "Имя", "Отчество", "Дата рождения", "Серия и номер паспорта", "Кем выдан", "Дата выдачи", "Адрес регистрации".
            Если поле не найдено, значение должно быть пустой строкой.
            Ответ должен быть только чистым JSON.
            """
        else:
            prompt = """
            Проанализируй эти изображения: паспорт иностранного гражданина, регистрация в РФ, патент.
            Извлеки все данные и верни их в виде ОДНОГО плоского JSON объекта.
            Ключи: "ФИО", "Гражданство", "Дата рождения", "Номер паспорта", "Адрес регистрации в РФ", "Номер патента".
            Если поле не найдено, значение должно быть пустой строкой.
            Ответ должен быть только чистым JSON.
            """
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content([prompt] + images_for_gemini, request_options={"timeout": 120})
        response.resolve()
        cleaned_text = response.text.strip().replace("`", "").lstrip("json").strip()
        recognized_data = json.loads(cleaned_text)
        if not isinstance(recognized_data, dict):
            raise ValueError("Gemini не вернул корректный JSON-объект.")
        logger.info(f"Успешное распознавание. Возвращаю данные: {recognized_data}")
        return recognized_data
    except Exception as e:
        logger.error(f"Ошибка при обработке документов с Gemini: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера Gemini: {str(e)}")


# НОВЫЙ ЭНДПОИНТ для генерации планов выкупа
@app.post("/generate-buyout-plans")
async def api_generate_buyout_plans(request: BuyoutPlanRequest):
    logger.info(f"Получен запрос на /generate-buyout-plans для: {request.deal_description}")
    
    plans = get_buyout_plans_with_gemini(request.deal_description)
    
    if plans:
        return plans
    else:
        raise HTTPException(
            status_code=400,
            detail="Не удалось сгенерировать план выкупа. Попробуйте описать его по-другому."
        )

# Корневой эндпоинт (без изменений)
@app.get("/")
async def root():
    return {"status": "Gemini Universal API Gateway is running"}