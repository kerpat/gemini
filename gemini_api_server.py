import os
import json
import logging
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from PIL import Image

# --- Настройка ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI(title="Gemini Document Recognition API")

# --- Настройка Gemini API ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("Ключ GEMINI_API_KEY не найден в .env")
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API успешно настроен на сервере.")
except Exception as e:
    logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА при настройке Gemini API: {e}")

# --- Модель данных для входящего запроса ---
class RecognizeDocsRequest(BaseModel):
    images_base64: list[str]  # Список фотографий, закодированных в Base64
    country: str              # Страна пользователя ('РФ' или 'Другая')

# --- Эндпоинт API ---
@app.post("/recognize-documents")
async def recognize_documents(request: RecognizeDocsRequest):
    logger.info(f"Получен запрос на распознавание для страны: {request.country}. Количество изображений: {len(request.images_base64)}")

    if not request.images_base64:
        raise HTTPException(status_code=400, detail="Не переданы изображения для распознавания.")

    try:
        # 1. Декодируем Base64 обратно в изображения
        images_for_gemini = []
        for b64_string in request.images_base64:
            image_bytes = base64.b64decode(b64_string)
            image = Image.open(BytesIO(image_bytes))
            images_for_gemini.append(image)

        # 2. Выбираем правильный промпт в зависимости от страны
        if request.country == 'РФ':
            prompt = """
            Проанализируй эти изображения: основной разворот паспорта РФ и страница с пропиской.
            Извлеки все данные и верни их в виде ОДНОГО плоского JSON объекта.
            Ключи: "Фамилия", "Имя", "Отчество", "Дата рождения", "Серия и номер паспорта", "Кем выдан", "Дата выдачи", "Адрес регистрации".
            Если поле не найдено, значение должно быть пустой строкой.
            Ответ должен быть только чистым JSON.
            """
        else: # Для иностранных граждан
            prompt = """
            Проанализируй эти изображения: паспорт иностранного гражданина, регистрация в РФ, патент.
            Извлеки все данные и верни их в виде ОДНОГО плоского JSON объекта.
            Ключи: "ФИО", "Гражданство", "Дата рождения", "Номер паспорта", "Адрес регистрации в РФ", "Номер патента".
            Если поле не найдено, значение должно быть пустой строкой.
            Ответ должен быть только чистым JSON.
            """
        
        # 3. Отправляем запрос в Gemini
        model = genai.GenerativeModel('gemini-2.5-flash-lie')
        response = model.generate_content([prompt] + images_for_gemini, request_options={"timeout": 120})
        response.resolve()

        # 4. Обрабатываем и возвращаем результат
        cleaned_text = response.text.strip().replace("`", "").lstrip("json").strip()
        recognized_data = json.loads(cleaned_text)

        if not isinstance(recognized_data, dict):
            raise ValueError("Gemini не вернул корректный JSON-объект.")

        logger.info(f"Успешное распознавание. Возвращаю данные: {recognized_data}")
        return recognized_data

    except Exception as e:
        logger.error(f"Ошибка при обработке документов с Gemini: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера Gemini: {str(e)}")

@app.get("/")
async def root():
    return {"status": "Gemini Document Recognition API is running"}