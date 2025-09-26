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

# ==============================================================================
# КОПИРУЕМ ВАШУ ФУНКЦИЮ РАСПОЗНАВАНИЯ ПРЯМО СЮДА (с небольшими изменениями)
# ==============================================================================
def recognize_documents_with_gemini(images: list, country: str) -> dict | None:
    """
    Основная логика распознавания документов, перенесенная на сервер.
    Принимает уже готовые объекты изображений PIL.Image.
    """
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        # Используем самую последнюю и надежную версию модели
        model = genai.GenerativeModel('gemini-1.5-flash-latest', safety_settings=safety_settings)

        if country == 'РФ':
            prompt = """
            Проанализируй эти три изображения:
            1. Основной разворот паспорта гражданина РФ.
            2. Страница с последней действительной пропиской.
            3. Селфи человека, держащего основной разворот паспорта.

            Твои задачи:
            1.  Извлеки ВСЕ данные из документов. Обрати особое внимание на правильное написание ФИО, дат и названий ведомств.
            2.  Сравни лицо на селфи с фотографией в паспорте и оцени их схожесть.
            3.  Верни результат СТРОГО в виде ОДНОГО JSON объекта. Ответ должен быть только чистым JSON, без слов "json", "```" или любых других комментариев.

            Ключи в JSON должны быть следующими:
            - "lastName": Фамилия.
            - "firstName": Имя.
            - "patronymic": Отчество (если есть, иначе null).
            - "birthDate": Дата рождения (в формате ДД.ММ.ГГГГ).
            - "birthPlace": Место рождения.
            - "passportSeries": Серия паспорта (4 цифры).
            - "passportNumber": Номер паспорта (6 цифр).
            - "issuedBy": Кем выдан паспорт (полностью).
            - "issueDate": Дата выдачи (в формате ДД.ММ.ГГГГ).
            - "departmentCode": Код подразделения.
            - "registrationAddress": Полный адрес регистрации из штампа.
            - "selfieMatch": Оценка схожести лиц на селфи и в паспорте ("Да", "Нет", "Неуверенно").

            Если какое-либо поле не удалось распознать, его значение должно быть null.
            """
        else: # Для иностранных граждан
            prompt = """
            Проанализируй эти четыре изображения:
            1. Основной разворот паспорта иностранного гражданина.
            2. Документ о временной регистрации на территории РФ.
            3. Патент на работу (если есть).
            4. Селфи человека, держащего основной разворот паспорта.

            Твои задачи:
            1.  Извлеки ВСЕ данные из документов. ФИО пиши латиницей, как в документе.
            2.  Сравни лицо на селфи с фотографией в паспорте.
            3.  Верни результат СТРОГО в виде ОДНОГО JSON объекта. Ответ должен быть только чистым JSON, без лишних слов или комментариев.

            Ключи в JSON должны быть следующими:
            - "fullNameLatin": Полное имя латиницей, как в документе.
            - "citizenship": Гражданство (страна).
            - "birthDate": Дата рождения (в формате ДД.ММ.ГГГГ).
            - "passportNumber": Номер паспорта.
            - "passportExpiryDate": Дата окончания срока действия паспорта (в формате ДД.ММ.ГГГГ).
            - "registrationAddress": Адрес временной регистрации в РФ.
            - "registrationExpiryDate": Срок окончания регистрации в РФ (в формате ДД.ММ.ГГГГ).
            - "patentNumber": Номер патента на работу (если есть, иначе null).
            - "patentExpiryDate": Срок действия патента (если есть, иначе null).
            - "selfieMatch": Оценка схожести лиц ("Да", "Нет", "Неуверенно").

            Если какое-либо поле не удалось распознать, его значение должно быть null.
            """
        
        response = model.generate_content([prompt] + images, request_options={"timeout": 120})
        response.resolve()
        
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        logger.info(f"Ответ от Gemini (распознавание документов): {cleaned_text}")
        
        recognized_data = json.loads(cleaned_text)
        return recognized_data if isinstance(recognized_data, dict) else None

    except Exception as e:
        logger.error(f"Критическая ошибка при распознавании документов с Gemini: {e}", exc_info=True)
        return None

# ==============================================================================
# API ЭНДПОИНТЫ
# ==============================================================================

@app.post("/recognize-documents")
async def api_recognize_documents(request: RecognizeDocsRequest):
    logger.info(f"Получен запрос на распознавание для страны: {request.country}. Количество изображений: {len(request.images_base64)}")

    if not request.images_base64:
        raise HTTPException(status_code=400, detail="Не переданы изображения для распознавания.")

    try:
        images_for_gemini = [Image.open(BytesIO(base64.b64decode(b64))) for b64 in request.images_base64]
        
        recognized_data = recognize_documents_with_gemini(images_for_gemini, request.country)

        if recognized_data:
            logger.info(f"Успешное распознавание. Возвращаю данные: {recognized_data}")
            return recognized_data
        else:
            raise HTTPException(status_code=500, detail="Gemini не смог обработать изображения или вернул некорректный формат.")

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса /recognize-documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.get("/")
async def root():
    return {"status": "Gemini Document Recognition API is running"}