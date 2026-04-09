import os
import requests
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class YandexGPTLiteGenerator:
    """YandexGPT Lite через REST API (без OpenAI клиента)"""

    def __init__(self):
        # Получаем данные из переменных окружения
        self.api_key = os.getenv("YANDEX_API_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")

        if not self.api_key:
            raise ValueError("❌ Необходимо указать YANDEX_API_KEY")
        if not self.folder_id:
            raise ValueError("❌ Необходимо указать YANDEX_FOLDER_ID")

        # URL и заголовки для API
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }

        # ID модели
        self.model_uri = f"gpt://{self.folder_id}/yandexgpt-lite"

        # Настройки генерации
        self.completion_options = {
            "stream": False,
            "temperature": 0.7,
            "maxTokens": "2000"
        }

        # Системный промпт
        self.system_prompt = (
            "Ты профессиональный копирайтер, специализирующийся на создании "
            "объявлений для Avito в категории 'Ремонт и строительство'. "
            "Твои тексты должны быть продающими, грамотными и структурированными."
        )

        logger.info(f"✅ YandexGPT Lite инициализирован (folder: {self.folder_id[:10]}...)")

    def generate(self, description: str, should_split: bool, probability: float) -> str:
        """
        Генерация черновика объявления

        :param description: Описание от пользователя
        :param should_split: Флаг раздельного ремонта
        :param probability: Вероятность предсказания
        :return: Сгенерированный текст
        """

        # Определяем тип ремонта
        repair_type = "отдельные виды работ" if should_split else "комплексный ремонт под ключ"

        # Формируем пользовательский промпт
        user_prompt = f"""Сгенерируй черновик объявления для Avito.

Информация от заказчика:
{description}

Тип услуги: {repair_type}
Уверенность классификации: {probability:.0%}

Напиши в следующем формате:

Заголовок: [привлекательное название услуги]
Описание: [2-3 предложения с ключевыми деталями]
Преимущества: [3-4 преимущества через запятую]"""

        # Формируем тело запроса
        payload = {
            "modelUri": self.model_uri,
            "completionOptions": self.completion_options,
            "messages": [
                {
                    "role": "system",
                    "text": self.system_prompt
                },
                {
                    "role": "user",
                    "text": user_prompt
                }
            ]
        }

        try:
            logger.debug(f"📤 Отправка запроса к YandexGPT...")

            response = requests.post(
                self.url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            # Проверяем статус ответа
            if response.status_code != 200:
                logger.error(f"❌ Ошибка API: {response.status_code}")
                logger.error(f"Ответ: {response.text}")
                raise Exception(f"API вернул ошибку {response.status_code}: {response.text}")

            # Парсим ответ
            result = response.json()

            # Извлекаем текст из ответа
            generated_text = result["result"]["alternatives"][0]["message"]["text"]

            logger.info(f"✅ Текст успешно сгенерирован (длина: {len(generated_text)} символов)")

            return generated_text

        except requests.exceptions.Timeout:
            logger.error("❌ Таймаут запроса к YandexGPT")
            return self._get_fallback_response(should_split)

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Сетевая ошибка: {str(e)}")
            return self._get_fallback_response(should_split)

        except KeyError as e:
            logger.error(f"❌ Неожиданный формат ответа: {str(e)}")
            logger.error(f"Полный ответ: {response.text}")
            return self._get_fallback_response(should_split)

        except Exception as e:
            logger.error(f"❌ Неизвестная ошибка: {str(e)}")
            return self._get_fallback_response(should_split)

    def generate_structured(self, description: str, should_split: bool, probability: float) -> Dict[str, str]:
        """
        Генерация с возвратом структурированного ответа

        :return: {"title": "...", "description": "...", "advantages": "..."}
        """
        text = self.generate(description, should_split, probability)

        # Парсим ответ
        result = {
            "title": "",
            "description": "",
            "advantages": "",
            "raw_text": text
        }

        lines = text.split('\n')
        current_key = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            lower_line = line.lower()

            if lower_line.startswith('заголовок:'):
                result['title'] = self._clean_line(line, 'заголовок:')
                current_key = 'title'
            elif lower_line.startswith('описание:'):
                result['description'] = self._clean_line(line, 'описание:')
                current_key = 'description'
            elif lower_line.startswith('преимущества:'):
                result['advantages'] = self._clean_line(line, 'преимущества:')
                current_key = 'advantages'
            elif current_key and result[current_key]:
                # Добавляем продолжение текста
                result[current_key] += ' ' + line

        return result

    def _clean_line(self, line: str, prefix: str) -> str:
        """Очистка строки от префикса"""
        # Удаляем префикс в любом регистре
        for p in [prefix, prefix.capitalize(), prefix.upper()]:
            if line.startswith(p):
                return line.replace(p, '').strip()
        return line.strip()

    def _get_fallback_response(self, should_split: bool) -> str:
        """
        Запасной ответ в случае ошибки API
        """
        if should_split:
            return """Заголовок: Отделочные работы любой сложности - качественно и в срок
Описание: Выполняю отдельные виды ремонтных и отделочных работ. Большой опыт, профессиональный инструмент, аккуратность и ответственность гарантирую.
Преимущества: Бесплатный выезд на замер, Работа по договору, Соблюдение сроков, Гарантия 12 месяцев"""
        else:
            return """Заголовок: Комплексный ремонт квартир и домов под ключ с гарантией
Описание: Выполняем полный комплекс ремонтно-отделочных работ. Одна бригада, фиксированная смета, поэтапная оплата. Работаем с 2016 года.
Преимущества: Все работы под ключ, Бесплатная консультация и смета, Гарантия 2 года, Чистота и порядок на объекте"""

    def set_temperature(self, temperature: float):
        """Изменить температуру генерации (0.0 - 1.0)"""
        self.completion_options["temperature"] = max(0.0, min(1.0, temperature))
        logger.info(f"🌡️ Температура установлена: {temperature}")

    def set_max_tokens(self, max_tokens: int):
        """Изменить максимальное количество токенов"""
        self.completion_options["maxTokens"] = str(max_tokens)
        logger.info(f"📏 Максимальное количество токенов: {max_tokens}")


# Пример использования
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # Создаем генератор
    generator = YandexGPTLiteGenerator()

    # Тестовый запрос
    test_description = "Нужно поклеить обои в комнате 18 квадратных метров, обои уже куплены"

    print("=" * 50)
    print("ТЕСТОВАЯ ГЕНЕРАЦИЯ")
    print("=" * 50)

    result = generator.generate(
        description=test_description,
        should_split=True,
        probability=0.85
    )

    print(result)
    print("\n" + "=" * 50)

    # Структурированный ответ
    structured = generator.generate_structured(
        description=test_description,
        should_split=True,
        probability=0.85
    )

    print("\nСТРУКТУРИРОВАННЫЙ ОТВЕТ:")
    print(f"Заголовок: {structured['title']}")
    print(f"Описание: {structured['description']}")
    print(f"Преимущества: {structured['advantages']}")