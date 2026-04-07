import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Пайплайн извлечения признаков из текстового описания.
    Воспроизводит логику из ноутбука HahatonML_Feature__engineering.ipynb.

    На выходе: DataFrame с 138 колонками:
    - 38 ручных признаков (лексические, структурные, доменные)
    - 100 TF-IDF признаков
    """

    # Параметры векторизатора (должны совпадать с обучением!)
    TFIDF_PARAMS = {
        'max_features': 100,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95,
        'lowercase': True,
        'strip_accents': 'unicode'
    }

    def __init__(self, tfidf_vectorizer_path: Optional[str] = None):
        """
        Инициализация пайплайна.

        :param tfidf_vectorizer_path: Путь к сохранённому векторизатору .joblib
        """
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None

        if tfidf_vectorizer_path:
            self._load_vectorizer(tfidf_vectorizer_path)
        else:
            # Инициализация для обучения (не для инференса!)
            self.tfidf_vectorizer = TfidfVectorizer(**self.TFIDF_PARAMS)
            logger.warning("⚠️ Векторизатор инициализирован без fit() — только для обучения!")

    def _load_vectorizer(self, path: str):
        """Загрузка предобученного векторизатора"""
        try:
            self.tfidf_vectorizer = joblib.load(path)
            vocab_size = len(self.tfidf_vectorizer.vocabulary_)
            logger.info("✅ Векторизатор загружен: %d токенов в словаре", vocab_size)
        except Exception as exc:
            logger.error("❌ Ошибка загрузки векторизатора из %s: %s", path, exc)
            raise

    def extract_split_indicators(self, text: str) -> Dict[str, int]:
        """Извлекает маркеры возможности разделения работы"""
        text = str(text).lower()

        # Маркеры разделения (высокая вероятность should_split=True)
        split_patterns = [
            r'отдельно.*отдельно', r'также отдельно', r'при необходимости отдельно',
            r'можем отдельно', r'берем отдельно', r'выполняем отдельно',
            r'делаем отдельно', r'можно заказать отдельно', r'беру как самостоятельную работу',
        ]

        # Маркеры комплекса (высокая вероятность should_split=False)
        complex_patterns = [
            r'все работы одной бригадой', r'в составе ремонта', r'в рамках комплекса',
            r'комплексом', r'под ключ без дробления', r'по отдельным видам работ не выезжаю',
            r'ищу заказы именно на комплекс', r'выполняем как часть ремонта', r'выполняем в составе ремонта',
        ]

        features = {}
        features['split_indicators_count'] = sum(
            1 for pattern in split_patterns if re.search(pattern, text, re.IGNORECASE)
        )
        features['complex_indicators_count'] = sum(
            1 for pattern in complex_patterns if re.search(pattern, text, re.IGNORECASE)
        )
        features['has_multiple_otdelno'] = 1 if re.search(r'отдельно.*отдельно', text) else 0
        features['has_ne_vyezzhayu'] = 1 if 'не выезжаю' in text else 0
        features['has_odnoy_brigradoy'] = 1 if 'одной бригадой' in text else 0

        return features

    def extract_work_types_and_locations(self, text: str) -> Dict[str, int]:
        """Извлекает типы работ и помещения"""
        text = str(text).lower()

        # Типы работ
        work_types = {
            'demolition': ['демонтаж', 'снос', 'разбор', 'сбивка', 'снятие'],
            'plumbing': ['сантехник', 'труб', 'вода', 'канализация', 'унитаз', 'ванн', 'раковин'],
            'electrical': ['электрик', 'проводк', 'розетк', 'выключател', 'светильник', 'щит'],
            'plastering': ['штукатур', 'шпаклев', 'выравнивани'],
            'painting': ['покраск', 'окраск', 'колеровк'],
            'wallpaper': ['обои', 'оклейк'],
            'flooring': ['пол', 'ламинат', 'линолеум', 'плитк', 'ковров'],
            'ceilings': ['потолок', 'гкл', 'гипсокартон', 'натяжн'],
            'tiling': ['плитк', 'кафель', 'керамогранит', 'мозаик'],
        }

        # Помещения
        locations = {
            'kitchen': ['кухн', 'кухня'],
            'bathroom': ['санузел', 'ванн', 'туалет'],
            'room': ['комнат', 'спальн', 'детск'],
            'corridor': ['коридор', 'прихож'],
            'office': ['офис', 'коммерческ'],
            'cottage': ['коттедж', 'дом'],
            'newbuilding': ['новостройк'],
            'secondary': ['вторичк'],
        }

        features = {}

        # Считаем упоминания типов работ
        for work_type, keywords in work_types.items():
            features[f'has_{work_type}'] = sum(1 for kw in keywords if kw in text)

        # Считаем упоминания помещений
        for loc, keywords in locations.items():
            features[f'has_{loc}'] = sum(1 for kw in keywords if kw in text)

        # Агрегированные признаки
        features['work_types_count'] = sum(
            1 for k, v in features.items()
            if k.startswith('has_') and v > 0 and 'location' not in k
        )
        features['locations_count'] = sum(
            1 for k, v in features.items()
            if k.startswith('has_') and v > 0 and 'location' in k
        )

        return features

    def extract_text_structure_features(self, text: str, split_count: int = 0) -> Dict[str, Any]:
        """Извлекает признаки структуры текста"""
        text = str(text)

        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_bullets': 1 if any(x in text for x in ['-', '•', '●', '▪']) else 0,
            'has_numbers': 1 if re.search(r'\d+', text) else 0,
            'comma_count': text.count(','),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'split_density': split_count / len(text) if len(text) > 0 else 0
        }

        return features

    def extract_domain_specific_features(self, text: str) -> Dict[str, int]:
        """Извлекает специфичные доменные признаки для ремонта"""
        text = str(text).lower()

        features = {}

        # Ключевые фразы для разделения
        features['has_kitchen_bathroom_split'] = 1 if (
                ('кухн' in text or 'кухня' in text) and
                ('санузел' in text or 'ванн' in text) and
                ('отдельно' in text)
        ) else 0

        # Фразы "комплекс" vs "отдельно"
        features['complex_to_split_ratio'] = (
                                                     text.count('комплекс') + text.count('комплексом') + 1
                                             ) / (text.count('отдельно') + 1)

        # Наличие конкретных работ в контексте разделения
        features['has_specific_work_with_otdelno'] = 1 if re.search(
            r'(демонтаж|штукатур|электрик|сантехник|плитк|обои).*отдельно|отдельно.*(демонтаж|штукатур|электрик|сантехник|плитк|обои)',
            text
        ) else 0

        # Маркеры "под ключ"
        features['has_pod_kluch'] = 1 if 'под ключ' in text else 0
        features['has_pod_kluch_bez_drobleniya'] = 1 if 'без дробления' in text else 0

        # Опыт и гарантии
        features['has_experience'] = 1 if any(x in text for x in ['опыт', 'лет', 'с 2016']) else 0
        features['has_guarantee'] = 1 if 'гаранти' in text else 0

        return features

    def transform(self, descriptions: List[str]) -> pd.DataFrame:
        """
        Преобразует список текстов в DataFrame признаков.

        :param descriptions: Список текстовых описаний
        :return: DataFrame с 138 колонками признаков
        """
        if not descriptions:
            raise ValueError("Список описаний не может быть пустым")

        all_features = []

        for idx, text in enumerate(descriptions):
            # Предобработка: заполнение пропусков
            text = str(text).strip() if pd.notna(text) else ''

            try:
                # Извлечение признаков
                split_feat = self.extract_split_indicators(text)
                work_feat = self.extract_work_types_and_locations(text)
                struct_feat = self.extract_text_structure_features(
                    text,
                    split_feat.get('split_indicators_count', 0)
                )
                domain_feat = self.extract_domain_specific_features(text)

                # Объединение всех ручных признаков (38 признаков)
                row_features = {**split_feat, **work_feat, **struct_feat, **domain_feat}
                all_features.append(row_features)

            except Exception as exc:
                logger.error("Ошибка при извлечении признаков для текста #%d: %s", idx, exc)
                raise

        # Создание DataFrame из ручных признаков
        df_features = pd.DataFrame(all_features)

        # TF-IDF векторизация (100 признаков)
        if self.tfidf_vectorizer is not None:
            try:
                # Используем тот же векторизатор, что и при обучении
                tfidf_matrix = self.tfidf_vectorizer.transform(descriptions)
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                )

                # Конкатенация: 38 ручных + 100 TF-IDF = 138 признаков
                df_features = pd.concat(
                    [df_features.reset_index(drop=True), tfidf_df.reset_index(drop=True)],
                    axis=1
                )

            except Exception as exc:
                logger.error("Ошибка TF-IDF векторизации: %s", exc)
                raise
        else:
            logger.warning("⚠️ Векторизатор не загружен — пропущен этап TF-IDF")

        # Валидация: должно быть ровно 138 признаков
        expected_features = 138
        if df_features.shape[1] != expected_features:
            logger.error(
                "❌ Несоответствие количества признаков: ожидалось %d, получено %d",
                expected_features, df_features.shape[1]
            )
            raise ValueError(
                f"Ожидалось {expected_features} признаков, получено {df_features.shape[1]}"
            )

        return df_features