import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Any, List

class FeaturePipeline:
    def __init__(self, tfidf_vectorizer_path: str = None):
        self.tfidf_vectorizer = None
        if tfidf_vectorizer_path:
            self.tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        else:
            # Инициализация для обучения (не для инференса)
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=100,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )

    def extract_split_indicators(self, text: str) -> Dict[str, int]:
        text = str(text).lower()
        split_patterns = [
            r'отдельно.*отдельно', r'также отдельно', r'при необходимости отдельно',
            r'можем отдельно', r'берем отдельно', r'выполняем отдельно',
            r'делаем отдельно', r'можно заказать отдельно', r'беру как самостоятельную работу',
        ]
        complex_patterns = [
            r'все работы одной бригадой', r'в составе ремонта', r'в рамках комплекса',
            r'комплексом', r'под ключ без дробления', r'по отдельным видам работ не выезжаю',
            r'ищу заказы именно на комплекс', r'выполняем как часть ремонта', r'выполняем в составе ремонта',
        ]
        features = {}
        features['split_indicators_count'] = sum(1 for pattern in split_patterns if re.search(pattern, text))
        features['complex_indicators_count'] = sum(1 for pattern in complex_patterns if re.search(pattern, text))
        features['has_multiple_otdelno'] = 1 if re.search(r'отдельно.*отдельно', text) else 0
        features['has_ne_vyezzhayu'] = 1 if 'не выезжаю' in text else 0
        features['has_odnoy_brigradoy'] = 1 if 'одной бригадой' in text else 0
        return features

    def extract_work_types_and_locations(self, text: str) -> Dict[str, int]:
        text = str(text).lower()
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
        locations = {
            'kitchen': ['кухн', 'кухня'], 'bathroom': ['санузел', 'ванн', 'туалет'],
            'room': ['комнат', 'спальн', 'детск'], 'corridor': ['коридор', 'прихож'],
            'office': ['офис', 'коммерческ'], 'cottage': ['коттедж', 'дом'],
            'newbuilding': ['новостройк'], 'secondary': ['вторичк'],
        }
        features = {}
        for work_type, keywords in work_types.items():
            features[f'has_{work_type}'] = sum(1 for kw in keywords if kw in text)
        for loc, keywords in locations.items():
            features[f'has_{loc}'] = sum(1 for kw in keywords if kw in text)
        
        features['work_types_count'] = sum(1 for k, v in features.items() if k.startswith('has_') and v > 0 and 'location' not in k)
        features['locations_count'] = sum(1 for k, v in features.items() if k.startswith('has_') and v > 0 and 'location' in k)
        return features

    def extract_text_structure_features(self, text: str, split_count: int = 0) -> Dict[str, Any]:
        text = str(text)
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_bullets': 1 if any(x in text for x in ['-', '•', '•']) else 0,
            'has_numbers': 1 if re.search(r'\d+', text) else 0,
            'comma_count': text.count(','),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'split_density': split_count / len(text) if len(text) > 0 else 0
        }
        return features

    def extract_domain_specific_features(self, text: str) -> Dict[str, int]:
        text = str(text).lower()
        features = {}
        features['has_kitchen_bathroom_split'] = 1 if (('кухн' in text or 'кухня' in text) and ('санузел' in text or 'ванн' in text) and ('отдельно' in text)) else 0
        features['complex_to_split_ratio'] = (text.count('комплекс') + text.count('комплексом') + 1) / (text.count('отдельно') + 1)
        features['has_specific_work_with_otdelno'] = 1 if re.search(r'(демонтаж|штукатур|электрик|сантехник|плитк|обои).*отдельно|отдельно.*(демонтаж|штукатур|электрик|сантехник|плитк|обои)', text) else 0
        features['has_pod_kluch'] = 1 if 'под ключ' in text else 0
        features['has_pod_kluch_bez_drobleniya'] = 1 if 'без дробления' in text else 0
        features['has_experience'] = 1 if any(x in text for x in ['опыт', 'лет', 'с 2016']) else 0
        features['has_guarantee'] = 1 if 'гаранти' in text else 0
        return features

    def transform(self, descriptions: List[str]) -> pd.DataFrame:
        """Преобразует список текстов в DataFrame признаков."""
        all_features = []
        for text in descriptions:
            split_feat = self.extract_split_indicators(text)
            work_feat = self.extract_work_types_and_locations(text)
            struct_feat = self.extract_text_structure_features(text, split_feat.get('split_indicators_count', 0))
            domain_feat = self.extract_domain_specific_features(text)
            
            row_features = {**split_feat, **work_feat, **struct_feat, **domain_feat}
            all_features.append(row_features)
        
        df_features = pd.DataFrame(all_features)
        
        # TF-IDF векторизация
        if self.tfidf_vectorizer is not None:
            tfidf_matrix = self.tfidf_vectorizer.transform(descriptions)
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
            df_features = pd.concat([df_features.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        
        return df_features