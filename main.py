import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# NLP libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import re
import string

# For NewsAPI
import os
from newsapi import NewsApiClient

class EnhancedStockPricePredictor:
    """
    Улучшенный класс для прогнозирования цен акций с использованием
    технических индикаторов и анализа тональности новостей.
    """
    
    def __init__(self, ticker):
        """
        Инициализация предиктора цен акций
        
        Args:
            ticker (str): Тикер акции для прогнозирования
        """
        self.ticker = ticker
        self.price_data = None
        self.news_data = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        # Инициализация VADER для базового анализа тональности
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.vader = SentimentIntensityAnalyzer()
        
        # Инициализация более сложной модели для анализа тональности
        try:
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_sentiment = pipeline("sentiment-analysis", model=self.finbert_model, tokenizer=self.finbert_tokenizer)
            self.nlp_available = True
        except:
            print("Предупреждение: Не удалось загрузить FinBERT модель. Будет использован только VADER.")
            self.nlp_available = False
    
    def fetch_price_data(self, start_date, end_date):
        
        print(f"Загрузка данных о ценах {self.ticker} с {start_date} по {end_date}...")
        
        try:
            self.price_data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            
            if self.price_data.empty:
                raise ValueError(f"Не удалось загрузить данные для {self.ticker}")
                
            print(f"Загружено {len(self.price_data)} записей о ценах")
            
            # Проверка на наличие пропущенных значений
            if self.price_data.isnull().sum().sum() > 0:
                print(f"Обнаружено {self.price_data.isnull().sum().sum()} пропущенных значений")
                # Заполнение простым методом (можно улучшить)
                self.price_data.fillna(method='ffill', inplace=True)
                self.price_data.fillna(method='bfill', inplace=True)
            
            return self.price_data
            
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return None
    
    def fetch_news_data(self, source_type='csv', **kwargs):
       
        if source_type == 'csv':
            return self._fetch_news_from_csv(kwargs.get('csv_path'))
        elif source_type == 'newsapi':
            return self._fetch_news_from_newsapi(
                kwargs.get('api_key'),
                kwargs.get('start_date'),
                kwargs.get('end_date')
            )
        elif source_type == 'rss':
            return self._fetch_news_from_rss(kwargs.get('rss_urls', []))
        else:
            print(f"Неизвестный тип источника: {source_type}")
            return None
    
    def _fetch_news_from_csv(self, csv_path):
      
        try:
            news_data = pd.read_csv(csv_path)
            
            # Предполагается, что файл содержит колонки 'date', 'title', 'text' и, возможно, 'ticker'
            required_columns = ['date']
            if not all(col in news_data.columns for col in required_columns):
                print(f"Предупреждение: CSV файл должен содержать колонку 'date' и хотя бы одну из: 'title', 'text'")
            
            # Фильтрация по тикеру (если необходимо)
            if 'ticker' in news_data.columns:
                news_data = news_data[news_data['ticker'] == self.ticker]
                print(f"Отфильтровано {len(news_data)} новостей для {self.ticker}")
            
            # Преобразование даты в datetime
            if 'date' in news_data.columns:
                news_data['date'] = pd.to_datetime(news_data['date'])
                news_data.set_index('date', inplace=True)
                news_data.sort_index(inplace=True)
            
            self.news_data = news_data
            print(f"Загружено {len(news_data)} новостных записей")
            
            # Обработка новостных данных
            self._process_news_data()
            
            return self.news_data
            
        except Exception as e:
            print(f"Ошибка при загрузке новостных данных из CSV: {e}")
            return None
    
    def _fetch_news_from_newsapi(self, api_key, start_date, end_date):
      
        try:
            newsapi = NewsApiClient(api_key=api_key)
            
            # Разбиваем на периоды по 30 дней из-за ограничений API
            all_articles = []
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            current = start
            while current < end:
                next_date = min(current + timedelta(days=30), end)
                
                print(f"Загрузка новостей с {current.strftime('%Y-%m-%d')} по {next_date.strftime('%Y-%m-%d')}...")
                
                # Запрос новостей по компании
                articles = newsapi.get_everything(
                    q=self.ticker,  # Поиск по тикеру
                    from_param=current.strftime('%Y-%m-%d'),
                    to=next_date.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='publishedAt',
                    page_size=100
                )
                
                if articles['status'] == 'ok':
                    all_articles.extend(articles['articles'])
                
                current = next_date
            
            # Преобразование в DataFrame
            if all_articles:
                news_df = pd.DataFrame(all_articles)
                news_df['date'] = pd.to_datetime(news_df['publishedAt'])
                news_df.rename(columns={'title': 'title', 'description': 'text'}, inplace=True)
                news_df.set_index('date', inplace=True)
                news_df.sort_index(inplace=True)
                
                self.news_data = news_df
                print(f"Загружено {len(news_df)} новостных записей через NewsAPI")
                
                # Обработка новостных данных
                self._process_news_data()
                
                return self.news_data
            else:
                print("Не найдено новостей по заданным параметрам")
                return None
            
        except Exception as e:
            print(f"Ошибка при загрузке новостей через NewsAPI: {e}")
            return None
    
    def _fetch_news_from_rss(self, rss_urls):
       
        try:
            import feedparser
            
            all_entries = []
            
            for url in rss_urls:
                print(f"Загрузка новостей из RSS: {url}")
                feed = feedparser.parse(url)
                
                for entry in feed.entries:
                    # Проверяем, содержит ли новость упоминание о компании
                    if self.ticker in entry.title or self.ticker in entry.summary:
                        all_entries.append({
                            'title': entry.title,
                            'text': entry.summary,
                            'date': entry.published if 'published' in entry else datetime.now().strftime('%Y-%m-%d'),
                            'link': entry.link
                        })
            
            if all_entries:
                news_df = pd.DataFrame(all_entries)
                news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
                # Если не удалось распознать дату, используем текущую
                news_df['date'].fillna(pd.Timestamp.now(), inplace=True)
                news_df.set_index('date', inplace=True)
                news_df.sort_index(inplace=True)
                
                self.news_data = news_df
                print(f"Загружено {len(news_df)} новостных записей из RSS фидов")
                
                # Обработка новостных данных
                self._process_news_data()
                
                return self.news_data
            else:
                print("Не найдено релевантных новостей в RSS фидах")
                return None
            
        except Exception as e:
            print(f"Ошибка при загрузке новостей из RSS: {e}")
            return None
    
    def _clean_text(self, text):
    
        if not isinstance(text, str):
            return ""
        
        # Удаление URL
        text = re.sub(r'http\S+', '', text)
        # Удаление HTML тегов
        text = re.sub(r'<.*?>', '', text)
        # Удаление специальных символов
        text = re.sub(r'[^\w\s]', '', text)
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _analyze_sentiment_vader(self, text):
     
        text = self._clean_text(text)
        if not text:
            return 0.0
        
        return self.vader.polarity_scores(text)['compound']
    
    def _analyze_sentiment_finbert(self, text):
        if not self.nlp_available or not isinstance(text, str) or not text:
            return 0.0
        
        text = self._clean_text(text)
        if not text:
            return 0.0
        
        # FinBERT может обрабатывать только определенное количество токенов
        if len(text) > 512:
            text = text[:512]
        
        try:
            # Получение оценки тональности
            result = self.finbert_sentiment(text)[0]
            label = result['label']
            score = result['score']
            
            # Преобразование оценки к шкале [-1, 1]
            if label == 'positive':
                return score
            elif label == 'negative':
                return -score
            else:  # neutral
                return 0.0
        except Exception as e:
            print(f"Ошибка при анализе тональности с FinBERT: {e}")
            return 0.0
    
    def _process_news_data(self):
        if self.news_data is None:
            print("Нет новостных данных для обработки")
            return None
        
        print("Обработка новостных данных и анализ тональности...")
        
        # Анализ тональности для заголовков
        if 'title' in self.news_data.columns:
            self.news_data['title_sentiment_vader'] = self.news_data['title'].apply(self._analyze_sentiment_vader)
            
            if self.nlp_available:
                # Применяем более сложную модель к подвыборке заголовков (для скорости)
                sample_size = min(100, len(self.news_data))
                sample_idx = np.random.choice(self.news_data.index, size=sample_size, replace=False)
                
                # Инициализируем колонку
                self.news_data['title_sentiment_finbert'] = 0.0
                
                # Применяем FinBERT только к подвыборке
                for idx in tqdm(sample_idx, desc="Анализ тональности заголовков"):
                    title = self.news_data.loc[idx, 'title']
                    sentiment = self._analyze_sentiment_finbert(title)
                    self.news_data.loc[idx, 'title_sentiment_finbert'] = sentiment
        
        # Анализ тональности для текста
        if 'text' in self.news_data.columns:
            self.news_data['text_sentiment_vader'] = self.news_data['text'].apply(self._analyze_sentiment_vader)
            
            if self.nlp_available:
                # Аналогично для текстов
                sample_size = min(50, len(self.news_data))  # Меньше для текста, т.к. он длиннее
                sample_idx = np.random.choice(self.news_data.index, size=sample_size, replace=False)
                
                # Инициализируем колонку
                self.news_data['text_sentiment_finbert'] = 0.0
                
                # Применяем FinBERT только к подвыборке
                for idx in tqdm(sample_idx, desc="Анализ тональности текстов"):
                    text = self.news_data.loc[idx, 'text']
                    sentiment = self._analyze_sentiment_finbert(text)
                    self.news_data.loc[idx, 'text_sentiment_finbert'] = sentiment
        
        # Агрегирование новостей по дням
        sentiment_columns = [col for col in self.news_data.columns if 'sentiment' in col]
        if sentiment_columns:
            # Создаем новый DataFrame для агрегированных данных
            daily_sentiment = self.news_data[sentiment_columns].resample('D').agg(['mean', 'std', 'count'])
            
            # Упрощаем мультииндекс
            daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
            
            # Соединение с ценовыми данными, если они есть
            if self.price_data is not None:
                self.price_data = self.price_data.join(daily_sentiment)
                
                # Заполнение пропущенных значений
                for col in daily_sentiment.columns:
                    if col in self.price_data.columns:
                        # Для средних значений используем предыдущие значения
                        if 'mean' in col:
                            self.price_data[col].fillna(method='ffill', inplace=True)
                            self.price_data[col].fillna(0, inplace=True)
                        # Для стандартного отклонения используем медиану
                        elif 'std' in col:
                            median_std = self.price_data[col].median()
                            self.price_data[col].fillna(median_std, inplace=True)
                        # Для количества новостей используем 0
                        elif 'count' in col:
                            self.price_data[col].fillna(0, inplace=True)
        
        return self.news_data
    
    def prepare_features(self, window_size=10, prediction_horizon=1):
        if self.price_data is None:
            raise ValueError("Необходимо сначала загрузить данные с помощью fetch_price_data()")
        
        print("Подготовка признаков для модели...")
        df = self.price_data.copy()
        
        # ---------- 1. Базовые признаки ----------
        
        # Логарифмические доходности (более стационарны, чем абсолютные цены)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['log_return_squared'] = df['log_return'] ** 2  # Для оценки волатильности
        
        # Объемные индикаторы
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma10'] = df['Volume'].rolling(window=10).mean()
        df['relative_volume'] = df['Volume'] / df['volume_ma10']
        
        # High-Low диапазон (волатильность внутри дня)
        df['hl_pct'] = (df['High'] - df['Low']) / df['Close']
        
        # ---------- 2. Технические индикаторы ----------
        
        # Скользящие средние (разные периоды)
        for period in [5, 10, 20, 50, 100]:
            df[f'ma_{period}'] = df['Close'].rolling(window=period).mean()
            # Расстояние до скользящей средней (%)
            df[f'ma_{period}_dist'] = (df['Close'] / df[f'ma_{period}'] - 1) * 100
        
        # Экспоненциальные скользящие средние
        for period in [5, 12, 26]:
            df[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_slope'] = df['macd'].diff(3)  # Изменение за 3 дня
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        df['bb_std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['Close'] * 100  # ATR в процентах от цены
        
        df['ad_line'] = ((2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'])) * df['Volume']
        df['ad_line'] = df['ad_line'].cumsum()
        df['chaikin_osc'] = df['ad_line'].ewm(span=3).mean() - df['ad_line'].ewm(span=10).mean()
        
        
        df['day_of_week'] = df.index.dayofweek  # 0=Понедельник, 6=Воскресенье
        df['month'] = df.index.month
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        df['quarter'] = df.index.quarter
        
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)  # 5 рабочих дней
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Дни до праздников/выходных (упрощенно)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # ---------- 4. Лаговые признаки ----------
        
        # Лаги доходности
        for i in range(1, window_size + 1):
            df[f'log_return_lag_{i}'] = df['log_return'].shift(i)
            df[f'volume_change_lag_{i}'] = df['volume_change'].shift(i)
        
        # Скользящие статистики по доходности
        for period in [5, 10, 20]:
            df[f'return_mean_{period}'] = df['log_return'].rolling(window=period).mean()
            df[f'return_std_{period}'] = df['log_return'].rolling(window=period).std()
            df[f'return_skew_{period}'] = df['log_return'].rolling(window=period).skew()
        

        df[f'target_return_{prediction_horizon}d'] = df['log_return'].shift(-prediction_horizon)
        
        # Удаляем строки с NaN (из-за смещений)
        df.dropna(inplace=True)
        
        # Разделяем признаки и целевую переменную
        y = df[f'target_return_{prediction_horizon}d']
        
        # Исключаем из признаков колонки, которые не должны использоваться для обучения
        exclude_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        exclude_columns.extend([col for col in df.columns if 'target' in col])
        
        # Список признаков для обучения модели
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        X = df[feature_columns]
        
        # Сохраняем список признаков для последующего использования
        self.feature_columns = feature_columns
        
        print(f"Подготовлено {len(X)} записей с {len(feature_columns)} признаками")
        
        return X, y, df
   
    def train_model(self, X, y, test_size=0.2, random_state=42, n_splits=5):
    print("Обучение моделей и оценка качества...")
    
    # Масштабирование признаков
    X_scaled = self.scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    # Разделение данных для временной кросс-валидации
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Определение моделей для обучения
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.001),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=random_state, learning_rate=0.1),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=random_state)
    }
    
    # Словари для хранения результатов
    results = {}
    metrics = {}
    
    # Обучение и оценка каждой модели
    for name, model in models.items():
        print(f"\nОбучение модели: {name}")
        
        # Инициализация списков для метрик
        cv_mae, cv_rmse, cv_mape, cv_r2 = [], [], [], []
        
        # Временная кросс-валидация
        for train_idx, test_idx in tqdm(tscv.split(X_scaled), desc=f"Кросс-валидация {name}", total=n_splits):
            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Обучение модели
            model.fit(X_train, y_train)
            
            # Прогноз
            y_pred = model.predict(X_test)
            
            # Расчет метрик
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(np.exp(y_test) - 1, np.exp(y_pred) - 1) * 100
            r2 = r2_score(y_test, y_pred)
            
            cv_mae.append(mae)
            cv_rmse.append(rmse)
            cv_mape.append(mape)
            cv_r2.append(r2)
        
        # Среднее значение метрик по всем фолдам
        metrics[name] = {
            'MAE': np.mean(cv_mae),
            'RMSE': np.mean(cv_rmse),
            'MAPE': np.mean(cv_mape),
            'R²': np.mean(cv_r2)
        }
        
        # Обучение модели на всем наборе данных
        model.fit(X_scaled, y)
        results[name] = model
        
        print(f"  MAE: {metrics[name]['MAE']:.6f}, RMSE: {metrics[name]['RMSE']:.6f}, MAPE: {metrics[name]['MAPE']:.2f}%, R²: {metrics[name]['R²']:.4f}")
    
    # Определение лучшей модели по RMSE
    best_model_name = min(metrics, key=lambda k: metrics[k]['RMSE'])
    self.best_model = results[best_model_name]
    self.best_model_name = best_model_name
    self.models = results
    
    print(f"\nЛучшая модель: {best_model_name} с RMSE = {metrics[best_model_name]['RMSE']:.6f}")
    
    return results, metrics

def evaluate_feature_importance(self, X, y, model=None, top_n=20):
    if model is None:
        if self.best_model is None:
            raise ValueError("Необходимо сначала обучить модель с помощью train_model()")
        model = self.best_model
    
    print(f"Оценка важности признаков для модели {self.best_model_name}...")
    
    # Получение важности признаков в зависимости от типа модели
    feature_importance = None
    
    if hasattr(model, 'feature_importances_'):
        # Для моделей на основе деревьев
        feature_importance = pd.DataFrame({
            'Признак': X.columns,
            'Важность': model.feature_importances_
        })
    elif hasattr(model, 'coef_'):
        # Для линейных моделей
        feature_importance = pd.DataFrame({
            'Признак': X.columns,
            'Важность': np.abs(model.coef_)
        })
    else:
        # Пермутационная важность признаков для остальных моделей
        from sklearn.inspection import permutation_importance
        
        print("Использование пермутационной важности признаков...")
        
        # Масштабирование признаков
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        # Расчет пермутационной важности
        perm_importance = permutation_importance(model, X_scaled, y, n_repeats=10, random_state=42)
        
        feature_importance = pd.DataFrame({
            'Признак': X.columns,
            'Важность': perm_importance.importances_mean
        })
    
    # Сортировка по убыванию важности
    feature_importance = feature_importance.sort_values('Важность', ascending=False).reset_index(drop=True)
    
    # Отображение топ-N признаков
    print(f"Топ-{top_n} важных признаков:")
    display_features = feature_importance.head(top_n).copy()
    
    # Нормализация для упрощения интерпретации
    display_features['Важность'] = display_features['Важность'] / display_features['Важность'].max() * 100
    
    for i, row in display_features.iterrows():
        print(f"{i+1}. {row['Признак']}: {row['Важность']:.2f}%")
    
    return feature_importance

def visualize_results(self, true_values, predictions, sentiment_feature=None, title=None):
    print("Визуализация результатов прогнозирования...")
    
    # Создаем DataFrame с результатами
    results_df = pd.DataFrame({
        'Факт': true_values,
        'Прогноз': predictions
    }, index=true_values.index)
    
    # Создаем подграфики
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title or f"Результаты прогнозирования {self.ticker}", fontsize=16)
    
    # 1. График сравнения фактических и прогнозных значений
    axes[0, 0].plot(results_df.index, results_df['Факт'], label='Фактические значения', color='blue')
    axes[0, 0].plot(results_df.index, results_df['Прогноз'], label='Прогноз', color='red', linestyle='--')
    axes[0, 0].set_title('Сравнение фактических и прогнозных значений доходности')
    axes[0, 0].set_xlabel('Дата')
    axes[0, 0].set_ylabel('Логарифмическая доходность')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Диаграмма рассеяния факт vs прогноз
    axes[0, 1].scatter(results_df['Факт'], results_df['Прогноз'], alpha=0.5)
    axes[0, 1].set_title('Диаграмма рассеяния: Факт vs Прогноз')
    axes[0, 1].set_xlabel('Фактическая доходность')
    axes[0, 1].set_ylabel('Прогнозная доходность')
    
    # Добавление линии идеального прогноза
    min_val = min(results_df['Факт'].min(), results_df['Прогноз'].min())
    max_val = max(results_df['Факт'].max(), results_df['Прогноз'].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Идеальный прогноз')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Распределение ошибок
    error = results_df['Факт'] - results_df['Прогноз']
    axes[1, 0].hist(error, bins=30, color='green', alpha=0.7)
    axes[1, 0].set_title('Распределение ошибок прогноза')
    axes[1, 0].set_xlabel('Ошибка (Факт - Прогноз)')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].grid(True)
    
    # 4. Дополнительный график в зависимости от наличия данных о тональности
    if sentiment_feature and sentiment_feature in self.price_data.columns:
        sentiment_data = self.price_data[sentiment_feature].reindex(results_df.index)
        
        # Для цветовой карты
        color_map = np.zeros(len(results_df))
        color_map[sentiment_data > 0] = 1  # Положительная тональность
        color_map[sentiment_data < 0] = -1  # Отрицательная тональность
        
        scatter = axes[1, 1].scatter(
            results_df['Факт'], 
            results_df['Прогноз'], 
            c=color_map, 
            cmap='RdYlGn',
            alpha=0.7,
            s=50
        )
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Негативная тональность'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Нейтральная тональность'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Позитивная тональность')
        ]
        
        axes[1, 1].legend(handles=legend_elements)
        axes[1, 1].set_title('Прогноз vs Факт с учетом тональности новостей')
        axes[1, 1].set_xlabel('Фактическая доходность')
        axes[1, 1].set_ylabel('Прогнозная доходность')
        axes[1, 1].grid(True)
    else:
        # Если нет данных о тональности, то показываем автокорреляцию ошибок
        from statsmodels.graphics.tsaplots import plot_acf
        
        plot_acf(error, lags=20, ax=axes[1, 1])
        axes[1, 1].set_title('Автокорреляция ошибок прогноза')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig, axes

def predict_future(self, horizon=5):
    if self.best_model is None:
        raise ValueError("Необходимо сначала обучить модель с помощью train_model()")
    
    if self.feature_columns is None or len(self.feature_columns) == 0:
        raise ValueError("Список признаков не определен. Сначала запустите prepare_features()")
    
    print(f"Прогнозирование цен на следующие {horizon} дней для {self.ticker}...")
    
    # Копия последних данных для прогнозирования
    last_data = self.price_data.copy().tail(30)  # используем последние 30 дней для контекста
    future_dates = pd.date_range(start=last_data.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='B')
    
    # Создаем пустой DataFrame для будущих дат
    future_df = pd.DataFrame(index=future_dates)
    
    # Инициализируем прогнозы
    predictions = []
    predicted_prices = []
    last_price = float(self.price_data['Close'].iloc[-1])
    
    for i in range(horizon):
        # Подготовка признаков для текущего дня
        if i == 0:
            X_future = last_data.iloc[-1:][self.feature_columns].copy()
        else:
            X_future = future_df.iloc[-1:][self.feature_columns].copy()
        
        X_future_scaled = self.scaler.transform(X_future)
        
        log_return_pred = self.best_model.predict(X_future_scaled)[0]
        predictions.append(log_return_pred)
        
        predicted_price = last_price * np.exp(log_return_pred)
        predicted_prices.append(predicted_price)
        last_price = predicted_price
        
        future_df.loc[future_dates[i], 'Predicted_Return'] = log_return_pred
        future_df.loc[future_dates[i], 'Predicted_Price'] = predicted_price
        
       
    forecast_df = pd.DataFrame({
        'Дата': future_dates,
        'Прогнозная доходность': predictions,
        'Прогнозная цена': predicted_prices
    })
    
    print(f"Прогноз на {horizon} дней:")
    print(forecast_df)
    
    return forecast_df

def analyze_news_impact(self, X, y, sentiment_columns):
    print("Анализ влияния новостных данных на качество прогноза...")
    
    # Проверка наличия сентимент-колонок в данных
    available_sentiment_cols = [col for col in sentiment_columns if col in X.columns]
    
    if not available_sentiment_cols:
        print("Отсутствуют колонки с сентиментом в данных!")
        return None
    
    print(f"Найдено {len(available_sentiment_cols)} колонок с сентиментом: {available_sentiment_cols}")
    
    # Создаем копии данных для сравнения
    X_with_news = X.copy()
    X_without_news = X.drop(columns=available_sentiment_cols)
    
    # Масштабирование признаков
    X_with_news_scaled = self.scaler.fit_transform(X_with_news)
    X_with_news_scaled = pd.DataFrame(X_with_news_scaled, index=X_with_news.index, columns=X_with_news.columns)
    
    scaler_without_news = StandardScaler()
    X_without_news_scaled = scaler_without_news.fit_transform(X_without_news)
    X_without_news_scaled = pd.DataFrame(X_without_news_scaled, index=X_without_news.index, columns=X_without_news.columns)
    
    # Временная кросс-валидация
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Модель для оценки
    if self.best_model_name is None:
        model_class = XGBRegressor
        model_params = {'n_estimators': 100, 'random_state': 42, 'learning_rate': 0.1}
    else:
        # Используем тот же класс модели, что и для лучшей модели
        model_class = self.best_model.__class__
        model_params = self.best_model.get_params()
    
    # Метрики для сравнения
    metrics_with_news = {'MAE': [], 'RMSE': [], 'R²': []}
    metrics_without_news = {'MAE': [], 'RMSE': [], 'R²': []}
    
    # Оценка с использованием временной кросс-валидации
    for train_idx, test_idx in tqdm(tscv.split(X_with_news_scaled), desc="Оценка влияния новостей", total=5):
        # Данные с новостями
        X_train_with = X_with_news_scaled.iloc[train_idx]
        X_test_with = X_with_news_scaled.iloc[test_idx]
        
        # Данные без новостей
        X_train_without = X_without_news_scaled.iloc[train_idx]
        X_test_without = X_without_news_scaled.iloc[test_idx]
        
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Обучение моделей
        model_with_news = model_class(**model_params)
        model_without_news = model_class(**model_params)
        
        model_with_news.fit(X_train_with, y_train)
        model_without_news.fit(X_train_without, y_train)
        
        # Прогнозирование
        y_pred_with = model_with_news.predict(X_test_with)
        y_pred_without = model_without_news.predict(X_test_without)
        
        # Расчет метрик
        metrics_with_news['MAE'].append(mean_absolute_error(y_test, y_pred_with))
        metrics_with_news['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred_with)))
        metrics_with_news['R²'].append(r2_score(y_test, y_pred_with))
        
        metrics_without_news['MAE'].append(mean_absolute_error(y_test, y_pred_without))
        metrics_without_news['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred_without)))
        metrics_without_news['R²'].append(r2_score(y_test, y_pred_without))
    
    # Усреднение метрик
    mean_metrics_with = {k: np.mean(v) for k, v in metrics_with_news.items()}
    mean_metrics_without = {k: np.mean(v) for k, v in metrics_without_news.items()}
    
    # Процентное улучшение
    improvement = {}
    for metric in ['MAE', 'RMSE']:
        improvement[metric] = ((mean_metrics_without[metric] - mean_metrics_with[metric]) / mean_metrics_without[metric]) * 100
    
    improvement['R²'] = ((mean_metrics_with['R²'] - mean_metrics_without['R²']) / abs(mean_metrics_without['R²'])) * 100
    
    # Вывод результатов
    print("\nСравнение метрик качества:")
    print(f"{'Метрика':<10} {'С новостями':<15} {'Без новостей':<15} {'Улучшение (%)':<15}")
    print('-' * 55)
    
    for metric in ['MAE', 'RMSE', 'R²']:
        print(f"{metric:<10} {mean_metrics_with[metric]:<15.6f} {mean_metrics_without[metric]:<15.6f} {improvement[metric]:<15.2f}")
    
    # Анализ корреляций между тональностью и доходностью
    correlations = {}
    for col in available_sentiment_cols:
        correlations[col] = y.corr(X[col])
    
    print("\nКорреляция между сентиментом и целевой переменной:")
    for col, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{col:<30}: {corr:.4f}")
    
    return {
        'metrics_with_news': mean_metrics_with,
        'metrics_without_news': mean_metrics_without,
        'improvement': improvement,
        'correlations': correlations
    }

def save_model(self, filepath):
    if self.best_model is None:
        raise ValueError("Нет обученной модели для сохранения")
    
    import joblib
    
    # Создание словаря с данными модели и метаданными
    model_data = {
        'model': self.best_model,
        'model_name': self.best_model_name,
        'scaler': self.scaler,
        'feature_columns': self.feature_columns,
        'ticker': self.ticker,
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Сохранение модели
    joblib.dump(model_data, filepath)
    print(f"Модель успешно сохранена в {filepath}")

    def train_model(self, X, y, test_size=0.2, random_state=42, n_splits=5):
    print("Обучение моделей и оценка качества...")
    
    # Масштабирование признаков
    X_scaled = self.scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    # Разделение данных для временной кросс-валидации
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Определение моделей для обучения
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.001),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=random_state, learning_rate=0.1),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=random_state)
    }
    
    # Словари для хранения результатов
    results = {}
    metrics = {}
    
    # Обучение и оценка каждой модели
    for name, model in models.items():
        print(f"\nОбучение модели: {name}")
        
        # Инициализация списков для метрик
        cv_mae, cv_rmse, cv_mape, cv_r2 = [], [], [], []
        
        # Временная кросс-валидация
        for train_idx, test_idx in tqdm(tscv.split(X_scaled), desc=f"Кросс-валидация {name}", total=n_splits):
            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Обучение модели
            model.fit(X_train, y_train)
            
            # Прогноз
            y_pred = model.predict(X_test)
            
            # Расчет метрик
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(np.exp(y_test) - 1, np.exp(y_pred) - 1) * 100
            r2 = r2_score(y_test, y_pred)
            
            cv_mae.append(mae)
            cv_rmse.append(rmse)
            cv_mape.append(mape)
            cv_r2.append(r2)
        
        # Среднее значение метрик по всем фолдам
        metrics[name] = {
            'MAE': np.mean(cv_mae),
            'RMSE': np.mean(cv_rmse),
            'MAPE': np.mean(cv_mape),
            'R²': np.mean(cv_r2)
        }
        
        # Обучение модели на всем наборе данных
        model.fit(X_scaled, y)
        results[name] = model
        
        print(f"  MAE: {metrics[name]['MAE']:.6f}, RMSE: {metrics[name]['RMSE']:.6f}, MAPE: {metrics[name]['MAPE']:.2f}%, R²: {metrics[name]['R²']:.4f}")
    
    # Определение лучшей модели по RMSE
    best_model_name = min(metrics, key=lambda k: metrics[k]['RMSE'])
    self.best_model = results[best_model_name]
    self.best_model_name = best_model_name
    self.models = results
    
    print(f"\nЛучшая модель: {best_model_name} с RMSE = {metrics[best_model_name]['RMSE']:.6f}")
    
    return results, metrics

def evaluate_feature_importance(self, X, y, model=None, top_n=20):

    if model is None:
        if self.best_model is None:
            raise ValueError("Необходимо сначала обучить модель с помощью train_model()")
        model = self.best_model
    
    print(f"Оценка важности признаков для модели {self.best_model_name}...")
    
    # Получение важности признаков в зависимости от типа модели
    feature_importance = None
    
    if hasattr(model, 'feature_importances_'):
        # Для моделей на основе деревьев
        feature_importance = pd.DataFrame({
            'Признак': X.columns,
            'Важность': model.feature_importances_
        })
    elif hasattr(model, 'coef_'):
        # Для линейных моделей
        feature_importance = pd.DataFrame({
            'Признак': X.columns,
            'Важность': np.abs(model.coef_)
        })
    else:
        # Пермутационная важность признаков для остальных моделей
        from sklearn.inspection import permutation_importance
        
        print("Использование пермутационной важности признаков...")
        
        # Масштабирование признаков
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        # Расчет пермутационной важности
        perm_importance = permutation_importance(model, X_scaled, y, n_repeats=10, random_state=42)
        
        feature_importance = pd.DataFrame({
            'Признак': X.columns,
            'Важность': perm_importance.importances_mean
        })
    
    # Сортировка по убыванию важности
    feature_importance = feature_importance.sort_values('Важность', ascending=False).reset_index(drop=True)
    
    # Отображение топ-N признаков
    print(f"Топ-{top_n} важных признаков:")
    display_features = feature_importance.head(top_n).copy()
    
    # Нормализация для упрощения интерпретации
    display_features['Важность'] = display_features['Важность'] / display_features['Важность'].max() * 100
    
    for i, row in display_features.iterrows():
        print(f"{i+1}. {row['Признак']}: {row['Важность']:.2f}%")
    
    return feature_importance

def visualize_results(self, true_values, predictions, sentiment_feature=None, title=None):
    print("Визуализация результатов прогнозирования...")
    
    # Создаем DataFrame с результатами
    results_df = pd.DataFrame({
        'Факт': true_values,
        'Прогноз': predictions
    }, index=true_values.index)
    
    # Создаем подграфики
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title or f"Результаты прогнозирования {self.ticker}", fontsize=16)
    
    # 1. График сравнения фактических и прогнозных значений
    axes[0, 0].plot(results_df.index, results_df['Факт'], label='Фактические значения', color='blue')
    axes[0, 0].plot(results_df.index, results_df['Прогноз'], label='Прогноз', color='red', linestyle='--')
    axes[0, 0].set_title('Сравнение фактических и прогнозных значений доходности')
    axes[0, 0].set_xlabel('Дата')
    axes[0, 0].set_ylabel('Логарифмическая доходность')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Диаграмма рассеяния факт vs прогноз
    axes[0, 1].scatter(results_df['Факт'], results_df['Прогноз'], alpha=0.5)
    axes[0, 1].set_title('Диаграмма рассеяния: Факт vs Прогноз')
    axes[0, 1].set_xlabel('Фактическая доходность')
    axes[0, 1].set_ylabel('Прогнозная доходность')
    
    # Добавление линии идеального прогноза
    min_val = min(results_df['Факт'].min(), results_df['Прогноз'].min())
    max_val = max(results_df['Факт'].max(), results_df['Прогноз'].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Идеальный прогноз')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Распределение ошибок
    error = results_df['Факт'] - results_df['Прогноз']
    axes[1, 0].hist(error, bins=30, color='green', alpha=0.7)
    axes[1, 0].set_title('Распределение ошибок прогноза')
    axes[1, 0].set_xlabel('Ошибка (Факт - Прогноз)')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].grid(True)
    
    # 4. Дополнительный график в зависимости от наличия данных о тональности
    if sentiment_feature and sentiment_feature in self.price_data.columns:
        sentiment_data = self.price_data[sentiment_feature].reindex(results_df.index)
        
        # Для цветовой карты
        color_map = np.zeros(len(results_df))
        color_map[sentiment_data > 0] = 1  # Положительная тональность
        color_map[sentiment_data < 0] = -1  # Отрицательная тональность
        
        scatter = axes[1, 1].scatter(
            results_df['Факт'], 
            results_df['Прогноз'], 
            c=color_map, 
            cmap='RdYlGn',
            alpha=0.7,
            s=50
        )
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Негативная тональность'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Нейтральная тональность'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Позитивная тональность')
        ]
        
        axes[1, 1].legend(handles=legend_elements)
        axes[1, 1].set_title('Прогноз vs Факт с учетом тональности новостей')
        axes[1, 1].set_xlabel('Фактическая доходность')
        axes[1, 1].set_ylabel('Прогнозная доходность')
        axes[1, 1].grid(True)
    else:
        # Если нет данных о тональности, то показываем автокорреляцию ошибок
        from statsmodels.graphics.tsaplots import plot_acf
        
        plot_acf(error, lags=20, ax=axes[1, 1])
        axes[1, 1].set_title('Автокорреляция ошибок прогноза')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig, axes

def predict_future(self, horizon=5):
    if self.best_model is None:
        raise ValueError("Необходимо сначала обучить модель с помощью train_model()")
    
    if self.feature_columns is None or len(self.feature_columns) == 0:
        raise ValueError("Список признаков не определен. Сначала запустите prepare_features()")
    
    print(f"Прогнозирование цен на следующие {horizon} дней для {self.ticker}...")
    
    # Копия последних данных для прогнозирования
    last_data = self.price_data.copy().tail(30)  # используем последние 30 дней для контекста
    future_dates = pd.date_range(start=last_data.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='B')
    
    # Создаем пустой DataFrame для будущих дат
    future_df = pd.DataFrame(index=future_dates)
    
    # Инициализируем прогнозы
    predictions = []
    predicted_prices = []
    last_price = float(self.price_data['Close'].iloc[-1])
    
    for i in range(horizon):
        # Подготовка признаков для текущего дня
        if i == 0:
            # Для первого дня используем исторические данные
            X_future = last_data.iloc[-1:][self.feature_columns].copy()
        else:
            X_future = future_df.iloc[-1:][self.feature_columns].copy()
        
        # Масштабирование признаков
        X_future_scaled = self.scaler.transform(X_future)
        
        # Прогнозирование логарифмической доходности
        log_return_pred = self.best_model.predict(X_future_scaled)[0]
        predictions.append(log_return_pred)
        
        # Расчет прогнозной цены
        predicted_price = last_price * np.exp(log_return_pred)
        predicted_prices.append(predicted_price)
        last_price = predicted_price
        
        future_df.loc[future_dates[i], 'Predicted_Return'] = log_return_pred
        future_df.loc[future_dates[i], 'Predicted_Price'] = predicted_price

    forecast_df = pd.DataFrame({
        'Дата': future_dates,
        'Прогнозная доходность': predictions,
        'Прогнозная цена': predicted_prices
    })
    
    print(f"Прогноз на {horizon} дней:")
    print(forecast_df)
    
    return forecast_df

def analyze_news_impact(self, X, y, sentiment_columns):
    print("Анализ влияния новостных данных на качество прогноза...")
    
    # Проверка наличия сентимент-колонок в данных
    available_sentiment_cols = [col for col in sentiment_columns if col in X.columns]
    
    if not available_sentiment_cols:
        print("Отсутствуют колонки с сентиментом в данных!")
        return None
    
    print(f"Найдено {len(available_sentiment_cols)} колонок с сентиментом: {available_sentiment_cols}")
    
    # Создаем копии данных для сравнения
    X_with_news = X.copy()
    X_without_news = X.drop(columns=available_sentiment_cols)
    
    # Масштабирование признаков
    X_with_news_scaled = self.scaler.fit_transform(X_with_news)
    X_with_news_scaled = pd.DataFrame(X_with_news_scaled, index=X_with_news.index, columns=X_with_news.columns)
    
    scaler_without_news = StandardScaler()
    X_without_news_scaled = scaler_without_news.fit_transform(X_without_news)
    X_without_news_scaled = pd.DataFrame(X_without_news_scaled, index=X_without_news.index, columns=X_without_news.columns)
    
    # Временная кросс-валидация
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Модель для оценки
    if self.best_model_name is None:
        model_class = XGBRegressor
        model_params = {'n_estimators': 100, 'random_state': 42, 'learning_rate': 0.1}
    else:
        # Используем тот же класс модели, что и для лучшей модели
        model_class = self.best_model.__class__
        model_params = self.best_model.get_params()
    
    # Метрики для сравнения
    metrics_with_news = {'MAE': [], 'RMSE': [], 'R²': []}
    metrics_without_news = {'MAE': [], 'RMSE': [], 'R²': []}
    
    # Оценка с использованием временной кросс-валидации
    for train_idx, test_idx in tqdm(tscv.split(X_with_news_scaled), desc="Оценка влияния новостей", total=5):
        # Данные с новостями
        X_train_with = X_with_news_scaled.iloc[train_idx]
        X_test_with = X_with_news_scaled.iloc[test_idx]
        
        # Данные без новостей
        X_train_without = X_without_news_scaled.iloc[train_idx]
        X_test_without = X_without_news_scaled.iloc[test_idx]
        
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Обучение моделей
        model_with_news = model_class(**model_params)
        model_without_news = model_class(**model_params)
        
        model_with_news.fit(X_train_with, y_train)
        model_without_news.fit(X_train_without, y_train)
        
        # Прогнозирование
        y_pred_with = model_with_news.predict(X_test_with)
        y_pred_without = model_without_news.predict(X_test_without)
        
        # Расчет метрик
        metrics_with_news['MAE'].append(mean_absolute_error(y_test, y_pred_with))
        metrics_with_news['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred_with)))
        metrics_with_news['R²'].append(r2_score(y_test, y_pred_with))
        
        metrics_without_news['MAE'].append(mean_absolute_error(y_test, y_pred_without))
        metrics_without_news['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred_without)))
        metrics_without_news['R²'].append(r2_score(y_test, y_pred_without))
    
    # Усреднение метрик
    mean_metrics_with = {k: np.mean(v) for k, v in metrics_with_news.items()}
    mean_metrics_without = {k: np.mean(v) for k, v in metrics_without_news.items()}
    
    # Процентное улучшение
    improvement = {}
    for metric in ['MAE', 'RMSE']:
        improvement[metric] = ((mean_metrics_without[metric] - mean_metrics_with[metric]) / mean_metrics_without[metric]) * 100
    
    improvement['R²'] = ((mean_metrics_with['R²'] - mean_metrics_without['R²']) / abs(mean_metrics_without['R²'])) * 100
    
    # Вывод результатов
    print("\nСравнение метрик качества:")
    print(f"{'Метрика':<10} {'С новостями':<15} {'Без новостей':<15} {'Улучшение (%)':<15}")
    print('-' * 55)
    
    for metric in ['MAE', 'RMSE', 'R²']:
        print(f"{metric:<10} {mean_metrics_with[metric]:<15.6f} {mean_metrics_without[metric]:<15.6f} {improvement[metric]:<15.2f}")
    
    # Анализ корреляций между тональностью и доходностью
    correlations = {}
    for col in available_sentiment_cols:
        correlations[col] = y.corr(X[col])
    
    print("\nКорреляция между сентиментом и целевой переменной:")
    for col, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{col:<30}: {corr:.4f}")
    
    return {
        'metrics_with_news': mean_metrics_with,
        'metrics_without_news': mean_metrics_without,
        'improvement': improvement,
        'correlations': correlations
    }

def save_model(self, filepath):
    """
    Сохранение обученной модели
    
    Args:
        filepath (str): Путь для сохранения модели
    """
    if self.best_model is None:
        raise ValueError("Нет обученной модели для сохранения")
    
    import joblib
    
    # Создание словаря с данными модели и метаданными
    model_data = {
        'model': self.best_model,
        'model_name': self.best_model_name,
        'scaler': self.scaler,
        'feature_columns': self.feature_columns,
        'ticker': self.ticker,
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Сохранение модели
    joblib.dump(model_data, filepath)
    print(f"Модель успешно сохранена в {filepath}")

    def load_model(self, filepath):
    """
    Загрузка ранее сохраненной модели
    
    Args:
        filepath (str): Путь к файлу с сохраненной моделью
    """
    import joblib
    
    try:
        # Загрузка данных модели
        model_data = joblib.load(filepath)
        
        # Обновление атрибутов объекта
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        
        # Проверка соответствия тикера
        if self.ticker != model_data['ticker']:
            print(f"Предупреждение: Модель обучена для {model_data['ticker']}, "
                  f"а используется для {self.ticker}.")
        
        print(f"Модель успешно загружена из {filepath}")
        print(f"Модель: {self.best_model_name}, сохранена: {model_data['saved_at']}")
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")

def backtesting(self, test_data, initial_capital=10000, transaction_cost=0.001):
    """
    Бэктестинг торговой стратегии на исторических данных
    
    Args:
        test_data (pd.DataFrame): Тестовые данные с прогнозами и реальными ценами
        initial_capital (float): Начальный капитал для бэктестинга
        transaction_cost (float): Комиссия за сделку (в % от суммы)
        
    Returns:
        pd.DataFrame: Результаты бэктестинга с метриками эффективности
    """
    print("Проведение бэктестинга торговой стратегии...")
    
    # Проверка наличия необходимых данных
    if 'Close' not in test_data.columns or 'Прогноз' not in test_data.columns:
        raise ValueError("Данные для бэктестинга должны содержать столбцы 'Close' и 'Прогноз'")
    
    # Создаем копию данных для бэктестинга
    backtest_df = test_data[['Close', 'Прогноз']].copy()
    
    # Определение торговых сигналов
    backtest_df['Signal'] = 0  # 0 - держать, 1 - купить, -1 - продать
    backtest_df['Signal'] = np.where(backtest_df['Прогноз'] > 0.001, 1, backtest_df['Signal'])  # Позитивный прогноз
    backtest_df['Signal'] = np.where(backtest_df['Прогноз'] < -0.001, -1, backtest_df['Signal'])  # Негативный прогноз
    
    # Инициализация счетчиков
    capital = initial_capital
    shares_owned = 0
    trades = []
    portfolio_values = []
    
    # Проход по историческим данным
    for i in range(len(backtest_df)):
        date = backtest_df.index[i]
        price = backtest_df['Close'].iloc[i]
        signal = backtest_df['Signal'].iloc[i]
        
        portfolio_value_before = capital + shares_owned * price
        
        # Обработка сигналов
        if signal == 1 and shares_owned == 0:  # Покупка
            max_shares = int(capital / price)  # Максимальное кол-во акций, которое можно купить
            if max_shares > 0:
                shares_to_buy = max_shares
                cost = shares_to_buy * price
                commission = cost * transaction_cost
                total_cost = cost + commission
                
                if total_cost <= capital:
                    capital -= total_cost
                    shares_owned = shares_to_buy
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares_to_buy,
                        'value': cost,
                        'commission': commission,
                        'capital': capital
                    })
        
        elif signal == -1 and shares_owned > 0:  # Продажа
            value = shares_owned * price
            commission = value * transaction_cost
            capital += value - commission
            
            trades.append({
                'date': date,
                'action': 'SELL',
                'price': price,
                'shares': shares_owned,
                'value': value,
                'commission': commission,
                'capital': capital
            })
            
            shares_owned = 0
        
        # Запись текущей стоимости портфеля
        portfolio_value = capital + shares_owned * price
        portfolio_values.append({
            'date': date,
            'cash': capital,
            'shares': shares_owned,
            'stock_value': shares_owned * price,
            'total_value': portfolio_value,
            'return': (portfolio_value / portfolio_value_before - 1) if i > 0 else 0
        })
    
    # Принудительная продажа всех акций в конце периода
    if shares_owned > 0:
        final_price = backtest_df['Close'].iloc[-1]
        value = shares_owned * final_price
        commission = value * transaction_cost
        capital += value - commission
        
        trades.append({
            'date': backtest_df.index[-1],
            'action': 'SELL (Final)',
            'price': final_price,
            'shares': shares_owned,
            'value': value,
            'commission': commission,
            'capital': capital
        })
    
    # Создание DataFrame с результатами
    trades_df = pd.DataFrame(trades)
    portfolio_df = pd.DataFrame(portfolio_values)
    
    # Расчет метрик эффективности
    total_return = (capital / initial_capital - 1) * 100
    
    # Расчет доходности стратегии "Купи и держи"
    buy_and_hold_return = ((backtest_df['Close'].iloc[-1] / backtest_df['Close'].iloc[0]) - 1) * 100
    
    # Расчет метрик риска
    if len(portfolio_df) > 1:
        returns = portfolio_df['return'].values
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Годовой коэфф. Шарпа (при дневных данных)
        max_drawdown = 0
        peak = portfolio_df['total_value'].iloc[0]
        
        for value in portfolio_df['total_value']:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        max_drawdown *= 100  # В процентах
    else:
        sharpe_ratio = np.nan
        max_drawdown = np.nan
    
    # Вывод результатов
    print(f"\nРезультаты бэктестинга:")
    print(f"Начальный капитал: ${initial_capital:.2f}")
    print(f"Конечный капитал: ${capital:.2f}")
    print(f"Общая доходность: {total_return:.2f}%")
    print(f"Доходность 'Купи и держи': {buy_and_hold_return:.2f}%")
    print(f"Альфа (разница): {total_return - buy_and_hold_return:.2f}%")
    print(f"Коэффициент Шарпа: {sharpe_ratio:.2f}")
    print(f"Максимальная просадка: {max_drawdown:.2f}%")
    print(f"Количество сделок: {len(trades_df)}")
    
    if len(trades_df) > 0:
        # Расчет дополнительных метрик по сделкам
        trades_df['profit'] = 0.0
        
        buy_idx = trades_df[trades_df['action'] == 'BUY'].index
        sell_idx = trades_df[trades_df['action'].str.contains('SELL')].index
        
        if len(buy_idx) == len(sell_idx):
            for i in range(len(sell_idx)):
                buy_value = trades_df.loc[buy_idx[i], 'value']
                buy_commission = trades_df.loc[buy_idx[i], 'commission']
                sell_value = trades_df.loc[sell_idx[i], 'value']
                sell_commission = trades_df.loc[sell_idx[i], 'commission']
                
                profit = sell_value - buy_value - buy_commission - sell_commission
                trades_df.loc[sell_idx[i], 'profit'] = profit
        
        # Процент выигрышных сделок
        winning_trades = trades_df[trades_df['profit'] > 0]
        win_rate = len(winning_trades) / (len(trades_df) // 2) * 100 if len(trades_df) > 0 else 0
        
        print(f"Процент выигрышных сделок: {win_rate:.2f}%")
    
    return {
        'trades': trades_df,
        'portfolio': portfolio_df,
        'metrics': {
            'total_return': total_return,
            'buy_and_hold_return': buy_and_hold_return,
            'alpha': total_return - buy_and_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades_df),
            'win_rate': win_rate if len(trades_df) > 0 else 0
        }
    }

def advanced_feature_engineering(self, df):
    """
    Расширенное создание признаков, включая поведенческие и рыночные метрики
    
    Args:
        df (pd.DataFrame): DataFrame с основными ценовыми данными
        
    Returns:
        pd.DataFrame: DataFrame с дополнительными признаками
    """
    print("Расширенное конструирование признаков...")
    
    # Копия исходных данных
    extended_df = df.copy()
    
    # ----- 1. Нелинейные преобразования существующих признаков -----
    
    # Логарифмические и квадратичные преобразования цен
    extended_df['log_close'] = np.log(extended_df['Close'])
    extended_df['close_squared'] = extended_df['Close'] ** 2
    
    # ----- 2. Исторические экстремумы -----
    
    # Скользящие максимумы/минимумы за различные периоды
    for period in [5, 10, 20, 50, 100]:
        extended_df[f'rolling_max_{period}'] = extended_df['High'].rolling(window=period).max()
        extended_df[f'rolling_min_{period}'] = extended_df['Low'].rolling(window=period).min()
        
        # Относительная позиция текущей цены между макс. и мин.
        denominator = extended_df[f'rolling_max_{period}'] - extended_df[f'rolling_min_{period}']
        extended_df[f'price_position_{period}'] = (extended_df['Close'] - extended_df[f'rolling_min_{period}']) / np.where(denominator != 0, denominator, 1)
    
    # ----- 3. Поведенческие метрики -----
    
    # Кумулятивный объем для оценки трендов
    extended_df['cum_volume'] = extended_df['Volume'].rolling(window=20).sum()
    
    # Количество дней роста/падения подряд
    extended_df['price_diff'] = extended_df['Close'].diff()
    extended_df['up_streak'] = 0
    extended_df['down_streak'] = 0
    
    up_streak = 0
    down_streak = 0
    
    for i in range(1, len(extended_df)):
        if extended_df['price_diff'].iloc[i] > 0:
            up_streak += 1
            down_streak = 0
        elif extended_df['price_diff'].iloc[i] < 0:
            down_streak += 1
            up_streak = 0
        
        extended_df['up_streak'].iloc[i] = up_streak
        extended_df['down_streak'].iloc[i] = down_streak
    
    # Gap-анализ (разница между ценами закрытия и открытия следующего дня)
    extended_df['overnight_gap'] = extended_df['Open'] - extended_df['Close'].shift(1)
    extended_df['overnight_gap_pct'] = extended_df['overnight_gap'] / extended_df['Close'].shift(1)
    
    # ----- 4. Индикаторы нестабильности -----
    
    # Исторические волатильности за разные периоды
    for period in [5, 10, 20, 50]:
        extended_df[f'volatility_{period}d'] = extended_df['log_return'].rolling(window=period).std() * np.sqrt(252)  # Годовая волатильность
    
    # Z-оценка текущей цены относительно исторической
    for period in [20, 50, 100]:
        rolling_mean = extended_df['Close'].rolling(window=period).mean()
        rolling_std = extended_df['Close'].rolling(window=period).std()
        extended_df[f'price_zscore_{period}'] = (extended_df['Close'] - rolling_mean) / rolling_std
    
    # ----- 5. Индикаторы ликвидности -----
    
    # Обобщенная мера ликвидности
    extended_df['liquidity'] = extended_df['Volume'] * extended_df['Close']
    extended_df['rel_liquidity'] = extended_df['liquidity'] / extended_df['liquidity'].rolling(window=20).mean()
    
    # ----- 6. Комбинации технических индикаторов -----
    
    # Пересечения скользящих средних (сигналы)
    extended_df['ma_cross_5_20'] = np.where(
        (extended_df['ma_5'].shift(1) < extended_df['ma_20'].shift(1)) &
        (extended_df['ma_5'] > extended_df['ma_20']),
        1, np.where(
            (extended_df['ma_5'].shift(1) > extended_df['ma_20'].shift(1)) &
            (extended_df['ma_5'] < extended_df['ma_20']),
            -1, 0
        )
    )
    
    # Сила тренда (комбинация нескольких индикаторов)
    if 'macd' in extended_df.columns and 'rsi' in extended_df.columns:
        # Нормализация MACD и RSI для объединения
        extended_df['macd_norm'] = (extended_df['macd'] - extended_df['macd'].rolling(window=50).min()) / \
                                  (extended_df['macd'].rolling(window=50).max() - extended_df['macd'].rolling(window=50).min())
        
        extended_df['rsi_norm'] = (extended_df['rsi'] - 30) / 40  # Нормализация RSI к [0,1] с фокусом на диапазон 30-70
        
        # Комбинированный индикатор силы тренда
        extended_df['trend_strength'] = (extended_df['macd_norm'] + extended_df['rsi_norm']) / 2
    
    # ----- 7. Признаки, специфичные для сезонности -----
    
    # День недели (кодирование с помощью синусов и косинусов)
    extended_df['day_of_week'] = extended_df.index.dayofweek
    extended_df['day_sin'] = np.sin(2 * np.pi * extended_df['day_of_week'] / 5)  # 5 рабочих дней
    extended_df['day_cos'] = np.cos(2 * np.pi * extended_df['day_of_week'] / 5)
    
    # Признаки месяца
    extended_df['month'] = extended_df.index.month
    extended_df['month_sin'] = np.sin(2 * np.pi * extended_df['month'] / 12)
    extended_df['month_cos'] = np.cos(2 * np.pi * extended_df['month'] / 12)
    
    # Квартал
    extended_df['quarter'] = extended_df.index.quarter
    
    # Дни до конца месяца
    extended_df['day_of_month'] = extended_df.index.day
    days_in_month = extended_df.index.days_in_month
    extended_df['days_to_month_end'] = days_in_month - extended_df['day_of_month']
    
    # ----- 8. Интеграция поведенческих метрик с новостным сентиментом -----
    
    # Если есть данные о тональности новостей, создаем интегрированные признаки
    sentiment_cols = [col for col in extended_df.columns if 'sentiment' in col]
    
    if sentiment_cols:
        # Выберем основной столбец тональности для примера
        main_sentiment_col = sentiment_cols[0]
        
        # Комбинированные технические и сентиментальные индикаторы
        if 'rsi' in extended_df.columns:
            # RSI, скорректированный по сентименту
            extended_df['sentiment_adjusted_rsi'] = extended_df['rsi'] * (1 + 0.2 * extended_df[main_sentiment_col])
        
        # Тренд, скорректированный по сентименту
        if 'ma_20_dist' in extended_df.columns:
            extended_df['sentiment_trend_alignment'] = extended_df['ma_20_dist'] * extended_df[main_sentiment_col]
    
    # Удаление временных столбцов
    cols_to_drop = ['day_of_week', 'month', 'day_of_month']
    extended_df.drop(columns=[col for col in cols_to_drop if col in extended_df.columns], inplace=True)
    
    print(f"Создано {len(extended_df.columns) - len(df.columns)} новых признаков")
    
    return extended_df

def ensemble_modeling(self, X, y, test_size=0.2, random_state=42, n_splits=5):
    print("Построение ансамблевой модели с использованием стекинга...")
    
    # Импорт библиотек для стекинга
    from sklearn.ensemble import StackingRegressor
    
    # Масштабирование признаков
    X_scaled = self.scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    # Разделение данных для временной кросс-валидации
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Базовые модели для ансамбля
    base_models = [
        ('linear', LinearRegression()),
        ('ridge', Ridge(alpha=1.0)),
        ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=random_state)),
        ('xgb', XGBRegressor(n_estimators=100, random_state=random_state, learning_rate=0.1)),
        ('lgbm', LGBMRegressor(n_estimators=100, random_state=random_state))
    ]
    
    # Мета-модель
    meta_model = Lasso(alpha=0.001)
    
    # Создание ансамблевой модели с использованием стекинга
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=tscv,
        n_jobs=-1  # Использовать все доступные ядра процессора
    )
    
    # Оценка ансамблевой модели с помощью временной кросс-валидации
    cv_mae, cv_rmse, cv_mape, cv_r2 = [], [], [], []
    
    # Временная кросс-валидация для оценки ансамблевой модели
    for train_idx, test_idx in tqdm(tscv.split(X_scaled), desc="Оценка ансамблевой модели", total=n_splits):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Обучение модели
        stacking_model.fit(X_train, y_train)
        
        # Прогноз
        y_pred = stacking_model.predict(X_test)
        
        # Расчет метрик
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(np.exp(y_test) - 1, np.exp(y_pred) - 1) * 100
        r2 = r2_score(y_test, y_pred)
        
        cv_mae.append(mae)
        cv_rmse.append(rmse)
        cv_mape.append(mape)
        cv_r2.append(r2)
    
    # Среднее значение метрик
    mean_metrics = {
        'MAE': np.mean(cv_mae),
        'RMSE': np.mean(cv_rmse),
        'MAPE': np.mean(cv_mape),
        'R²': np.mean(cv_r2)
    }
    
    print(f"Результаты оценки ансамблевой модели:")
    print(f"  MAE: {mean_metrics['MAE']:.6f}, RMSE: {mean_metrics['RMSE']:.6f}, "
          f"MAPE: {mean_metrics['MAPE']:.2f}%, R²: {mean_metrics['R²']:.4f}")
    
    # Обучение ансамблевой модели на всем наборе данных
    stacking_model.fit(X_scaled, y)
    
    # Сохраняем ансамблевую модель как лучшую
    self.best_model = stacking_model
    self.best_model_name = "Stacking Ensemble"
    
    # Анализ вклада каждой базовой модели
    print("\nВклад базовых моделей в ансамбль:")
    
    # Если мета-модель - линейная, можно извлечь коэффициенты
    if hasattr(stacking_model.final_estimator_, 'coef_'):
        coefficients = stacking_model.final_estimator_.coef_
        
        # Получение имен базовых моделей
        model_names = [name for name, _ in base_models]
        
        # Вывод вклада каждой модели
        for name, coef in zip(model_names, coefficients):
            print(f"  {name}: {np.abs(coef) / np.sum(np.abs(coefficients)) * 100:.2f}%")
    
    return stacking_model, mean_metrics

def ablation_study(self, X, y, features_groups, test_size=0.2, random_state=42):
    print("Исследование влияния различных групп признаков на качество модели...")
    
    # Проверка корректности групп признаков
    all_features = set(X.columns)
    for group_name, group_features in features_groups.items():
        missing_features = set(group_features) - all_features
        if missing_features:
            print(f"Предупреждение: В группе '{group_name}' указаны отсутствующие признаки: {missing_features}")
    
    # Модель для оценки (используем XGBoost для скорости)
    model = XGBRegressor(n_estimators=100, random_state=random_state, learning_rate=0.1)
    
    # Временная кросс-валидация
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Результаты для всех признаков
    results = {
        'all_features': self._evaluate_features_group(X, y, list(X.columns), model, tscv)
    }
    
    # Оценка каждой группы признаков отдельно
    for group_name, group_features in features_groups.items():
        # Фильтрация только доступных признаков
        available_features = list(set(group_features) & all_features)
        
        if available_features:
            results[f'only_{group_name}'] = self._evaluate_features_group(
                X, y, available_features, model, tscv
            )
    
    # Оценка всех признаков, кроме каждой группы
    for group_name, group_features in features_groups.items():
        available_features = list(set(group_features) & all_features)
        
        if available_features:
            # Признаки без текущей группы
            excluded_features = list(all_features - set(available_features))
            
            results[f'exclude_{group_name}'] = self._evaluate_features_group(
                X, y, excluded_features, model, tscv
            )
    
    # Преобразование результатов в DataFrame
    results_df = pd.DataFrame(results).T
    
    # Рассчитаем снижение эффективности при исключении каждой группы
    baseline_rmse = results_df.loc['all_features', 'RMSE']
    
    for group_name in features_groups:
        if f'exclude_{group_name}' in results_df.index:
            exclude_rmse = results_df.loc[f'exclude_{group_name}', 'RMSE']
            
            # Изменение RMSE в процентах
            change_pct = (exclude_rmse - baseline_rmse) / baseline_rmse * 100
            results_df.loc[f'exclude_{group_name}', 'RMSE_change_pct'] = change_pct
    
    # Сортировка по важности групп (по изменению RMSE)
    if 'RMSE_change_pct' in results_df.columns:
        important_groups = results_df[results_df.index.str.startswith('exclude_')].sort_values('RMSE_change_pct', ascending=False)
        
        print("\nВажность групп признаков (по снижению качества при исключении):")
        for idx, row in important_groups.iterrows():
            group_name = idx.replace('exclude_', '')
            change = row['RMSE_change_pct']
            direction = "ухудшение" if change > 0 else "улучшение"
            print(f"  {group_name}: {abs(change):.2f}% ({direction})")
    
    return results_df
