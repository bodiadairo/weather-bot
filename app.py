"""
Погодний ШІ-бот, який навчається на виборі користувачів.

Як це працює:
1. Бот бере живі дані про погоду з Open-Meteo (безкоштовно, без ключа).
2. Gemini пише кілька варіантів відповіді в різних стилях.
3. Користувач обирає найкращий варіант.
4. Вибір зберігається в пам'ять. Наступного разу бот частіше пише в тих стилях,
   які обирають люди, а найкращі відповіді показує моделі як приклади.

Запуск на своєму комп'ютері:  streamlit run app.py
"""

import json
import random
import uuid
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from google import genai
from pydantic import BaseModel


# =====================================================================
# НАЛАШТУВАННЯ (це можна спокійно змінювати)
# =====================================================================

# Моделі Gemini: перша основна, друга запасна (якщо перша недоступна або перевантажена).
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

DEFAULT_CITY = "Prague"  # місто, яке відкривається одразу
DEFAULT_LANG = "cs"  # мова інтерфейсу й відповідей за замовчуванням: "cs", "uk" або "en"
NUM_VARIANTS = 3  # скільки варіантів відповіді показувати
MAX_EXAMPLES = 5  # скільки улюблених відповідей давати моделі як приклади
MEMORY_FILE = "feedback.json"  # локальний файл пам'яті (якщо Gist не налаштований)
GIST_FILENAME = "feedback.json"  # назва файлу всередині GitHub Gist

# Стилі відповідей. Підказки для ШІ англійською: так модель розуміє їх найточніше.
STYLES = {
    "short": "Very short: one or two sentences with only the key numbers.",
    "detailed": "Detailed: temperature range, feels-like temperature, precipitation and wind, "
    "broken down by part of the day.",
    "advice": "Practical advice: what to wear, whether to take an umbrella, the best time to go outside. "
    "Include the key numbers.",
    "friendly": "Warm and playful, like a friend texting, with one or two fitting emojis. "
    "The numbers must still be exact.",
    "bullets": "A short bullet list (3-5 bullets) grouped by time of day or by weather parameter.",
}

# Тексти інтерфейсу трьома мовами.
TEXTS = {
    "uk": {
        "lang_name": "Українська",
        "ai_lang": "Ukrainian",
        "title": "Погодний ШІ-бот",
        "subtitle": "Живі дані Open-Meteo + Gemini. Бот вчиться на тому, які відповіді ти обираєш.",
        "city": "Місто",
        "now_in": "Зараз",
        "temp": "Температура",
        "feels": "Відчувається як",
        "wind": "Вітер",
        "precip": "Опади",
        "chart": "Температура на найближчі 24 години",
        "hint": "Наприклад: «Чи потрібна завтра парасолька?»",
        "ask": "Спитай про погоду…",
        "pick_first": "Спочатку обери найкращу відповідь вище",
        "choose": "Яка відповідь найкраща? Твій вибір навчає бота.",
        "choose_btn": "Обрати",
        "thinking": "Пишу кілька варіантів відповіді…",
        "learning": "Чого навчився бот",
        "votes": "Усього виборів",
        "won": "обрали {wins} з {shown}",
        "examples": "Відповіді, які сподобалися найбільше",
        "no_examples": "Поки що немає. Постав питання й обери відповідь!",
        "memory_gist": "Пам'ять: GitHub Gist (зберігається назавжди)",
        "memory_local": "Пам'ять: локальний файл (може скинутися після перезапуску)",
        "memory_error": "Не вдалося прочитати пам'ять",
        "admin": "Для автора",
        "password": "Пароль",
        "reset": "Скинути навчання",
        "reset_done": "Навчання скинуто",
        "no_key": "Не знайдено ключ GEMINI_API_KEY. Додай його в Secrets (див. інструкцію).",
        "city_not_found": "Не вдалося знайти місто «{city}». Спробуй іншу назву.",
        "weather_error": "Не вдалося отримати погоду. Спробуй трохи пізніше.",
        "ai_error": "ШІ не зміг відповісти",
        "rate_limit": "Забагато запитів до ШІ. Зачекай хвилину й спробуй ще раз.",
        "save_error": "Не вдалося зберегти вибір",
        "styles": {
            "short": "Коротко",
            "detailed": "Детально",
            "advice": "Порада",
            "friendly": "Дружньо",
            "bullets": "Списком",
        },
    },
    "cs": {
        "lang_name": "Čeština",
        "ai_lang": "Czech",
        "title": "AI bot o počasí",
        "subtitle": "Živá data z Open-Meteo + Gemini. Bot se učí z toho, které odpovědi vybíráš.",
        "city": "Město",
        "now_in": "Teď",
        "temp": "Teplota",
        "feels": "Pocitově",
        "wind": "Vítr",
        "precip": "Srážky",
        "chart": "Teplota na příštích 24 hodin",
        "hint": "Například: „Budu zítra potřebovat deštník?“",
        "ask": "Zeptej se na počasí…",
        "pick_first": "Nejdřív vyber nejlepší odpověď výše",
        "choose": "Která odpověď je nejlepší? Tvůj výběr bota učí.",
        "choose_btn": "Vybrat",
        "thinking": "Píšu několik variant odpovědi…",
        "learning": "Co se bot naučil",
        "votes": "Celkem výběrů",
        "won": "vybráno {wins} z {shown}",
        "examples": "Odpovědi, které se líbily nejvíc",
        "no_examples": "Zatím žádné. Polož otázku a vyber odpověď!",
        "memory_gist": "Paměť: GitHub Gist (uloží se natrvalo)",
        "memory_local": "Paměť: lokální soubor (po restartu se může smazat)",
        "memory_error": "Paměť se nepodařilo načíst",
        "admin": "Pro autora",
        "password": "Heslo",
        "reset": "Resetovat učení",
        "reset_done": "Učení bylo resetováno",
        "no_key": "Chybí klíč GEMINI_API_KEY. Přidej ho do Secrets (viz návod).",
        "city_not_found": "Město „{city}“ se nepodařilo najít. Zkus jiný název.",
        "weather_error": "Počasí se nepodařilo načíst. Zkus to o něco později.",
        "ai_error": "AI nedokázala odpovědět",
        "rate_limit": "Příliš mnoho dotazů na AI. Počkej minutu a zkus to znovu.",
        "save_error": "Výběr se nepodařilo uložit",
        "styles": {
            "short": "Stručně",
            "detailed": "Podrobně",
            "advice": "Rada",
            "friendly": "Přátelsky",
            "bullets": "V bodech",
        },
    },
    "en": {
        "lang_name": "English",
        "ai_lang": "English",
        "title": "Weather AI bot",
        "subtitle": "Live Open-Meteo data + Gemini. The bot learns from the answers you pick.",
        "city": "City",
        "now_in": "Now",
        "temp": "Temperature",
        "feels": "Feels like",
        "wind": "Wind",
        "precip": "Precipitation",
        "chart": "Temperature for the next 24 hours",
        "hint": "For example: “Do I need an umbrella tomorrow?”",
        "ask": "Ask about the weather…",
        "pick_first": "Pick the best answer above first",
        "choose": "Which answer is best? Your pick trains the bot.",
        "choose_btn": "Pick",
        "thinking": "Writing a few answer options…",
        "learning": "What the bot has learned",
        "votes": "Total picks",
        "won": "picked {wins} of {shown}",
        "examples": "Most liked answers",
        "no_examples": "None yet. Ask a question and pick an answer!",
        "memory_gist": "Memory: GitHub Gist (kept permanently)",
        "memory_local": "Memory: local file (may reset after a restart)",
        "memory_error": "Couldn't read the memory",
        "admin": "For the author",
        "password": "Password",
        "reset": "Reset learning",
        "reset_done": "Learning reset",
        "no_key": "GEMINI_API_KEY is missing. Add it in Secrets (see the guide).",
        "city_not_found": "Couldn't find the city “{city}”. Try another name.",
        "weather_error": "Couldn't load the weather. Try again a bit later.",
        "ai_error": "The AI couldn't answer",
        "rate_limit": "Too many requests to the AI. Wait a minute and try again.",
        "save_error": "Couldn't save your pick",
        "styles": {
            "short": "Short",
            "detailed": "Detailed",
            "advice": "Advice",
            "friendly": "Friendly",
            "bullets": "Bullet points",
        },
    },
}

# Коди погоди WMO, які повертає Open-Meteo: код -> (емодзі, опис для ШІ).
WEATHER_CODES = {
    0: ("☀️", "clear sky"),
    1: ("🌤️", "mainly clear"),
    2: ("⛅", "partly cloudy"),
    3: ("☁️", "overcast"),
    45: ("🌫️", "fog"),
    48: ("🌫️", "freezing fog"),
    51: ("🌦️", "light drizzle"),
    53: ("🌦️", "drizzle"),
    55: ("🌧️", "dense drizzle"),
    56: ("🌧️", "freezing drizzle"),
    57: ("🌧️", "dense freezing drizzle"),
    61: ("🌦️", "light rain"),
    63: ("🌧️", "rain"),
    65: ("🌧️", "heavy rain"),
    66: ("🌧️", "freezing rain"),
    67: ("🌧️", "heavy freezing rain"),
    71: ("🌨️", "light snow"),
    73: ("🌨️", "snow"),
    75: ("❄️", "heavy snow"),
    77: ("🌨️", "snow grains"),
    80: ("🌦️", "light rain showers"),
    81: ("🌧️", "rain showers"),
    82: ("⛈️", "violent rain showers"),
    85: ("🌨️", "light snow showers"),
    86: ("❄️", "heavy snow showers"),
    95: ("⛈️", "thunderstorm"),
    96: ("⛈️", "thunderstorm with hail"),
    99: ("⛈️", "thunderstorm with heavy hail"),
}


# =====================================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# =====================================================================


def get_secret(name):
    """Читає секрет зі Streamlit Secrets. Якщо його немає, повертає None замість помилки."""
    try:
        return st.secrets[name]
    except Exception:
        return None


def describe(code):
    """Повертає (емодзі, опис) для коду погоди."""
    return WEATHER_CODES.get(code, ("🌡️", "unknown"))


def fmt(value, unit=""):
    """Акуратно показує число, навіть якщо даних немає."""
    return "n/a" if value is None else f"{value}{unit}"


def error_status(error):
    """Дістає HTTP-код помилки (наприклад, 429 = забагато запитів)."""
    return getattr(error, "status_code", None) or getattr(error, "code", None)


# =====================================================================
# ПОГОДА (Open-Meteo, без ключа)
# =====================================================================


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def find_city(name, lang):
    """Знаходить координати міста через безкоштовний геокодер Open-Meteo."""
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": name, "count": 1, "language": lang, "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None
    place = results[0]
    label = place["name"] + (f", {place['country']}" if place.get("country") else "")
    return {
        "label": label,
        "lat": place["latitude"],
        "lon": place["longitude"],
        "timezone": place.get("timezone") or "auto",
    }


@st.cache_data(ttl=600, show_spinner=False)  # погода оновлюється раз на 10 хвилин
def get_weather(lat, lon, timezone):
    """Завантажує живу погоду: зараз, по годинах і по днях (учора + 7 днів уперед)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,"
        "weather_code,wind_speed_10m,cloud_cover",
        "hourly": "temperature_2m,apparent_temperature,precipitation_probability,precipitation,"
        "weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "precipitation_probability_max,wind_speed_10m_max,sunrise,sunset,uv_index_max",
        "past_days": 1,
        "forecast_days": 7,
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def first_hour_index(weather):
    """Знаходить у погодинному прогнозі поточну годину."""
    current_hour = weather["current"]["time"][:13]  # наприклад "2026-10-22T11"
    for i, time in enumerate(weather["hourly"]["time"]):
        if time[:13] >= current_hour:
            return i
    return 0


def weather_to_text(weather, place):
    """Перетворює дані з API на компактний текст, який зрозуміє ШІ."""
    cur, day, hour = weather["current"], weather["daily"], weather["hourly"]
    now = datetime.fromisoformat(cur["time"])
    lines = [
        f"Location: {place}",
        f"Current local time: {now:%Y-%m-%d %H:%M} ({now:%A})",
        f"NOW: {describe(cur['weather_code'])[1]}, {fmt(cur['temperature_2m'])}°C "
        f"(feels like {fmt(cur['apparent_temperature'])}°C), humidity {fmt(cur['relative_humidity_2m'])}%, "
        f"precipitation {fmt(cur['precipitation'])} mm, wind {fmt(cur['wind_speed_10m'])} km/h, "
        f"clouds {fmt(cur['cloud_cover'])}%",
        "",
        "DAILY (yesterday, today and the next days):",
    ]
    for i, date in enumerate(day["time"]):
        weekday = datetime.fromisoformat(date).strftime("%A")
        lines.append(
            f"- {date} ({weekday}): {describe(day['weather_code'][i])[1]}, "
            f"{fmt(day['temperature_2m_min'][i])}..{fmt(day['temperature_2m_max'][i])}°C, "
            f"precipitation {fmt(day['precipitation_sum'][i], ' mm')} "
            f"(max probability {fmt(day['precipitation_probability_max'][i], '%')}), "
            f"max wind {fmt(day['wind_speed_10m_max'][i], ' km/h')}, UV {fmt(day['uv_index_max'][i])}, "
            f"sunrise {(day['sunrise'][i] or '')[11:]}, sunset {(day['sunset'][i] or '')[11:]}"
        )
    lines += [
        "",
        "HOURLY, next 48 hours "
        "(time | weather | temp °C | feels like °C | precipitation probability % | precipitation mm | wind km/h):",
    ]
    start = first_hour_index(weather)
    for i in range(start, min(start + 48, len(hour["time"]))):
        lines.append(
            f"{hour['time'][i].replace('T', ' ')} | {describe(hour['weather_code'][i])[1]} | "
            f"{fmt(hour['temperature_2m'][i])} | {fmt(hour['apparent_temperature'][i])} | "
            f"{fmt(hour['precipitation_probability'][i])} | {fmt(hour['precipitation'][i])} | "
            f"{fmt(hour['wind_speed_10m'][i])}"
        )
    return "\n".join(lines)


# =====================================================================
# ПАМ'ЯТЬ БОТА (GitHub Gist або локальний файл)
# =====================================================================


def gist_enabled():
    """Gist використовується, якщо в Secrets є і токен, і ID gist-а."""
    return bool(get_secret("GITHUB_TOKEN") and get_secret("GIST_ID"))


def gist_request(method, **kwargs):
    """Запит до GitHub API для нашого gist-а."""
    response = requests.request(
        method,
        f"https://api.github.com/gists/{get_secret('GIST_ID')}",
        headers={
            "Authorization": f"Bearer {get_secret('GITHUB_TOKEN')}",
            "Accept": "application/vnd.github+json",
        },
        timeout=10,
        **kwargs,
    )
    response.raise_for_status()
    return response


def empty_memory():
    return {"styles": {}, "examples": []}


def memory_source():
    """Звідки зараз читається пам'ять (потрібно, щоб кеш не плутав gist і файл)."""
    return f"gist:{get_secret('GIST_ID')}" if gist_enabled() else f"file:{MEMORY_FILE}"


@st.cache_data(ttl=30, show_spinner=False)
def load_memory(source):
    """Читає пам'ять: скільки разів кожен стиль показували й обирали, та улюблені відповіді."""
    if source.startswith("gist:"):
        files = gist_request("GET").json().get("files", {})
        text = (files.get(GIST_FILENAME) or {}).get("content") or "{}"
    else:
        try:
            with open(MEMORY_FILE, encoding="utf-8") as file:
                text = file.read()
        except FileNotFoundError:
            text = "{}"
    data = json.loads(text or "{}")
    if not isinstance(data, dict):
        data = {}
    data.setdefault("styles", {})
    data.setdefault("examples", [])
    return data


def save_memory(data):
    """Зберігає пам'ять туди ж, звідки її читаємо."""
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if gist_enabled():
        gist_request("PATCH", json={"files": {GIST_FILENAME: {"content": text}}})
    else:
        with open(MEMORY_FILE, "w", encoding="utf-8") as file:
            file.write(text)
    load_memory.clear()  # щоб усі одразу побачили нову статистику


# =====================================================================
# НАВЧАННЯ НА ВИБОРІ КОРИСТУВАЧІВ
# =====================================================================


def style_score(memory, style):
    """Частка виборів стилю. +1 і +NUM_VARIANTS згладжують старт:
    новий стиль має шанс 1 з NUM_VARIANTS, а не нуль."""
    stats = memory["styles"].get(style, {})
    return (stats.get("wins", 0) + 1) / (stats.get("shown", 0) + NUM_VARIANTS)


def pick_styles(memory):
    """Обирає стилі для нових відповідей:
    найкращі за статистикою (використання знань) + один випадковий (дослідження)."""
    count = min(NUM_VARIANTS, len(STYLES))
    ranked = sorted(STYLES, key=lambda s: (style_score(memory, s), random.random()), reverse=True)
    chosen = ranked[: count - 1] + [random.choice(ranked[count - 1 :])]
    random.shuffle(chosen)  # перемішуємо, щоб люди не обирали просто перший варіант
    return chosen


def record_choice(memory, answers, chosen_index, question):
    """Записує вибір: усім показаним стилям +1 показ, обраному ще +1 перемога."""
    for answer in answers:
        stats = memory["styles"].setdefault(answer["style"], {})
        stats["shown"] = stats.get("shown", 0) + 1
        stats.setdefault("wins", 0)
    winner = answers[chosen_index]
    memory["styles"][winner["style"]]["wins"] += 1
    memory["examples"].append(
        {
            "question": question,
            "answer": winner["text"],
            "style": winner["style"],
            "time": datetime.now().isoformat(timespec="minutes"),
        }
    )
    memory["examples"] = memory["examples"][-50:]  # зберігаємо тільки 50 останніх
    return memory


# =====================================================================
# ШІ (Gemini)
# =====================================================================


class Answer(BaseModel):
    style: str
    text: str


class Answers(BaseModel):
    answers: list[Answer]


def build_prompt(place, weather_text, styles, memory, history, lang):
    """Складає інструкцію для ШІ: правила, стилі, приклади улюблених відповідей і дані."""
    style_lines = "\n".join(f"{i + 1}. [{s}] {STYLES[s]}" for i, s in enumerate(styles))
    examples = memory["examples"][-MAX_EXAMPLES:]
    example_lines = (
        "\n".join(f"- Question: {e['question']}\n  Chosen answer ({e['style']}): {e['answer']}" for e in examples)
        or "(no examples yet)"
    )
    history_lines = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:]) or "(new conversation)"
    language = TEXTS[lang]["ai_lang"]  # мова відповідей = мова, вибрана в бічній панелі
    return f"""You are a friendly weather assistant for {place}.
Answer ONLY from the live weather data below. Never invent numbers.
If the data does not cover the question (another city, a date beyond the forecast, and so on), say so honestly.
Use °C, mm and km/h. Always write every answer in {language}, even if the question is in another language.

Write exactly {len(styles)} different answers, one per style, in this order:
{style_lines}

Users of this app picked these answers as the best ones in the past.
Learn from their tone, length and structure, but take numbers only from the current data,
and keep writing in {language} even if an example is in another language:
{example_lines}

Recent conversation:
{history_lines}

LIVE WEATHER DATA (Open-Meteo):
{weather_text}"""


def ask_gemini(api_key, prompt, question, styles):
    """Просить Gemini написати відповіді у вигляді JSON. Якщо модель недоступна, пробує запасну."""
    client = genai.Client(api_key=api_key)
    schema = Answers.model_json_schema()
    schema["properties"]["answers"]["minItems"] = len(styles)
    schema["properties"]["answers"]["maxItems"] = len(styles)
    last_error = None
    for model in MODELS:
        try:
            interaction = client.interactions.create(
                model=model,
                system_instruction=prompt,
                input=question,
                response_format={"type": "text", "mime_type": "application/json", "schema": schema},
                generation_config={"thinking_level": "low"},
                store=False,  # не зберігати розмову на серверах Google
            )
            result = Answers.model_validate_json(interaction.output_text or "")
            answers = [
                {"style": style, "text": item.text.strip()}
                for style, item in zip(styles, result.answers)
                if item.text.strip()
            ]
            if answers:
                return answers
            last_error = RuntimeError("empty answer")
        except Exception as error:
            last_error = error
            if error_status(error) in (401, 403):  # проблема з ключем: інша модель не допоможе
                raise
    raise last_error


# =====================================================================
# ІНТЕРФЕЙС
# =====================================================================

st.set_page_config(page_title="Weather AI bot", page_icon="🌦️")

if "messages" not in st.session_state:
    st.session_state.messages = []  # історія чату
    st.session_state.pending = None  # варіанти відповіді, які чекають на вибір
    st.session_state.place = None  # поточне місто
    st.session_state.flash = None  # коротке повідомлення після дії


def choose_answer(index):
    """Спрацьовує, коли користувач натискає «Обрати» біля варіанта."""
    pending = st.session_state.pending
    if not pending:
        return
    try:
        load_memory.clear()  # беремо найсвіжішу пам'ять, щоб не затерти чужі голоси
        memory = record_choice(load_memory(memory_source()), pending["answers"], index, pending["question"])
        save_memory(memory)
    except Exception as error:
        st.session_state.flash = ("error", f"{TEXTS[st.session_state.lang]['save_error']}: {error}")
    chosen = pending["answers"][index]
    st.session_state.messages.append({"role": "assistant", "content": chosen["text"], "style": chosen["style"]})
    st.session_state.pending = None


# ---------- Бічна панель: мова й місто ----------
with st.sidebar:
    languages = list(TEXTS)
    lang = st.selectbox(
        "🌐 Jazyk / Мова / Language",
        languages,
        index=languages.index(DEFAULT_LANG),
        format_func=lambda code: TEXTS[code]["lang_name"],
        key="lang",
    )
    T = TEXTS[lang]
    city = st.text_input(T["city"], value=DEFAULT_CITY).strip() or DEFAULT_CITY

st.title(f"🌦️ {T['title']}")
st.caption(T["subtitle"])

if st.session_state.flash:
    kind, text = st.session_state.flash
    (st.error if kind == "error" else st.success)(text)
    st.session_state.flash = None

# ---------- Погода ----------
try:
    place = find_city(city, lang)
    if place is None:
        st.error(T["city_not_found"].format(city=city))
        st.stop()
    weather = get_weather(place["lat"], place["lon"], place["timezone"])
except requests.RequestException:
    st.error(T["weather_error"])
    st.stop()

if st.session_state.place != place["label"]:  # нове місто, отже нова розмова
    st.session_state.place = place["label"]
    st.session_state.messages = []
    st.session_state.pending = None

current = weather["current"]
st.subheader(f"{describe(current['weather_code'])[0]} {T['now_in']}: {place['label']}")
col1, col2, col3, col4 = st.columns(4)
col1.metric(T["temp"], f"{fmt(current['temperature_2m'])} °C")
col2.metric(T["feels"], f"{fmt(current['apparent_temperature'])} °C")
col3.metric(T["wind"], f"{fmt(current['wind_speed_10m'])} km/h")
col4.metric(T["precip"], f"{fmt(current['precipitation'])} mm")

with st.expander(T["chart"]):
    start = first_hour_index(weather)
    chart = pd.DataFrame(
        {
            "time": pd.to_datetime(weather["hourly"]["time"][start : start + 24]),
            "temp": weather["hourly"]["temperature_2m"][start : start + 24],
        }
    )
    st.line_chart(chart, x="time", y="temp", x_label="", y_label="°C", height=220)

# ---------- Пам'ять і бічна панель зі статистикою ----------
memory_error = None
try:
    memory = load_memory(memory_source())
except Exception as error:
    memory, memory_error = empty_memory(), str(error)

with st.sidebar:
    st.divider()
    st.subheader(f"🧠 {T['learning']}")
    total = sum(s.get("wins", 0) for s in memory["styles"].values())
    st.caption(f"{T['votes']}: {total}")
    for style in sorted(STYLES, key=lambda s: style_score(memory, s), reverse=True):
        stats = memory["styles"].get(style, {})
        wins, shown = stats.get("wins", 0), stats.get("shown", 0)
        st.markdown(f"**{T['styles'].get(style, style)}** · {T['won'].format(wins=wins, shown=shown)}")
        st.progress(min(wins / shown, 1.0) if shown else 0.0)
    with st.expander(T["examples"]):
        examples = memory["examples"][-MAX_EXAMPLES:][::-1]
        if not examples:
            st.caption(T["no_examples"])
        for example in examples:
            st.markdown(f"*{example['question']}*")
            st.markdown(example["answer"])
            st.caption(T["styles"].get(example["style"], example["style"]))
            st.divider()
    if memory_error:
        st.warning(f"{T['memory_error']}: {memory_error}")
    st.caption(T["memory_gist"] if gist_enabled() else T["memory_local"])

    admin_password = get_secret("ADMIN_PASSWORD")
    if admin_password:
        with st.expander(T["admin"]):
            entered = st.text_input(T["password"], type="password")
            if entered == str(admin_password) and st.button(T["reset"]):
                try:
                    save_memory(empty_memory())
                    st.session_state.flash = ("success", T["reset_done"])
                except Exception as error:
                    st.session_state.flash = ("error", f"{T['save_error']}: {error}")
                st.rerun()

# ---------- Чат ----------
api_key = get_secret("GEMINI_API_KEY")
if not api_key:
    st.error(T["no_key"])
    st.stop()

if not st.session_state.messages and not st.session_state.pending:
    st.info(T["hint"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("style"):
            st.caption(f"✓ {T['styles'].get(message['style'], message['style'])}")

pending = st.session_state.pending
if pending:
    with st.chat_message("assistant"):
        st.markdown(f"**{T['choose']}**")
        for i, answer in enumerate(pending["answers"]):
            with st.container(border=True):
                st.caption(T["styles"].get(answer["style"], answer["style"]))
                st.markdown(answer["text"])
                st.button(
                    T["choose_btn"],
                    key=f"pick_{pending['id']}_{i}",
                    icon="👍",
                    on_click=choose_answer,
                    args=(i,),
                )

question = st.chat_input(T["pick_first"] if pending else T["ask"], disabled=bool(pending), max_chars=500)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner(T["thinking"]):
            try:
                styles = pick_styles(memory)
                prompt = build_prompt(
                    place["label"],
                    weather_to_text(weather, place["label"]),
                    styles,
                    memory,
                    st.session_state.messages[:-1],
                    lang,
                )
                answers = ask_gemini(api_key, prompt, question, styles)
            except Exception as error:
                st.session_state.messages.pop()  # прибираємо питання, щоб його можна було поставити ще раз
                st.error(T["rate_limit"] if error_status(error) == 429 else f"{T['ai_error']}: {error}")
                st.stop()
    st.session_state.pending = {"id": uuid.uuid4().hex[:8], "question": question, "answers": answers}
    st.rerun()
