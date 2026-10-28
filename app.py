"""
Weather AI bot that learns from the answers its users pick.

How it works:
1. The bot fetches live weather data from Open-Meteo (free, no API key needed).
2. Gemini writes several answers in different styles.
3. The user picks the best one.
4. The pick is saved to memory. Next time the bot uses the styles people pick
   more often and shows the model the most liked answers as examples.

Run it on your own computer:  streamlit run app.py
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
# SETTINGS (safe to change)
# =====================================================================

# Gemini models: the first one is the main model, the second one is a backup
# (used when the first one is unavailable or overloaded).
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

DEFAULT_CITY = "Prague"  # city shown when the app opens
DEFAULT_LANG = "cs"  # default language of the interface and the answers
LANGUAGES = ["cs", "en"]  # languages offered in the sidebar
NUM_VARIANTS = 3  # how many answer options to show
MAX_EXAMPLES = 5  # how many liked answers the model gets as examples
MEMORY_FILE = "feedback.json"  # local memory file (used when no Gist is set up)
GIST_FILENAME = "feedback.json"  # file name inside the GitHub Gist

# Answer styles. These are instructions for the AI, so they stay in English.
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

# Interface texts in Czech and English.
TEXTS = {
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

# WMO weather codes returned by Open-Meteo: code -> (emoji, description for the AI).
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
# HELPERS
# =====================================================================


def get_secret(name):
    """Reads a value from Streamlit Secrets. Returns None instead of an error when it's missing."""
    try:
        return st.secrets[name]
    except Exception:
        return None


def describe(code):
    """Returns (emoji, description) for a weather code."""
    return WEATHER_CODES.get(code, ("🌡️", "unknown"))


def fmt(value, unit=""):
    """Formats a number neatly, even when the value is missing."""
    return "n/a" if value is None else f"{value}{unit}"


def error_status(error):
    """Gets the HTTP status code of an error (for example, 429 = too many requests)."""
    return getattr(error, "status_code", None) or getattr(error, "code", None)


# =====================================================================
# WEATHER (Open-Meteo, no API key needed)
# =====================================================================


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def find_city(name, lang):
    """Finds a city's coordinates with the free Open-Meteo geocoder."""
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


@st.cache_data(ttl=600, show_spinner=False)  # the weather is refreshed every 10 minutes
def get_weather(lat, lon, timezone):
    """Loads live weather: right now, hourly and daily (yesterday + 7 days ahead)."""
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
    """Finds the current hour in the hourly forecast."""
    current_hour = weather["current"]["time"][:13]  # for example "2026-10-22T11"
    for i, time in enumerate(weather["hourly"]["time"]):
        if time[:13] >= current_hour:
            return i
    return 0


def weather_to_text(weather, place):
    """Turns the API data into compact text that the AI can read."""
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
# BOT MEMORY (GitHub Gist or a local file)
# =====================================================================


def gist_enabled():
    """The Gist is used when Secrets contain both the token and the Gist ID."""
    return bool(get_secret("GITHUB_TOKEN") and get_secret("GIST_ID"))


def gist_request(method, **kwargs):
    """Sends a request to the GitHub API for our Gist."""
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
    """Memory with no picks yet."""
    return {"styles": {}, "examples": []}


def memory_source():
    """Where the memory is read from (so the cache never mixes up the Gist and the file)."""
    return f"gist:{get_secret('GIST_ID')}" if gist_enabled() else f"file:{MEMORY_FILE}"


@st.cache_data(ttl=30, show_spinner=False)
def load_memory(source):
    """Reads the memory: how often each style was shown and picked, plus the liked answers."""
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
    """Saves the memory to the same place it is read from."""
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if gist_enabled():
        gist_request("PATCH", json={"files": {GIST_FILENAME: {"content": text}}})
    else:
        with open(MEMORY_FILE, "w", encoding="utf-8") as file:
            file.write(text)
    load_memory.clear()  # so everyone sees the new statistics right away


# =====================================================================
# LEARNING FROM USERS' PICKS
# =====================================================================


def style_score(memory, style):
    """Share of picks for a style. The +1 and +NUM_VARIANTS smooth the start:
    a new style gets a 1 in NUM_VARIANTS chance instead of zero."""
    stats = memory["styles"].get(style, {})
    return (stats.get("wins", 0) + 1) / (stats.get("shown", 0) + NUM_VARIANTS)


def pick_styles(memory):
    """Chooses the styles for new answers:
    the best ones so far (exploitation) + one random style (exploration)."""
    count = min(NUM_VARIANTS, len(STYLES))
    ranked = sorted(STYLES, key=lambda s: (style_score(memory, s), random.random()), reverse=True)
    chosen = ranked[: count - 1] + [random.choice(ranked[count - 1 :])]
    random.shuffle(chosen)  # shuffle so people don't just pick the first option
    return chosen


def record_choice(memory, answers, chosen_index, question):
    """Records a pick: every shown style gets +1 shown, the picked one also gets +1 win."""
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
    memory["examples"] = memory["examples"][-50:]  # keep only the last 50
    return memory


# =====================================================================
# AI (Gemini)
# =====================================================================


class Answer(BaseModel):
    style: str
    text: str


class Answers(BaseModel):
    answers: list[Answer]


def build_prompt(place, weather_text, styles, memory, history, lang):
    """Builds the instructions for the AI: rules, styles, liked examples and the weather data."""
    style_lines = "\n".join(f"{i + 1}. [{s}] {STYLES[s]}" for i, s in enumerate(styles))
    examples = memory["examples"][-MAX_EXAMPLES:]
    example_lines = (
        "\n".join(f"- Question: {e['question']}\n  Chosen answer ({e['style']}): {e['answer']}" for e in examples)
        or "(no examples yet)"
    )
    history_lines = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:]) or "(new conversation)"
    language = TEXTS[lang]["ai_lang"]  # answer language = the language selected in the sidebar
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
    """Asks Gemini for the answers as JSON. If a model is unavailable, tries the backup model."""
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
                store=False,  # don't store the conversation on Google's servers
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
            if error_status(error) in (401, 403):  # a problem with the key: another model won't help
                raise
    raise last_error


# =====================================================================
# INTERFACE
# =====================================================================

st.set_page_config(page_title="Weather AI bot", page_icon="🌦️")

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history
    st.session_state.pending = None  # answer options waiting for a pick
    st.session_state.place = None  # current city
    st.session_state.flash = None  # short message shown after an action


def choose_answer(index):
    """Runs when the user clicks the pick button next to an answer option."""
    pending = st.session_state.pending
    if not pending:
        return
    try:
        load_memory.clear()  # read the freshest memory so other people's picks aren't overwritten
        memory = record_choice(load_memory(memory_source()), pending["answers"], index, pending["question"])
        save_memory(memory)
    except Exception as error:
        texts = TEXTS.get(st.session_state.get("lang"), TEXTS[DEFAULT_LANG])
        st.session_state.flash = ("error", f"{texts['save_error']}: {error}")
    chosen = pending["answers"][index]
    st.session_state.messages.append({"role": "assistant", "content": chosen["text"], "style": chosen["style"]})
    st.session_state.pending = None


# ---------- Sidebar: language and city ----------
with st.sidebar:
    if st.session_state.get("lang") not in LANGUAGES:  # an old browser tab may remember a removed language
        st.session_state.pop("lang", None)
    lang = st.selectbox(
        "🌐 Jazyk / Language",
        LANGUAGES,
        index=LANGUAGES.index(DEFAULT_LANG),
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

# ---------- Weather ----------
try:
    place = find_city(city, lang)
    if place is None:
        st.error(T["city_not_found"].format(city=city))
        st.stop()
    weather = get_weather(place["lat"], place["lon"], place["timezone"])
except requests.RequestException:
    st.error(T["weather_error"])
    st.stop()

if st.session_state.place != place["label"]:  # a new city means a new conversation
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

# ---------- Memory and learning statistics in the sidebar ----------
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

# ---------- Chat ----------
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
                st.session_state.messages.pop()  # remove the question so it can be asked again
                st.error(T["rate_limit"] if error_status(error) == 429 else f"{T['ai_error']}: {error}")
                st.stop()
    st.session_state.pending = {"id": uuid.uuid4().hex[:8], "question": question, "answers": answers}
    st.rerun()
