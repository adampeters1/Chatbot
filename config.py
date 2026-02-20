"""
Central configuration for the chatbot project.
Contains label mappings, file paths, and shared constants.
"""

# ──────────────────────────────────────────────
# Label-to-Category Mapping
# ──────────────────────────────────────────────
LABEL_MAP = {
    0:  {"category": "greeting",
         "description": "Greeting or saying hello."},
    1:  {"category": "farewell",
         "description": "Saying goodbye or farewell."},
    2:  {"category": "thank_you",
         "description": "Expressing gratitude or thanks."},
    3:  {"category": "affirmation",
         "description": "Agreeing or confirming something."},
    4:  {"category": "negation",
         "description": "Disagreeing or denying something."},
    5:  {"category": "small_talk",
         "description": "Engaging in casual or light conversation with no specific purpose."},
    6:  {"category": "bot_capabilities",
         "description": "Inquiries about the bot's features or abilities."},
    7:  {"category": "feedback_positive",
         "description": "Providing positive feedback about the bot, service, or experience."},
    8:  {"category": "feedback_negative",
         "description": "Providing negative feedback about the bot, service, or experience."},
    9:  {"category": "clarification",
         "description": "Asking for clarification or more information about a previous statement or question."},
    10: {"category": "suggestion",
         "description": "Offering a suggestion or recommendation for improvement."},
    11: {"category": "language_change",
         "description": "Requesting a change in the language being used by the bot or information about language options."},
}

NUM_CLASSES = len(LABEL_MAP)

# ──────────────────────────────────────────────
# Reverse Lookup: Category Name → Label Integer
# ──────────────────────────────────────────────
CATEGORY_TO_LABEL = {
    v["category"]: k for k, v in LABEL_MAP.items()
}

# ──────────────────────────────────────────────
# File Paths
# ──────────────────────────────────────────────
DATA_PATH = "data/intent_data.csv"

# ──────────────────────────────────────────────
# Data Split Defaults
# ──────────────────────────────────────────────
TEST_RATIO = 0.2
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Preprocessing Defaults
# ──────────────────────────────────────────────
PREPROCESSING = {
    "use_stop_words": True,
    "use_stemming": True,
    "number_strategy": "remove",
    "tokenize_method": "regex",
}