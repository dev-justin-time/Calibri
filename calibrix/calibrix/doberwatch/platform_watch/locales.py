# SPDX-License-Identifier: MIT
"""OpenWatch locales — 10 language deployments with cultural short names.

Goal: maximum human ground covered with a single binary + 10 static bundles.
Each locale is a short, memorable name rooted in that culture's word for
light / mirror / watchfulness — not a translation of "OpenWatch" but a
sibling name that means the same thing locally.

Coverage: ~4.7B first-language speakers + ~1.5B L2, every inhabited continent,
all top-10 internet languages by design. Not exhaustive — phase 2 adds
id-ID, tr-TR, ur-PK, vi-VN, it-IT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class Locale:
    code: str          # BCP-47
    lang: str          # ISO 639-1
    name: str          # short cultural name (brand)
    meaning: str       # literal sense in English
    language: str      # English label for the language
    native: str        # native label
    dir: str           # ltr | rtl
    region: str        # primary region shorthand
    speakers_m: int    # L1+L2 approx, millions, 2024-25 estimate
    note: str          # why this name, cultural hook

LOCALES: List[Locale] = [
    Locale("en", "en", "Beacon", "beacon / signal fire",
           "English", "English", "ltr", "Global", 1450,
           "Old English beacn — a light that warns ships. Watchtower logic. The root shared with French/German."),
    Locale("zh-Hans", "zh", "Mingjing", "bright mirror",
           "Mandarin", "中文", "ltr", "East Asia", 1120,
           "明镜 — the classical mirror that reflects without distortion. Seen in Tang poetry and legal tradition."),
    Locale("es-419", "es", "Vela", "vigil / candle watch",
           "Spanish", "Español", "ltr", "LatAm + Spain", 560,
           "Hacer vela — to keep vigil. Also a candle; light that stays up while others sleep."),
    Locale("hi", "hi", "Drishti", "sight / clear vision",
           "Hindi", "हिन्दी", "ltr", "South Asia", 610,
           "दृष्टि — not just seeing, but discerning. Used in philosophy for right view."),
    Locale("ar", "ar", "Noor", "light",
           "Arabic", "العربية", "rtl", "MENA", 400,
           "نور — light that reveals. Shared across Urdu/Persian/Swahili. One syllable, instantly recognizable."),
    Locale("pt-BR", "pt", "Claro", "clear / bright",
           "Portuguese", "Português", "ltr", "Brazil + Lusophone", 265,
           "Claro — what a clean record looks like. Also a common word for 'of course' in Brazil."),
    Locale("fr", "fr", "Éclaire", "it illuminates",
           "French", "Français", "ltr", "W. Africa + Europe", 310,
           "Éclairer — to light, to clarify. Verb form feels active, not a trophy."),
    Locale("ru", "ru", "Dozor", "watch / patrol",
           "Russian", "Русский", "ltr", "E. Europe + Central Asia", 255,
           "Дозор — the night watch. Cultural anchor (Lukyanenko). Means patrol, not police."),
    Locale("ja", "ja", "Akashi", "revealing light / proof",
           "Japanese", "日本語", "ltr", "Japan", 125,
           "明かし — to make clear, to disclose. Related to akashi/ming. Short, gentle, not militaristic."),
    Locale("bn", "bn", "Alo", "light",
           "Bengali", "বাংলা", "ltr", "Bengal", 275,
           "আলো — daily word, warm. Covers Bangladesh + West Bengal + diaspora. Pairs with Noor next door."),
]

# Fast lookups
_BY_CODE: Dict[str, Locale] = {loc.code: loc for loc in LOCALES}
_BY_LANG: Dict[str, Locale] = {loc.lang: loc for loc in LOCALES}

# Locale-aware UI strings for the blog header/summary. Keep keys minimal;
# full translation of criteria happens via template helpers, not here.
STRINGS: Dict[str, Dict[str, str]] = {
    "en": {"title": "OpenWatch Daily", "summary": "Summary", "platforms": "Platforms — same prompt pack, same rubric, same gates", "best": "Best", "worst": "Worst", "no_platforms": "No platforms in this report.", "switch": "Language"},
    "zh": {"title": "每日明镜", "summary": "概览", "platforms": "平台 — 同一套题，同量尺，同门槛", "best": "最佳", "worst": "最差", "no_platforms": "本报告暂无平台。", "switch": "语言"},
    "es": {"title": "Vela Diaria", "summary": "Resumen", "platforms": "Plataformas — mismo examen, misma regla, mismo umbral", "best": "Mejor", "worst": "Peor", "no_platforms": "Sin plataformas en este informe.", "switch": "Idioma"},
    "hi": {"title": "दृष्टि दैनिक", "summary": "सार", "platforms": "प्लेटफ़ॉर्म — एक ही प्रश्न, एक ही कसौटी", "best": "सर्वोत्तम", "worst": "सबसे कम", "no_platforms": "इस रिपोर्ट में कोई प्लेटफ़ॉर्म नहीं।", "switch": "भाषा"},
    "ar": {"title": "نور اليومي", "summary": "الملخص", "platforms": "المنصات — نفس الأسئلة، نفس المعيار", "best": "الأفضل", "worst": "الأضعف", "no_platforms": "لا توجد منصات في هذا التقرير.", "switch": "اللغة"},
    "pt": {"title": "Claro Diário", "summary": "Resumo", "platforms": "Plataformas — mesma prova, mesma régua", "best": "Melhor", "worst": "Pior", "no_platforms": "Nenhuma plataforma neste relatório.", "switch": "Idioma"},
    "fr": {"title": "Éclaire Quotidien", "summary": "Résumé", "platforms": "Plateformes — même épreuve, même grille", "best": "Meilleur", "worst": "Plus faible", "no_platforms": "Aucune plateforme dans ce rapport.", "switch": "Langue"},
    "ru": {"title": "Дозор — ежедневно", "summary": "Сводка", "platforms": "Платформы — один набор заданий, одна шкала", "best": "Лучший", "worst": "Худший", "no_platforms": "В отчёте пока нет платформ.", "switch": "Язык"},
    "ja": {"title": "あかし日報", "summary": "サマリー", "platforms": "プラットフォーム — 同一課題、同一基準", "best": "最高", "worst": "最低", "no_platforms": "このレポートにプラットフォームはありません。", "switch": "言語"},
    "bn": {"title": "আলো দৈনিক", "summary": "সারাংশ", "platforms": "প্ল্যাটফর্ম — একই প্রশ্ন, একই মাপকাঠি", "best": "সেরা", "worst": "সবচেয়ে কম", "no_platforms": "এই প্রতিবেদনে কোনো প্ল্যাটফর্ম নেই।", "switch": "ভাষা"},
}

def get_locale(code: str) -> Locale:
    if code in _BY_CODE:
        return _BY_CODE[code]
    base = code.split("-")[0]
    if base in _BY_LANG:
        return _BY_LANG[base]
    return _BY_CODE["en"]

def t(lang: str, key: str) -> str:
    base = lang.split("-")[0]
    return STRINGS.get(base, STRINGS["en"]).get(key, STRINGS["en"].get(key, key))

__all__ = ["Locale", "LOCALES", "get_locale", "t", "STRINGS", "PACK_HASH_LOCALES_VERSION"]

# Version the locale set alongside the prompt pack — bump when adding a locale
PACK_HASH_LOCALES_VERSION = "openwatch.locales.v1"
