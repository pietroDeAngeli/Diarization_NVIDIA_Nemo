import json
import re

# Convert the Diarizer output JSON into the company format JSON 
def convert_json_from_file(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as file:
        input_data = json.load(file)
    
    output_json = {
        "name": None,
        "country": None,
        "tokens": [],
        "channel_name": None,
        "stream_url": [None],
        "transcription_content": input_data.get("transcription", ""),
        "primary_language": None,
        "media_type": None
    }

    for word_data in input_data["words"]:
        number = re.findall(r'\d+', word_data["speaker"])
        if number:
            speaker = f"spk{number[0]}"
        else:
            speaker = "Unknown"
        token = {
            "data": word_data["word"].strip(),
            "language": None,
            "speaker": {
                "id": speaker,
                "check": False,
                "name": None
            },
            "start_millis": int(word_data["start_time"] * 1000),
            "end_millis": int(word_data["end_time"] * 1000)
        }
        output_json["tokens"].append(token)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(output_json, outfile, ensure_ascii=False, indent=4)


# Dizionario che mappa i codici ISO 639-1 alle rispettive lingue
iso_to_languages = {
    "be": "Belarusian",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hr": "Croatian",
    "it": "Italian",
    "pl": "Polish",
    "ru": "Russian",
    "uk": "Ukrainian"
}

def convert_iso_to_language(iso_list):
    for iso in iso_list:
        language_name = iso_to_languages.get(iso, "Unknown")
        return language_name