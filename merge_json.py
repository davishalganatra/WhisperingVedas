import json
import os
from pathlib import Path

def merge_json_files(texts_dir, output_file):
    texts = [
        {
            "name": "Ramcharitmanas",
            "files": [
                {"file": "Ramcharitmanas\\1_बाल_काण्ड_data.json", "section": "Bal Kand"},
                {"file": "Ramcharitmanas\\2_अयोध्या_काण्ड_data.json", "section": "Ayodhya Kand"},
                {"file": "Ramcharitmanas\\3_अरण्य_काण्ड_data.json", "section": "Aranya Kand"},
                {"file": "Ramcharitmanas\\4_किष्किंधा_काण्ड_data.json", "section": "Kishkindha Kand"},
                {"file": "Ramcharitmanas\\5_सुंदर_काण्ड_data.json", "section": "Sundar Kand"},
                {"file": "Ramcharitmanas\\6_लंका_काण्ड_data.json", "section": "Lanka Kand"},
                {"file": "Ramcharitmanas\\7_उत्तर_काण्ड_data.json", "section": "Uttar Kand"}
            ]
        },
        {
            "name": "Srimad Bhagavad Gita",
            "files": [
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_1.json", "section": "Chapter 1"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_2.json", "section": "Chapter 2"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_3.json", "section": "Chapter 3"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_4.json", "section": "Chapter 4"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_5.json", "section": "Chapter 5"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_6.json", "section": "Chapter 6"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_7.json", "section": "Chapter 7"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_8.json", "section": "Chapter 8"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_9.json", "section": "Chapter 9"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_10.json", "section": "Chapter 10"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_11.json", "section": "Chapter 11"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_12.json", "section": "Chapter 12"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_13.json", "section": "Chapter 13"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_14.json", "section": "Chapter 14"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_15.json", "section": "Chapter 15"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_16.json", "section": "Chapter 16"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_17.json", "section": "Chapter 17"},
                {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_18.json", "section": "Chapter 18"}
            ]
        },
        {
            "name": "Mahabharata",
            "files": []
        },
        {
            "name": "Valmiki Ramayana",
            "files": [
                {"file": "ValmikiRamayana\\1_balakanda.json", "section": "Balakanda"},
                {"file": "ValmikiRamayana\\2_ayodhyakanda.json", "section": "Ayodhyakanda"},
                {"file": "ValmikiRamayana\\3_aranyakanda.json", "section": "Aranyakanda"},
                {"file": "ValmikiRamayana\\4_kishkindhakanda.json", "section": "Kishkindhakanda"},
                {"file": "ValmikiRamayana\\5_sundarakanda.json", "section": "Sundarakanda"},
                {"file": "ValmikiRamayana\\6_yudhhakanda.json", "section": "Yudhhakanda"},
                {"file": "ValmikiRamayana\\7_uttarakanda.json", "section": "Uttarakanda"}
            ]
        },
        {
            "name": "Rigveda",
            "files": [
                {"file": "Rigveda\\rigveda_mandala_1.json", "section": "Mandala 1"},
                {"file": "Rigveda\\rigveda_mandala_2.json", "section": "Mandala 2"},
                {"file": "Rigveda\\rigveda_mandala_3.json", "section": "Mandala 3"},
                {"file": "Rigveda\\rigveda_mandala_4.json", "section": "Mandala 4"},
                {"file": "Rigveda\\rigveda_mandala_5.json", "section": "Mandala 5"},
                {"file": "Rigveda\\rigveda_mandala_6.json", "section": "Mandala 6"},
                {"file": "Rigveda\\rigveda_mandala_7.json", "section": "Mandala 7"},
                {"file": "Rigveda\\rigveda_mandala_8.json", "section": "Mandala 8"},
                {"file": "Rigveda\\rigveda_mandala_9.json", "section": "Mandala 9"},
                {"file": "Rigveda\\rigveda_mandala_10.json", "section": "Mandala 10"}
            ]
        },
        {
            "name": "Yajurveda Shukla",
            "files": [
                {"file": "Yajurveda\\vajasneyi_madhyadina_samhita.json", "section": "Vajasaneyi Madhyandina Samhita"},
                {"file": "Yajurveda\\vajasneyi_kanva_samhita_chapters.json", "section": "Vajasaneyi Kanva Samhita"}
            ]
        },
        {
            "name": "Atharvaveda",
            "files": []
        }
    ]
    knowledge_base = {"texts": []}
    total_verses = 0
    print(f"Checking files in {texts_dir}...")
    for text in texts:
        text_entry = {"name": text["name"], "sections": []}
        print(f"\nProcessing {text['name']}...")
        for file_info in text["files"]:
            file_path = os.path.join(texts_dir, file_info["file"])
            print(f"Checking {file_path}...")
            if not os.path.exists(file_path):
                print(f"  Skipped: File not found")
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                verses = []
                if isinstance(data, list):
                    print(f"  Found list of {len(data)} items")
                    for i, item in enumerate(data):
                        verse = {
                            "id": f"{text['name']}_{file_info['section']}_{i}",
                            "text": item.get("text") or item.get("verse") or item.get("content") or "",
                            "translation": item.get("translation") or item.get("meaning") or item.get("english") or "",
                            "emotion": item.get("emotion", []),
                            "goal": item.get("goal", []),
                            "chakra": item.get("chakra", None),
                            "mantra": item.get("mantra", None),
                            "frequency": item.get("frequency", 432.0),
                            "tone": item.get("tone", "G"),
                            "repetitions": item.get("repetitions", 108)
                        }
                        verses.append(verse)
                elif isinstance(data, dict):
                    print(f"  Found dictionary")
                    if "verses" in data or "shlokas" in data or "hymns" in data:
                        verse_list = data.get("verses") or data.get("shlokas") or data.get("hymns")
                        print(f"    Contains {len(verse_list)} verses/shlokas/hymns")
                        for i, item in enumerate(verse_list):
                            verse = {
                                "id": f"{text['name']}_{file_info['section']}_{i}",
                                "text": item.get("text") or item.get("verse") or item.get("content") or "",
                                "translation": item.get("translation") or item.get("meaning") or item.get("english") or "",
                                "emotion": item.get("emotion", []),
                                "goal": item.get("goal", []),
                                "chakra": item.get("chakra", None),
                                "mantra": item.get("mantra", None),
                                "frequency": item.get("frequency", 432.0),
                                "tone": item.get("tone", "G"),
                                "repetitions": item.get("repetitions", 108)
                            }
                            verses.append(verse)
                    else:
                        print(f"    Scanning dictionary keys")
                        for key, value in data.items():
                            if isinstance(value, dict):
                                verse = {
                                    "id": f"{text['name']}_{file_info['section']}_{key}",
                                    "text": value.get("text") or value.get("verse") or value.get("content") or "",
                                    "translation": value.get("translation") or value.get("meaning") or value.get("english") or "",
                                    "emotion": value.get("emotion", []),
                                    "goal": value.get("goal", []),
                                    "chakra": value.get("chakra", None),
                                    "mantra": value.get("mantra", None),
                                    "frequency": value.get("frequency", 432.0),
                                    "tone": value.get("tone", "G"),
                                    "repetitions": value.get("repetitions", 108)
                                }
                                verses.append(verse)
                if verses:
                    print(f"  Added {len(verses)} verses")
                    text_entry["sections"].append({"name": file_info["section"], "verses": verses})
                    total_verses += len(verses)
                else:
                    print(f"  No verses found in {file_path}")
            except Exception as e:
                print(f"  Error processing {file_path}: {str(e)}")
                continue
        if text_entry["sections"]:
            knowledge_base["texts"].append(text_entry)
            print(f"Added {text['name']} with {len(text_entry['sections'])} sections")
    if not knowledge_base["texts"]:
        print("No texts processed. Creating minimal output.")
        knowledge_base = {
            "texts": [
                {
                    "name": "Default",
                    "sections": [
                        {
                            "name": "Default",
                            "verses": [
                                {
                                    "id": "default_0",
                                    "text": "Om Namah Shivaya",
                                    "translation": "Universal chant",
                                    "emotion": [],
                                    "goal": [],
                                    "chakra": None,
                                    "mantra": "Om Namah Shivaya",
                                    "frequency": 432.0,
                                    "tone": "G",
                                    "repetitions": 108
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        total_verses = 1
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        print(f"\nSuccess: Created {output_file} with {total_verses} verses")
    except Exception as e:
        print(f"\nFailed to write {output_file}: {str(e)}")

if __name__ == "__main__":
    texts_dir = r"D:\sacred_texts"
    output_file = r"D:\mcp server\mcp_server\data\vedic_knowledge\vedic_knowledge.json"
    merge_json_files(texts_dir, output_file)