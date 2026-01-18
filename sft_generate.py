import pandas as pd
import json

chinese_json = pd.read_json('sft_data/chinese.json')
english_json = pd.read_json('sft_data/english.json')
other_chinese_json = pd.read_json('sft_data/other_chinese.json')

sampled_chinese = chinese_json.sample(n=600, random_state=42)
sampled_english = english_json.sample(n=600, random_state=42)
sampled_other_chinese = other_chinese_json.sample(n=300, random_state=42)

final_sampled = pd.concat([sampled_chinese, sampled_english, sampled_other_chinese])
final_sampled = final_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

with open("sft_data/sft_1500.json", "w", encoding="utf-8") as f:
    json.dump(final_sampled.to_dict(orient="records"), f, ensure_ascii=False, indent=4)
