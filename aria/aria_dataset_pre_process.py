import json

sequences_to_keep = [
    "loc1_script2_seq1_rec1", "loc1_script2_seq1_rec2", "loc1_script2_seq3_rec1",
    "loc1_script2_seq3_rec2", "loc1_script2_seq4_rec1", "loc1_script2_seq4_rec2",
    "loc1_script2_seq6_rec1", "loc1_script2_seq6_rec2", "loc1_script2_seq7_rec1",
    "loc1_script2_seq7_rec2", "loc1_script2_seq8_rec1", "loc1_script2_seq8_rec2",
    "loc1_script3_seq2_rec1", "loc2_script2_seq1_rec1", "loc2_script2_seq1_rec2",
    "loc2_script2_seq3_rec1", "loc2_script2_seq3_rec2", "loc2_script2_seq4_rec1",
    "loc2_script2_seq5_rec1", "loc2_script2_seq5_rec2", "loc2_script2_seq6_rec1",
    "loc2_script2_seq6_rec2", "loc2_script2_seq8_rec1", "loc2_script2_seq8_rec2",
    "loc2_script3_seq1_rec2", "loc2_script3_seq2_rec1", "loc2_script3_seq2_rec2",
    "loc2_script3_seq4_rec1", "loc2_script3_seq4_rec2", "loc3_script2_seq1_rec1",
    "loc3_script2_seq1_rec2", "loc3_script2_seq3_rec1", "loc3_script2_seq3_rec2",
    "loc3_script2_seq4_rec1", "loc3_script2_seq4_rec2", "loc3_script2_seq5_rec1",
    "loc3_script2_seq5_rec2", "loc3_script2_seq7_rec1", "loc3_script2_seq7_rec2",
    "loc3_script3_seq1_rec2", "loc3_script3_seq2_rec1", "loc3_script3_seq2_rec2",
    "loc3_script3_seq4_rec1", "loc3_script3_seq4_rec2"
]

try:
    with open('/mas/robots/prg-aria/raw/AriaEverydayActivities_download_urls.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = {'sequences': {}}

    for seq_name, seq_data in data.get('sequences', {}).items():
        if seq_name in sequences_to_keep:
            processed_data['sequences'][seq_name] = seq_data

    with open('/mas/robots/prg-aria/raw/AriaEverydayActivities_download_urls_processed.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4)

    print("Successfully created 'AriaEverydayActivities_download_urls_processed.json' file.")

except FileNotFoundError:
    print("Error: 'AriaEverydayActivities_download_urls.json' file not found. Please check if the file is in the correct location.")
except json.JSONDecodeError:
    print("Error: 'AriaEverydayActivities_download_urls.json' file is not in the correct format.")