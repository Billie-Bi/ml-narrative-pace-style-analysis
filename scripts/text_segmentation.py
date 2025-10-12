import re
import os
import pandas as pd

# Input and output path configuration
INPUT_FILE = "data/To the Lighthouse.txt"
OUTPUT_BASE_DIR = "Narrative_Analysis_output/text_segmentation"
OUTPUT_FILE = os.path.join(OUTPUT_BASE_DIR, "to_the_lighthouse_paragraphs.csv")

# Verify input file existence
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file does not exist: {INPUT_FILE}")

# Read source text
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

part_titles = ["THE WINDOW", "TIME PASSES", "THE LIGHTHOUSE"]
part_pattern = r'(THE WINDOW|TIME PASSES|THE LIGHTHOUSE)'
parts_split = re.split(part_pattern, text)

parts_dict = {}
for i in range(1, len(parts_split), 2):
    part_name = parts_split[i].title()
    part_text = parts_split[i+1].strip()
    parts_dict[part_name] = part_text

# Function to split text into chapters
def split_chapters(part_text):
    chapters = re.split(r'^\s*(\d+)\s*$', part_text, flags=re.MULTILINE)
    chapter_texts = []
    chapter_numbers = []
    for idx, chunk in enumerate(chapters):
        if idx % 2 == 1:
            chapter_numbers.append(int(chunk))
        elif idx % 2 == 0 and chunk.strip():
            chapter_texts.append(chunk.strip())
    if len(chapter_numbers) != len(chapter_texts):
        chapter_numbers = list(range(1, len(chapter_texts)+1))
    return list(zip(chapter_numbers, chapter_texts))

data = []
for part_name, part_text in parts_dict.items():
    chapters = split_chapters(part_text)
    for chapter_id, chapter_text in chapters:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', chapter_text) if p.strip()]
        for paragraph_id, paragraph_text in enumerate(paragraphs, start=1):
            data.append([part_name, chapter_id, paragraph_id, paragraph_text])

# Create output directory
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# Create DataFrame and save to CSV
df = pd.DataFrame(data, columns=['part', 'chapter_id', 'paragraph_id', 'text'])
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

print(f"Splitting completed! CSV file has been saved to {OUTPUT_FILE}")
print(df.head())