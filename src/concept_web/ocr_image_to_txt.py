import os
from pathlib import Path

from BeamerBot.src_code.ocr_image_files import ocr_image
from dotenv import load_dotenv

load_dotenv()

readingDir = Path(os.getenv('readingsDir'))

inputDir = readingDir / "L11/snips/"

outputDir = readingDir / "L11/"
# %%

# Define the directory containing the images

output_text = ""

# Loop through all images in the directory
for img_path in inputDir.glob("*.png"):
    text = ocr_image(img_path)
    output_text += text + "\n\n"

# %%
# Save the concatenated text to a .txt file
output_file = Path(outputDir / "11.1 textbook_page136_153.txt")
output_file.write_text(output_text)
