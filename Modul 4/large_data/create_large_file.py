import nltk
from nltk.corpus import gutenberg, brown
import os

# Make sure corpora are downloaded first
nltk.download('gutenberg')
nltk.download('brown')

# Choose some large text sources
texts = []
for fileid in gutenberg.fileids():
    texts.append(gutenberg.raw(fileid))
for fileid in brown.fileids()[:50]:  # only first 50 to avoid too many small files
    texts.append(brown.raw(fileid))

# Concatenate into one big base string
base_text = "\n".join(texts)

# Write the text multiple times to make it really large
output_file = "large_text.txt"
repeat = 100  # adjust this to increase file size

with open(output_file, "w", encoding="utf-8") as f:
    for i in range(repeat):
        f.write(base_text)
        f.write("\n\n")



# Report file size
print(f"Created {output_file} with size {os.path.getsize(output_file) / (1024**2):.2f} MB")
