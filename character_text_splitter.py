from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Source: adapted from the CDC/ACIP dengue vaccine recommendations (MMWR 2021), a public-domain US government document.
clinical_text = """Dengvaxia Efficacy

Pooled vaccine efficacy against virologically confirmed dengue was 82% among seropositive children aged 9–16 years.

Safety and Adverse Reactions

The most frequently reported adverse reactions were headache (40%), injection site pain (32%), and myalgia (29%).

Storage and Administration

Store vaccine antigen and saline diluent at 2°C–8°C and do not freeze.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever

"""
# Example 1: Normal Character Text Splitter
# splitter1 = CharacterTextSplitter(
#     chunk_size = 100,
#     chunk_overlap = 0,
#     separator = "\n\n" # ["\n\n", "\n", ". ", " ", ""]
# )


# chunks1 = splitter1.split_text(clinical_text)

# for i, chunk in enumerate(chunks1, 1):
#     print(f"Chunk {i}: ({len(chunk)} chars)")
#     print(f'"{chunk}"')
#     print()

# Example 2: Recursive Character Text Splitter

print("\n" + "=" * 60)
print("2. RECURSIVE CHARACTER TEXT SPLITTER SOLUTION")
print("="*60)

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size = 100,
    chunk_overlap = 0
)

chunk2 = recursive_splitter.split_text(clinical_text)
print("Same problem text, but with RecursiveCharacterTextSplitter:")
for i, chunk in enumerate(chunk2, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()