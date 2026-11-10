from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Source: adapted from the CDC/ACIP dengue vaccine recommendations (MMWR 2021), a public-domain US government document.
clinical_text = """Dengvaxia Efficacy
Pooled vaccine efficacy against virologically confirmed dengue was 82% among seropositive children aged 9–16 years.
Efficacy was highest for DENV-4 (89%) and lowest for DENV-1 and DENV-2 (67% each).
The vaccine protected against hospitalization (79%) and severe dengue (84%) over five years of follow-up.

Safety and Adverse Reactions
The most frequently reported adverse reactions were headache (40%), injection site pain (32%), and myalgia (29%).
No dengue-related deaths occurred among participants in the trial.
Seronegative recipients had an increased risk of hospitalization and severe dengue after a first natural infection.

Storage and Administration
Store vaccine antigen and saline diluent at 2°C–8°C and do not freeze.
After reconstitution, administer Dengvaxia immediately or use within 30 minutes.
Dengvaxia is given as 3 doses 6 months apart, by subcutaneous injection only."""


# Semantic Chunker - groups by meaning, not structure
semantic_splitter = SemanticChunker(
    embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview"),
    breakpoint_threshold_type="percentile",  # or "standard_deveiation"
    breakpoint_threshold_amount=70,
)

chunks = semantic_splitter.split_text(clinical_text)

print("SEMANTIC CHUNKING RESULTS:")
print("=" * 50)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
