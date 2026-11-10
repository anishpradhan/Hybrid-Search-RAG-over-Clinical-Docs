from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", max_retries=6)

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

# Create a prompt
prompt = f"""
Your are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{clinical_text}

Return the text with <<<SPLIT>>> markers where you want to split:
"""

# Get AI response
print("Asking AI to chunk the text...")
response = llm.invoke(prompt)
marked_text = response.content[0]['text']

# Split text at the markers
chunks = marked_text.split("<<<SPLIT>>>")

# Clean up the chunks (remove extra whitespace)
clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned: # Only keep non-empty chunks
        clean_chunks.append(cleaned)

# Show results
print("\n AGENTIC CHUNKING RESULTS:")
print("=" * 50)

for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()

