# AI Gen Email Assistant

A Python + Streamlit app that uses a multi-agent LLM pipeline to turn user intent and context into personalized, polished emails.

## Features

- Streamlit chat interface for email generation
- OpenAI-backed multi-agent orchestration
- LangGraph router flow for fallback + retry
- Structured outputs with Pydantic schemas
- Clarification flow when key details are missing
- Regenerate support and pipeline artifact visibility
- Tone selector, email preview/editor, and export options
- Edit-history logging to personalize subsequent suggestions

## Agent Pipeline

The app uses these agents:

1. **Input Parsing Agent** - Validates prompt and extracts goal, recipient, tone, and constraints.
2. **Intent Detection Agent** - Classifies intent (outreach, follow-up, apology, info, etc.).
3. **Tone Stylist Agent** - Builds tokenized tone instructions and a composed tone directive.
4. **Draft Writer Agent** - Generates a structured email draft.
5. **Personalization Agent** - Injects profile memory and prior-message context.
6. **Review & Validator Agent** - Checks grammar, tone alignment, and contextual coherence.
7. **Routing & Memory Agent** - Handles fallback routing decisions and logs memory notes.

## Project Structure

```text
.
├── .env
├── .gitignore
├── .dockerignore
├── Dockerfile
├── requirements.txt
└── src
    ├── app.py
    └── email_pipeline
        ├── __init__.py
        ├── agents.py
        ├── orchestrator.py
        ├── prompts.py
        └── schemas.py
```

## Requirements

- Python 3.10+ (recommended)
- OpenAI API key

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional PDF export dependency:

```bash
pip install reportlab
```

## Environment Variables

Set these in `.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

## Run the App

```bash
streamlit run src/app.py
```

Then open the local Streamlit URL shown in your terminal.

## How to Use

1. Fill optional context in the sidebar (recipient, sender, company, key points, constraints, tone selector).
2. Enter what you want the email to achieve in chat.
3. If asked a clarifying question, reply in chat.
4. Review generated subject/body.
5. Edit the draft in the **Email preview and editor** section and click **Save edits**.
6. Use **Regenerate email** for alternate drafts that consider prior chat plus saved edit history.
7. Export as `.txt`, `.eml`, or PDF (if `reportlab` is installed).
8. Expand **Pipeline details** to inspect intermediate artifacts.

## Docker (Optional)

Build image:

```bash
docker build -t ai-gen-email-assistant .
```

Run container:

```bash
docker run --rm -p 8501:8501 --env-file .env ai-gen-email-assistant
```

Then open [http://localhost:8501](http://localhost:8501).

## Notes

- This project is **generate-only** (it does not send emails).
- Keep `.env` private; do not commit secrets.
- If model output quality varies, tighten constraints and key points in the sidebar.
- PDF export button is shown only when `reportlab` is available.
