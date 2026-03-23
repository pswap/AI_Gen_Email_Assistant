from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from email_pipeline.orchestrator import run_email_pipeline
from email_pipeline.schemas import UserEmailContext

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    PDF_EXPORT_AVAILABLE = True
except Exception:
    PDF_EXPORT_AVAILABLE = False

def _prior_messages_from_chat(max_messages: int = 12) -> list[str]:
    out: list[str] = []
    for msg in st.session_state.get("chat_messages", [])[-max_messages:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            out.append(f"{role}: {content}")
    return out


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

st.set_page_config(page_title="Multi-agent Email Assistant", page_icon="MAIL", layout="centered")
st.title("Multi-agent Email Assistant")
st.caption("Convert your intent + context into a personalized, polished email (generate-only).")

if not OPENAI_API_KEY:
    st.warning("Set `OPENAI_API_KEY` in `.env` to use OpenAI.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

with st.sidebar:
    st.header("Context (optional)")
    recipient_name = st.text_input("Recipient name")
    recipient_role = st.text_input("Recipient role / title")
    sender_name = st.text_input("Your name")
    sender_role = st.text_input("Your role / title")
    company = st.text_input("Company / org")
    product_or_service = st.text_input("Product / service")
    tone_choice = st.selectbox(
        "Tone selector",
        options=["auto", "formal", "friendly", "assertive", "neutral", "casual"],
        index=0,
    )
    tone_hint = st.text_input("Tone hint (optional)", placeholder="e.g. professional, friendly")
    length_hint = st.text_input("Length hint (optional)", placeholder="e.g. short, 1 paragraph")

    key_points_text = st.text_area(
        "Key points (one per line)",
        placeholder="- include the project update\n- propose next steps\n- mention timeline",
        height=120,
    )

    constraints_text = st.text_area(
        "Constraints (one per line, optional)",
        placeholder="- keep it under 140 words\n- use a warm, professional tone",
        height=90,
    )


def _split_lines(text: str) -> list[str]:
    items: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        items.append(line)
    return items


def _build_context() -> UserEmailContext:
    key_points = _split_lines(key_points_text)
    constraints = _split_lines(constraints_text)

    # Softly map UI hints into constraints. The intent agent can still override tone.
    if tone_choice != "auto":
        constraints.append(f"Tone preference: {tone_choice}")
    if tone_hint.strip():
        constraints.append(f"Tone preference: {tone_hint.strip()}")
    if length_hint.strip():
        constraints.append(f"Length preference: {length_hint.strip()}")

    return UserEmailContext(
        recipient_name=recipient_name.strip() or None,
        recipient_role=recipient_role.strip() or None,
        sender_name=sender_name.strip() or None,
        sender_role=sender_role.strip() or None,
        company=company.strip() or None,
        product_or_service=product_or_service.strip() or None,
        key_points=key_points,
        constraints=constraints,
    )


def _make_email_download_text(subject: str, body: str) -> str:
    return f"Subject: {subject}\n\n{body}\n"


def _make_pdf_bytes(subject: str, body: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"Subject: {subject}")
    y -= 30
    c.setFont("Helvetica", 11)
    for line in body.splitlines():
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)
        c.drawString(40, y, line[:150])
        y -= 16
    c.save()
    buffer.seek(0)
    return buffer.read()


def _collect_prior_context(max_messages: int = 12, max_edits: int = 8) -> list[str]:
    context_lines: list[str] = []
    context_lines.extend(_prior_messages_from_chat(max_messages=max_messages))
    for edit in st.session_state.get("draft_edit_log", [])[-max_edits:]:
        context_lines.append(
            f"edit_feedback: changed subject to '{edit.get('subject','')[:80]}' and updated body."
        )
    return context_lines


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False

if "pending_base_user_message" not in st.session_state:
    st.session_state.pending_base_user_message = None

if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None

if "last_generated_user_message" not in st.session_state:
    st.session_state.last_generated_user_message = None
if "latest_subject" not in st.session_state:
    st.session_state.latest_subject = ""
if "latest_body" not in st.session_state:
    st.session_state.latest_body = ""
if "draft_edit_log" not in st.session_state:
    st.session_state.draft_edit_log = []

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

regen = st.button(
    "Regenerate email",
    disabled=bool(st.session_state.awaiting_clarification) or not st.session_state.last_generated_user_message,
)

if regen:
    base = st.session_state.last_generated_user_message
    ctx = _build_context()
    result = run_email_pipeline(
        client=client,
        model=OPENAI_MODEL,
        user_message=base,
        context=ctx,
        prior_messages=_collect_prior_context(),
        max_iterations=3,
        temperature=0.6,
    )
    st.session_state.pipeline_state = result.model_dump()
    st.session_state.awaiting_clarification = False
    st.session_state.last_generated_user_message = base

    if result.status == "complete" and result.polished:
        subject = result.polished.subject.strip()
        body = result.polished.body.strip()
        st.session_state.latest_subject = subject
        st.session_state.latest_body = body
        assistant_content = f"**Subject:** {subject}\n\n{body}"
    else:
        assistant_content = result.clarifying_question or "Could not generate the email. Please try again."

    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_content})
    with st.chat_message("assistant"):
        st.markdown(assistant_content)


user_text = st.chat_input("Describe what you want the email to achieve...")
if user_text:
    st.session_state.chat_messages.append({"role": "user", "content": user_text})

    ctx = _build_context()

    composed_user_message = user_text
    if st.session_state.awaiting_clarification and st.session_state.pending_base_user_message:
        composed_user_message = (
            f"{st.session_state.pending_base_user_message}\n\n"
            f"User clarification: {user_text}"
        )

    with st.chat_message("assistant"):
        try:
            result = run_email_pipeline(
                client=client,
                model=OPENAI_MODEL,
                user_message=composed_user_message,
                context=ctx,
                prior_messages=_collect_prior_context(),
                max_iterations=3,
                temperature=0.3,
            )
        except Exception as e:
            assistant_content = f"Error generating email: `{e}`"
            st.markdown(assistant_content)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": assistant_content}
            )
        else:
            st.session_state.pipeline_state = result.model_dump()

            if result.status == "needs_clarification":
                st.session_state.awaiting_clarification = True
                st.session_state.pending_base_user_message = (
                    st.session_state.pending_base_user_message
                    if st.session_state.pending_base_user_message
                    else user_text
                )
                assistant_content = result.clarifying_question or "Please provide more details."
                st.markdown(assistant_content)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": assistant_content}
                )
            else:
                st.session_state.awaiting_clarification = False
                st.session_state.pending_base_user_message = None
                st.session_state.last_generated_user_message = (
                    st.session_state.last_generated_user_message or user_text
                )

                if result.polished:
                    subject = result.polished.subject.strip()
                    body = result.polished.body.strip()
                    st.session_state.latest_subject = subject
                    st.session_state.latest_body = body
                    assistant_content = f"**Subject:** {subject}\n\n{body}"
                else:
                    assistant_content = "No email generated. Please try again."

                st.markdown(assistant_content)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": assistant_content}
                )

if st.session_state.latest_subject or st.session_state.latest_body:
    st.subheader("Email preview and editor")
    edited_subject = st.text_input("Edit subject", value=st.session_state.latest_subject)
    edited_body = st.text_area("Edit body", value=st.session_state.latest_body, height=260)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Save edits"):
            st.session_state.latest_subject = edited_subject.strip()
            st.session_state.latest_body = edited_body.strip()
            st.session_state.draft_edit_log.append(
                {"subject": st.session_state.latest_subject, "body": st.session_state.latest_body}
            )
            st.success("Draft edits saved. Future suggestions will consider these edits.")
    with col_b:
        st.download_button(
            "Export .txt",
            data=_make_email_download_text(edited_subject, edited_body),
            file_name="generated_email.txt",
            mime="text/plain",
        )
    with col_c:
        if PDF_EXPORT_AVAILABLE:
            st.download_button(
                "Export PDF",
                data=_make_pdf_bytes(edited_subject, edited_body),
                file_name="generated_email.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("PDF export unavailable (install `reportlab`).")

    st.download_button(
        "Export .eml",
        data=(
            "From: \n"
            "To: \n"
            f"Subject: {edited_subject}\n"
            "Content-Type: text/plain; charset=utf-8\n\n"
            f"{edited_body}\n"
        ),
        file_name="generated_email.eml",
        mime="message/rfc822",
    )


if st.session_state.pipeline_state and st.session_state.pipeline_state.get("status") == "complete":
    with st.expander("Pipeline details (intermediate artifacts)"):
        ps = st.session_state.pipeline_state
        st.json(
            {
                "parsed_input": ps.get("parsed_input"),
                "intent_classification": ps.get("intent_classification"),
                "tone_spec": ps.get("tone_spec"),
                "context_pack": ps.get("context_pack"),
                "draft": ps.get("draft"),
                "polished": ps.get("polished"),
                "review_feedback": ps.get("review_feedback"),
                "memory_log": ps.get("memory_log"),
                "routing_decisions": ps.get("routing_decisions"),
                "profile_memory": ps.get("profile_memory"),
            }
        )

