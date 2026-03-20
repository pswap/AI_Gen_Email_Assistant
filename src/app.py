from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from email_pipeline.orchestrator import run_email_pipeline
from email_pipeline.schemas import UserEmailContext


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
        prior_messages=_prior_messages_from_chat(),
        max_iterations=3,
        temperature=0.6,
    )
    st.session_state.pipeline_state = result.model_dump()
    st.session_state.awaiting_clarification = False
    st.session_state.last_generated_user_message = base

    if result.status == "complete" and result.polished:
        subject = result.polished.subject.strip()
        body = result.polished.body.strip()
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
                prior_messages=_prior_messages_from_chat(),
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
                    assistant_content = f"**Subject:** {subject}\n\n{body}"
                else:
                    assistant_content = "No email generated. Please try again."

                st.markdown(assistant_content)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": assistant_content}
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

