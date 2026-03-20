from __future__ import annotations

import json
from typing import List, Optional

from openai import OpenAI

from . import prompts
from .schemas import (
    ContextPack,
    EmailDraft,
    IntentClassification,
    ParsedInput,
    PersonalizedEmail,
    ReviewResult,
    RoutingDecision,
    ToneStyleSpec,
    UserEmailContext,
)


class AgentError(RuntimeError):
    pass


class OpenAIResponseAgent:
    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def _parse(self, *, messages: list[dict], response_model: type, temperature: float) -> object:
        completion = self._client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=response_model,
            temperature=temperature,
        )
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise AgentError(f"Model refused: {message.refusal}")
        if not getattr(message, "parsed", None):
            raise AgentError("Model did not return a parsed structured response.")
        return message.parsed


class InputParsingAgent(OpenAIResponseAgent):
    """Validates prompt, extracts intent summary, recipient, tone, constraints."""

    def run(
        self,
        *,
        user_message: str,
        context: UserEmailContext,
        temperature: float,
    ) -> ParsedInput:
        payload = {"user_message": user_message, "provided_context": context.model_dump()}
        messages = [
            {"role": "system", "content": prompts.INPUT_PARSING_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]
        return self._parse(
            messages=messages,
            response_model=ParsedInput,
            temperature=temperature,
        )


class IntentDetectionAgent(OpenAIResponseAgent):
    """Classifies intent: outreach, follow-up, apology, info, etc."""

    def run(
        self,
        *,
        user_message: str,
        parsed: ParsedInput,
        context: UserEmailContext,
        temperature: float,
    ) -> IntentClassification:
        payload = {
            "user_message": user_message,
            "parsed_input": parsed.model_dump(),
            "provided_context": context.model_dump(),
        }
        messages = [
            {"role": "system", "content": prompts.INTENT_DETECTION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]
        return self._parse(
            messages=messages,
            response_model=IntentClassification,
            temperature=temperature,
        )


class ToneStylistAgent(OpenAIResponseAgent):
    """Adjusts tone using tokenized prompts (tone_tokens + composed directive)."""

    def run(
        self,
        *,
        parsed: ParsedInput,
        intent: IntentClassification,
        context: UserEmailContext,
        temperature: float,
    ) -> ToneStyleSpec:
        payload = {
            "parsed_input": parsed.model_dump(),
            "intent_classification": intent.model_dump(),
            "provided_context": context.model_dump(),
        }
        messages = [
            {"role": "system", "content": prompts.TONE_STYLIST_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]
        return self._parse(
            messages=messages,
            response_model=ToneStyleSpec,
            temperature=temperature,
        )


class DraftWriterAgent(OpenAIResponseAgent):
    """Generates main body with structure and clarity."""

    def run(
        self,
        *,
        parsed: ParsedInput,
        intent: IntentClassification,
        tone: ToneStyleSpec,
        context_pack: ContextPack,
        context: UserEmailContext,
        revision_fixes: Optional[List[str]] = None,
        temperature: float,
    ) -> EmailDraft:
        payload = {
            "parsed_input": parsed.model_dump(),
            "intent_classification": intent.model_dump(),
            "tone_style_spec": tone.model_dump(),
            "context_pack": context_pack.model_dump(),
            "provided_context": context.model_dump(),
            "revision_fixes": revision_fixes or [],
        }
        messages = [
            {"role": "system", "content": prompts.DRAFT_WRITER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Write the email draft.\n\n"
                    f"{json.dumps(payload, indent=2)}\n\n"
                    "If revision_fixes is non-empty, incorporate them."
                ),
            },
        ]
        return self._parse(
            messages=messages,
            response_model=EmailDraft,
            temperature=temperature,
        )


class PersonalizationAgent(OpenAIResponseAgent):
    """Injects user profile data and prior messages."""

    def run(
        self,
        *,
        draft: EmailDraft,
        parsed: ParsedInput,
        intent: IntentClassification,
        tone: ToneStyleSpec,
        context_pack: ContextPack,
        context: UserEmailContext,
        prior_messages: Optional[List[str]] = None,
        refinement_fixes: Optional[List[str]] = None,
        profile_memory: Optional[dict] = None,
        temperature: float,
    ) -> PersonalizedEmail:
        payload = {
            "draft": draft.model_dump(),
            "parsed_input": parsed.model_dump(),
            "intent_classification": intent.model_dump(),
            "tone_style_spec": tone.model_dump(),
            "context_pack": context_pack.model_dump(),
            "provided_context": context.model_dump(),
            "prior_messages": prior_messages or [],
            "refinement_fixes": refinement_fixes or [],
            "profile_memory": profile_memory or {},
        }
        messages = [
            {"role": "system", "content": prompts.PERSONALIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]
        return self._parse(
            messages=messages,
            response_model=PersonalizedEmail,
            temperature=temperature,
        )


class ReviewValidatorAgent(OpenAIResponseAgent):
    """Grammar, tone alignment, contextual coherence."""

    def run(
        self,
        *,
        parsed: ParsedInput,
        intent: IntentClassification,
        tone: ToneStyleSpec,
        context_pack: ContextPack,
        polished: PersonalizedEmail,
        temperature: float,
    ) -> ReviewResult:
        payload = {
            "parsed_input": parsed.model_dump(),
            "intent_classification": intent.model_dump(),
            "tone_style_spec": tone.model_dump(),
            "context_pack": context_pack.model_dump(),
            "email": polished.model_dump(),
        }
        messages = [
            {"role": "system", "content": prompts.REVIEW_VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]
        return self._parse(
            messages=messages,
            response_model=ReviewResult,
            temperature=temperature,
        )


class RoutingMemoryAgent(OpenAIResponseAgent):
    """Fallback routing decision + memory note (orchestrator also logs structurally)."""

    def run(
        self,
        *,
        summary: dict,
        temperature: float,
    ) -> RoutingDecision:
        messages = [
            {"role": "system", "content": prompts.ROUTING_MEMORY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(summary, indent=2)},
        ]
        return self._parse(
            messages=messages,
            response_model=RoutingDecision,
            temperature=temperature,
        )
