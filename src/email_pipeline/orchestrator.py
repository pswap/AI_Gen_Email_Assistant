from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from .agents import (
    DraftWriterAgent,
    InputParsingAgent,
    IntentDetectionAgent,
    PersonalizationAgent,
    ReviewValidatorAgent,
    RoutingMemoryAgent,
    ToneStylistAgent,
)
from .schemas import (
    ContextPack,
    EmailDraft,
    EmailPipelineState,
    FixTarget,
    IntentClassification,
    MemoryLogEntry,
    ParsedInput,
    PersonalizedEmail,
    ReviewResult,
    RoutingAction,
    ToneStyleSpec,
    UserEmailContext,
)


def _log(state: EmailPipelineState, agent_name: str, step: str, detail: str) -> None:
    state.memory_log.append(MemoryLogEntry(agent_name=agent_name, step=step, detail=detail[:2000]))


def _merge_profile_memory(context: UserEmailContext) -> dict:
    return {
        "recipient_name": context.recipient_name,
        "recipient_role": context.recipient_role,
        "sender_name": context.sender_name,
        "sender_role": context.sender_role,
        "company": context.company,
        "product_or_service": context.product_or_service,
        "key_points": context.key_points,
        "constraints": context.constraints,
    }


def build_context_pack(
    parsed: ParsedInput,
    intent: IntentClassification,
    context: UserEmailContext,
) -> ContextPack:
    rn = context.recipient_name or parsed.recipient
    rr = context.recipient_role
    recipient_bits = [x for x in [rn, rr] if x]
    recipient_persona = (
        ", ".join(recipient_bits) if recipient_bits else (parsed.recipient or "Recipient (unspecified)")
    )

    sn = context.sender_name
    sr = context.sender_role
    sender_bits = [x for x in [sn, sr] if x]
    sender_persona = ", ".join(sender_bits) if sender_bits else "Sender"

    constraints = list(dict.fromkeys([*context.constraints, *parsed.constraints]))

    return ContextPack(
        sender_persona=sender_persona,
        recipient_persona=recipient_persona,
        company_or_brand=context.company,
        product_or_service=context.product_or_service,
        key_points=list(context.key_points),
        personalization_opportunities=list(context.key_points) + [parsed.extracted_goal],
        constraints=constraints,
    )


def _polished_to_draft(p: PersonalizedEmail) -> EmailDraft:
    return EmailDraft(
        subject=p.subject,
        body=p.body,
        greeting="",
        closing="",
        signature="",
    )


def run_email_pipeline(
    *,
    client: OpenAI,
    model: str,
    user_message: str,
    context: UserEmailContext,
    prior_messages: Optional[List[str]] = None,
    max_iterations: int = 3,
    temperature: float = 0.3,
) -> EmailPipelineState:
    """
    Multi-agent pipeline:
    Input Parsing -> Intent Detection -> Tone Stylist -> Draft Writer ->
    Personalization -> Review & Validator loop, with Routing & Memory logging.
    """

    state = EmailPipelineState(user_message=user_message, context=context)
    state.profile_memory = _merge_profile_memory(context)

    input_agent = InputParsingAgent(client, model)
    intent_agent = IntentDetectionAgent(client, model)
    tone_agent = ToneStylistAgent(client, model)
    draft_agent = DraftWriterAgent(client, model)
    personalize_agent = PersonalizationAgent(client, model)
    review_agent = ReviewValidatorAgent(client, model)
    routing_agent = RoutingMemoryAgent(client, model)

    parsed: ParsedInput = input_agent.run(
        user_message=user_message,
        context=context,
        temperature=temperature,
    )
    state.parsed_input = parsed
    _log(state, "InputParsingAgent", "parsed", parsed.model_dump_json())

    if not parsed.prompt_valid:
        summary = {
            "stage": "after_input_parsing",
            "prompt_valid": False,
            "validation_issues": parsed.validation_issues,
        }
        routing = routing_agent.run(summary=summary, temperature=0.2)
        state.routing_decisions.append(routing)
        _log(state, "RoutingMemoryAgent", routing.action.value, routing.memory_note)
        state.status = "needs_clarification"
        state.clarifying_question = (
            parsed.clarifying_question
            or "; ".join(parsed.validation_issues)
            or "Please provide a clearer request for the email you want."
        )
        return state

    if parsed.missing_critical_info:
        summary = {
            "stage": "after_input_parsing",
            "missing_critical_info": parsed.missing_critical_info,
        }
        routing = routing_agent.run(summary=summary, temperature=0.2)
        state.routing_decisions.append(routing)
        _log(state, "RoutingMemoryAgent", routing.action.value, routing.memory_note)
        state.status = "needs_clarification"
        state.clarifying_question = parsed.clarifying_question or (
            "To proceed: " + ", ".join(parsed.missing_critical_info[:3])
        )
        return state

    intent: IntentClassification = intent_agent.run(
        user_message=user_message,
        parsed=parsed,
        context=context,
        temperature=temperature,
    )
    state.intent_classification = intent
    _log(state, "IntentDetectionAgent", intent.intent_category.value, intent.rationale)

    context_pack = build_context_pack(parsed, intent, context)
    state.context_pack = context_pack
    _log(state, "Orchestrator", "context_pack", context_pack.model_dump_json())

    tone: ToneStyleSpec = tone_agent.run(
        parsed=parsed,
        intent=intent,
        context=context,
        temperature=temperature,
    )
    state.tone_spec = tone
    _log(
        state,
        "ToneStylistAgent",
        tone.target_tone.value,
        tone.composed_tone_directive[:500],
    )

    draft = draft_agent.run(
        parsed=parsed,
        intent=intent,
        tone=tone,
        context_pack=context_pack,
        context=context,
        revision_fixes=None,
        temperature=temperature,
    )
    state.draft = draft
    _log(state, "DraftWriterAgent", "draft", f"subject={draft.subject[:80]!r}")

    critic_fixes: list[str] = []
    polished: Optional[PersonalizedEmail] = None

    for iteration in range(max_iterations):
        polished = personalize_agent.run(
            draft=draft,
            parsed=parsed,
            intent=intent,
            tone=tone,
            context_pack=context_pack,
            context=context,
            prior_messages=prior_messages,
            refinement_fixes=critic_fixes if critic_fixes else None,
            profile_memory=state.profile_memory,
            temperature=temperature,
        )
        state.polished = polished
        _log(
            state,
            "PersonalizationAgent",
            f"iteration_{iteration}",
            f"subject={polished.subject[:80]!r}",
        )

        review: ReviewResult = review_agent.run(
            parsed=parsed,
            intent=intent,
            tone=tone,
            context_pack=context_pack,
            polished=polished,
            temperature=0.2,
        )
        state.review_feedback.append(review)
        _log(
            state,
            "ReviewValidatorAgent",
            "review",
            f"pass={review.pass_} score={review.score}",
        )

        if review.pass_:
            state.status = "complete"
            summary = {"stage": "review_pass", "score": review.score}
            routing = routing_agent.run(summary=summary, temperature=0.2)
            state.routing_decisions.append(routing)
            _log(state, "RoutingMemoryAgent", routing.action.value, routing.memory_note)
            return state

        critic_fixes = review.fixes
        summary = {
            "stage": "review_fail",
            "iteration": iteration,
            "fix_target": review.fix_target.value,
            "blocking_issues": review.blocking_issues,
        }
        routing = routing_agent.run(summary=summary, temperature=0.2)
        state.routing_decisions.append(routing)
        _log(state, "RoutingMemoryAgent", routing.action.value, routing.memory_note)

        if review.fix_target == FixTarget.rewrite_draft:
            draft = draft_agent.run(
                parsed=parsed,
                intent=intent,
                tone=tone,
                context_pack=context_pack,
                context=context,
                revision_fixes=review.fixes,
                temperature=temperature,
            )
            state.draft = draft
            critic_fixes = []
        else:
            draft = _polished_to_draft(polished)

    state.status = "complete"
    return state
