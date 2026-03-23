from __future__ import annotations

from typing import List, Optional

from langgraph.graph import END, StateGraph
from openai import OpenAI
from typing_extensions import TypedDict

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


class PipelineGraphState(TypedDict):
    state: EmailPipelineState
    parsed: Optional[ParsedInput]
    intent: Optional[IntentClassification]
    tone: Optional[ToneStyleSpec]
    context_pack: Optional[ContextPack]
    draft: Optional[EmailDraft]
    polished: Optional[PersonalizedEmail]
    review: Optional[ReviewResult]
    critic_fixes: List[str]
    iteration: int


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
    if prior_messages:
        state.profile_memory["prior_messages"] = prior_messages[-20:]

    input_agent = InputParsingAgent(client, model)
    intent_agent = IntentDetectionAgent(client, model)
    tone_agent = ToneStylistAgent(client, model)
    draft_agent = DraftWriterAgent(client, model)
    personalize_agent = PersonalizationAgent(client, model)
    review_agent = ReviewValidatorAgent(client, model)
    routing_agent = RoutingMemoryAgent(client, model)

    def parse_input_node(gs: PipelineGraphState) -> PipelineGraphState:
        parsed = input_agent.run(user_message=user_message, context=context, temperature=temperature)
        gs["parsed"] = parsed
        gs["state"].parsed_input = parsed
        _log(gs["state"], "InputParsingAgent", "parsed", parsed.model_dump_json())
        return gs

    def parse_router(gs: PipelineGraphState) -> str:
        parsed = gs["parsed"]
        assert parsed is not None
        if not parsed.prompt_valid or parsed.missing_critical_info:
            summary = {
                "stage": "after_input_parsing",
                "prompt_valid": parsed.prompt_valid,
                "validation_issues": parsed.validation_issues,
                "missing_critical_info": parsed.missing_critical_info,
            }
            routing = routing_agent.run(summary=summary, temperature=0.2)
            gs["state"].routing_decisions.append(routing)
            _log(gs["state"], "RoutingMemoryAgent", routing.action.value, routing.memory_note)
            gs["state"].status = "needs_clarification"
            gs["state"].clarifying_question = (
                parsed.clarifying_question
                or "; ".join(parsed.validation_issues)
                or ("To proceed: " + ", ".join(parsed.missing_critical_info[:3]))
            )
            return "stop"
        return "continue"

    def intent_node(gs: PipelineGraphState) -> PipelineGraphState:
        parsed = gs["parsed"]
        assert parsed is not None
        intent = intent_agent.run(
            user_message=user_message,
            parsed=parsed,
            context=context,
            temperature=temperature,
        )
        gs["intent"] = intent
        gs["state"].intent_classification = intent
        _log(gs["state"], "IntentDetectionAgent", intent.intent_category.value, intent.rationale)
        return gs

    def tone_node(gs: PipelineGraphState) -> PipelineGraphState:
        parsed = gs["parsed"]
        intent = gs["intent"]
        assert parsed is not None and intent is not None
        context_pack = build_context_pack(parsed, intent, context)
        gs["context_pack"] = context_pack
        gs["state"].context_pack = context_pack
        _log(gs["state"], "Orchestrator", "context_pack", context_pack.model_dump_json())

        tone = tone_agent.run(parsed=parsed, intent=intent, context=context, temperature=temperature)
        gs["tone"] = tone
        gs["state"].tone_spec = tone
        _log(gs["state"], "ToneStylistAgent", tone.target_tone.value, tone.composed_tone_directive[:500])
        return gs

    def draft_node(gs: PipelineGraphState) -> PipelineGraphState:
        parsed = gs["parsed"]
        intent = gs["intent"]
        tone = gs["tone"]
        context_pack = gs["context_pack"]
        assert parsed is not None and intent is not None and tone is not None and context_pack is not None

        draft = draft_agent.run(
            parsed=parsed,
            intent=intent,
            tone=tone,
            context_pack=context_pack,
            context=context,
            revision_fixes=gs.get("critic_fixes") or None,
            temperature=temperature,
        )
        gs["draft"] = draft
        gs["state"].draft = draft
        gs["critic_fixes"] = []
        _log(gs["state"], "DraftWriterAgent", "draft", f"subject={draft.subject[:80]!r}")
        return gs

    def personalize_node(gs: PipelineGraphState) -> PipelineGraphState:
        parsed = gs["parsed"]
        intent = gs["intent"]
        tone = gs["tone"]
        context_pack = gs["context_pack"]
        draft = gs["draft"]
        assert all([parsed, intent, tone, context_pack, draft])

        polished = personalize_agent.run(
            draft=draft,
            parsed=parsed,
            intent=intent,
            tone=tone,
            context_pack=context_pack,
            context=context,
            prior_messages=prior_messages,
            refinement_fixes=gs.get("critic_fixes") or None,
            profile_memory=gs["state"].profile_memory,
            temperature=temperature,
        )
        gs["polished"] = polished
        gs["state"].polished = polished
        _log(
            gs["state"],
            "PersonalizationAgent",
            f"iteration_{gs['iteration']}",
            f"subject={polished.subject[:80]!r}",
        )
        return gs

    def review_node(gs: PipelineGraphState) -> PipelineGraphState:
        parsed = gs["parsed"]
        intent = gs["intent"]
        tone = gs["tone"]
        context_pack = gs["context_pack"]
        polished = gs["polished"]
        assert all([parsed, intent, tone, context_pack, polished])

        review = review_agent.run(
            parsed=parsed,
            intent=intent,
            tone=tone,
            context_pack=context_pack,
            polished=polished,
            temperature=0.2,
        )
        gs["review"] = review
        gs["state"].review_feedback.append(review)
        _log(gs["state"], "ReviewValidatorAgent", "review", f"pass={review.pass_} score={review.score}")
        return gs

    def review_router(gs: PipelineGraphState) -> str:
        review = gs["review"]
        assert review is not None
        if review.pass_:
            gs["state"].status = "complete"
            routing = routing_agent.run(summary={"stage": "review_pass", "score": review.score}, temperature=0.2)
            gs["state"].routing_decisions.append(routing)
            _log(gs["state"], "RoutingMemoryAgent", routing.action.value, routing.memory_note)
            return "done"

        gs["iteration"] += 1
        summary = {
            "stage": "review_fail",
            "iteration": gs["iteration"],
            "fix_target": review.fix_target.value,
            "blocking_issues": review.blocking_issues,
        }
        routing = routing_agent.run(summary=summary, temperature=0.2)
        gs["state"].routing_decisions.append(routing)
        _log(gs["state"], "RoutingMemoryAgent", routing.action.value, routing.memory_note)

        if gs["iteration"] >= max_iterations:
            gs["state"].status = "complete"
            return "done"

        gs["critic_fixes"] = review.fixes
        if review.fix_target == FixTarget.rewrite_draft:
            return "rewrite"
        gs["draft"] = _polished_to_draft(gs["polished"])
        return "retry_personalize"

    graph = StateGraph(PipelineGraphState)
    graph.add_node("parse_input", parse_input_node)
    graph.add_node("intent", intent_node)
    graph.add_node("tone", tone_node)
    graph.add_node("draft", draft_node)
    graph.add_node("personalize", personalize_node)
    graph.add_node("review", review_node)

    graph.set_entry_point("parse_input")
    graph.add_conditional_edges(
        "parse_input",
        parse_router,
        {"continue": "intent", "stop": END},
    )
    graph.add_edge("intent", "tone")
    graph.add_edge("tone", "draft")
    graph.add_edge("draft", "personalize")
    graph.add_edge("personalize", "review")
    graph.add_conditional_edges(
        "review",
        review_router,
        {"done": END, "rewrite": "draft", "retry_personalize": "personalize"},
    )

    app = graph.compile()
    final_graph_state = app.invoke(
        {
            "state": state,
            "parsed": None,
            "intent": None,
            "tone": None,
            "context_pack": None,
            "draft": None,
            "polished": None,
            "review": None,
            "critic_fixes": [],
            "iteration": 0,
        }
    )
    return final_graph_state["state"]
