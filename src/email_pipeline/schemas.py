from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# --- User-facing context (from UI / profile) ---


class UserEmailContext(BaseModel):
    recipient_name: Optional[str] = None
    recipient_role: Optional[str] = None
    sender_name: Optional[str] = None
    sender_role: Optional[str] = None
    company: Optional[str] = None
    product_or_service: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(
        default_factory=list,
        description="Optional constraints like length, formality, or must-include items.",
    )


# --- 1) Input Parsing Agent ---


class ParsedInput(BaseModel):
    """Validates the user prompt and extracts structured fields."""

    prompt_valid: bool = Field(..., description="Whether the prompt is usable for email generation.")
    validation_issues: List[str] = Field(
        default_factory=list,
        description="Problems with the prompt (empty, abusive, off-topic, etc.).",
    )
    extracted_goal: str = Field(..., description="What the user wants the email to achieve.")
    recipient: Optional[str] = Field(
        default=None,
        description="Recipient name or role if inferable; else null.",
    )
    tone_preference: Optional[str] = Field(
        default=None,
        description="Requested tone if stated (e.g. formal, friendly).",
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Explicit constraints from the user message.",
    )
    missing_critical_info: List[str] = Field(
        default_factory=list,
        description="Critical missing pieces needed before drafting.",
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="If missing_critical_info is non-empty, one concise question.",
    )


# --- 2) Intent Detection Agent ---


class IntentCategory(str, Enum):
    outreach = "outreach"
    follow_up = "follow_up"
    apology = "apology"
    info = "info"
    meeting_request = "meeting_request"
    thank_you = "thank_you"
    introduction = "introduction"
    collaboration = "collaboration"
    other = "other"


class IntentClassification(BaseModel):
    intent_category: IntentCategory
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., description="Brief reason for the classification.")


# --- 3) Tone Stylist Agent (tokenized) ---


class ToneTarget(str, Enum):
    formal = "formal"
    friendly = "friendly"
    assertive = "assertive"
    neutral = "neutral"
    casual = "casual"


class ToneToken(BaseModel):
    """Single slot in a tokenized tone prompt."""

    slot: str = Field(..., description="Token name, e.g. greeting_register, hedging, cta_strength.")
    value: str = Field(..., description="Instruction text for that slot.")


class ToneStyleSpec(BaseModel):
    target_tone: ToneTarget
    tone_tokens: List[ToneToken] = Field(
        default_factory=list,
        description="Ordered tokenized style instructions for downstream agents.",
    )
    composed_tone_directive: str = Field(
        ...,
        description="Single paragraph summarizing tone for the Draft Writer.",
    )


# --- Shared bundle for drafting (assembled in orchestrator + optional LLM context) ---


class ContextPack(BaseModel):
    sender_persona: str
    recipient_persona: str
    company_or_brand: Optional[str] = None
    product_or_service: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    personalization_opportunities: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)


# --- 4) Draft Writer Agent ---


class EmailDraft(BaseModel):
    subject: str
    body: str
    greeting: str
    closing: str
    signature: str


# --- 5) Personalization Agent ---


class PersonalizedEmail(BaseModel):
    subject: str
    body: str


# --- 6) Review & Validator Agent ---


class FixTarget(str, Enum):
    polish = "polish"
    rewrite_draft = "rewrite_draft"


class ReviewCriterionScore(BaseModel):
    criterion: str
    score: int = Field(..., ge=0, le=10)
    notes: Optional[str] = None


class ReviewResult(BaseModel):
    pass_: bool = Field(..., description="Whether the email passes validation.")
    score: int = Field(..., ge=0, le=100)
    grammar_ok: bool = Field(..., description="Grammar and mechanics acceptable.")
    tone_aligned: bool = Field(..., description="Tone matches intent and tone spec.")
    context_coherent: bool = Field(..., description="Coherent with stated facts and context.")
    criteria: List[ReviewCriterionScore] = Field(default_factory=list)
    blocking_issues: List[str] = Field(default_factory=list)
    fixes: List[str] = Field(default_factory=list)
    fix_target: FixTarget = Field(
        description="polish: light edit; rewrite_draft: regenerate draft from writer.",
    )


# --- 7) Routing & Memory Agent ---


class RoutingAction(str, Enum):
    proceed = "proceed"
    request_clarification = "request_clarification"
    retry_after_validation = "retry_after_validation"


class RoutingDecision(BaseModel):
    action: RoutingAction
    reason: str
    memory_note: str = Field(
        ...,
        description="One line to append to the session memory log.",
    )


class MemoryLogEntry(BaseModel):
    agent_name: str
    step: str
    detail: str


class EmailPipelineState(BaseModel):
    user_message: str
    context: UserEmailContext

    parsed_input: Optional[ParsedInput] = None
    intent_classification: Optional[IntentClassification] = None
    tone_spec: Optional[ToneStyleSpec] = None
    context_pack: Optional[ContextPack] = None
    draft: Optional[EmailDraft] = None
    polished: Optional[PersonalizedEmail] = None
    review_feedback: List[ReviewResult] = Field(default_factory=list)

    routing_decisions: List[RoutingDecision] = Field(default_factory=list)
    memory_log: List[MemoryLogEntry] = Field(default_factory=list)
    profile_memory: Dict[str, Any] = Field(default_factory=dict)

    status: Literal["needs_clarification", "complete"] = "complete"
    clarifying_question: Optional[str] = None
