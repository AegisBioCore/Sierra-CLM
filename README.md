from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from semantic_type_system import SemanticTypeSystem
from semantic_construction import (
    SemanticConstruction,
    SemanticFrame,
    StructuralParticipant,
)
from reference_resolution import (
    ReferenceContext,
    ReferenceResolver,
    ResolvedFrame,
)
from context_cell import (
    ContextManager,
    ContextState,
    OpenContext,
    ClosedContextCell,
)
from validation import Validator, ValidationResult, ValidationStatus
from sop import SOP, SOPAction, SOPDecision
from lookup import LocalMemory, LookupEngine, LookupRequest, LookupStatus
from webgate import (
    WebGate,
    WebGateRequest,
    WebGateDecisionType,
    WebQuery,
    WebQueryRequest,
)
from commit import Commit, CommitRequest, CommitResult
from response_composer import (
    ResponseComposer,
    ExecutionResult,
    StructuredResponse,
    ResponseType,
)


# ============================================================
# TURN RESULT  — aggregated output of one pipeline turn
# ============================================================

@dataclass
class TurnResult:
    """
    Complete result of one pipeline turn.

    response      : the StructuredResponse to return to the caller
    sop_action    : the SOPAction that was executed
    commit_result : populated if ASSERT was executed
    lookup_result : populated if LOOKUP was executed
    closed_cell   : populated if a ClosedContextCell was created
    context_state : the OpenContext state at the end of the turn
                    (None if the context was closed)
    """
    response      : StructuredResponse
    sop_action    : SOPAction
    commit_result : Optional[CommitResult]        = None
    lookup_result : Any                           = None   # LookupResult | None
    closed_cell   : Optional[ClosedContextCell]   = None
    context_state : Optional[ContextState]        = None


# ============================================================
# CONTEXT BINDING POLICY
# ============================================================

class ContextBindingPolicy:
    """
    Defines when and how the ReferenceContext is updated.

    Rules:
      1. After a successful ASSERT commit: the first StructuralParticipant
         in the frame becomes the new active reference.
         Operates on StructuralParticipant.norm — no domain knowledge.
      2. After a QUERY that resolves via LOOKUP: active_ref is NOT changed
         (a query does not assert a new entity into context).
      3. After CLARIFY / REJECT: context is NOT changed.

    This class does NOT call set_active() from tests.
    It is called exclusively by SierraPipeline._handle_assert().
    """

    @staticmethod
    def should_update_active(
        commit_result : CommitResult,
        frame         : SemanticFrame,
    ) -> bool:
        """
        Return True when active_ref should be updated.

        Conditions (ALL must hold):
          - commit_result.status == "OK"
          - frame.operation == "ASSERT"
          - frame has at least one StructuralParticipant
        """
        return (
            commit_result.status == "OK"
            and frame.operation == "ASSERT"
            and bool(frame.participants)
        )

    @staticmethod
    def candidate_for_active(frame: SemanticFrame) -> Optional[str]:
        """
        Return the norm of the first StructuralParticipant, if any.
        This is the value that will become active_ref.
        No domain knowledge; operates only on the structural type.
        """
        if frame.participants:
            return frame.participants[0].norm
        return None


# ============================================================
# SIERRA PIPELINE ORCHESTRATOR
# ============================================================

class SierraPipeline:
    """
    Connects all contracts into a working pipeline.

    One instance per session (stateful: ReferenceContext, ContextManager,
    LocalMemory persist across turns).

    Public API:
        result: TurnResult = pipeline.process(raw_text: str)
    """

    def __init__(self) -> None:
        # Stage instances (stateless)
        self._ts     = SemanticTypeSystem()
        self._sc     = SemanticConstruction()
        self._resolver = ReferenceResolver()
        self._validator= Validator()
        self._sop    = SOP()
        self._webgate  = WebGate()
        self._webquery = WebQuery()
        self._commit   = Commit()
        self._composer = ResponseComposer()
        self._policy   = ContextBindingPolicy()

        # Session state
        self._ref_ctx  = ReferenceContext()
        self._ctx_mgr  = ContextManager()
        self._memory   = LocalMemory()
        self._lookup   = LookupEngine(self._memory)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(self, raw_text: str) -> TurnResult:
        """
        Process one user utterance through the full pipeline.
        """
        existing = self._ctx_mgr.open
        if existing is not None and existing.state in (
            ContextState.RESOLVING, ContextState.AMBIGUOUS
        ):
            ctx = existing
        else:
            ctx = self._ctx_mgr.open_turn()

        analysed = self._ts.analyse(raw_text)
        frame = self._sc.build(analysed)

        # Bind a conversational role from an explicit declaration before
        # reference resolution. This is structural: if a frame contains a
        # SPEAKER reference and an explicit participant, the participant is
        # the declared referent of SPEAKER. No sentence-specific rule is used.
        self._bind_explicit_role(frame)
        ctx.set_frame(frame)

        rf = self._resolver.resolve(frame, self._ref_ctx)
        ctx.set_resolved(rf)

        vr = self._validator.validate(rf)
        self._sync_context_state(ctx, vr)

        decision = self._sop.decide(vr, rf, ctx.state)
        return self._route(decision, rf, vr, ctx)

    # ------------------------------------------------------------------
    # Structural role binding
    # ------------------------------------------------------------------

    def _bind_explicit_role(self, frame: SemanticFrame) -> None:
        """
        Bind an explicitly declared conversational role before resolution.

        The rule is deliberately domain-neutral: a role reference can only
        be bound from an explicit structural participant in the same frame.
        It does not inspect the utterance text and does not contain examples
        such as ``eu sunt adrian``.
        """
        if not frame.references or not frame.participants:
            return

        participant = frame.participants[0].norm
        if not participant:
            return

        for ref in frame.references:
            kind = getattr(ref, "kind", None)
            kind_text = str(getattr(kind, "value", kind)).strip().upper()
            if kind_text == "SPEAKER":
                self._ref_ctx.speaker = participant
                return

    # ------------------------------------------------------------------
    # Context state synchronisation
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_context_state(ctx: OpenContext, vr: ValidationResult) -> None:
        if vr.status == ValidationStatus.INVALID:
            if ctx.state != ContextState.INVALID:
                ctx.transition(ContextState.INVALID, "Validation: " + "; ".join(vr.violations))
            return

        if vr.status == ValidationStatus.INCOMPLETE:
            if ctx.state != ContextState.RESOLVING:
                ctx.transition(ContextState.RESOLVING, "Validation: incomplete — " + ", ".join(vr.missing))
            return

        if vr.status == ValidationStatus.VALID and ctx.state != ContextState.READY:
            ctx.transition(ContextState.READY, "Validation: VALID")

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(
        self,
        decision : SOPDecision,
        rf       : ResolvedFrame,
        vr       : ValidationResult,
        ctx      : OpenContext,
    ) -> TurnResult:
        action = decision.action

        if action == SOPAction.ASSERT:
            return self._handle_assert(decision, rf, vr, ctx)
        if action == SOPAction.LOOKUP:
            return self._handle_lookup(decision, rf, vr, ctx)
        if action == SOPAction.CLARIFY:
            return self._handle_clarify(decision, vr, ctx)
        if action == SOPAction.WEBGATE:
            return self._handle_webgate(decision, rf, vr, ctx)
        if action == SOPAction.REJECT:
            return self._handle_reject(decision, vr, ctx)

        resp = StructuredResponse(
            response_type = ResponseType.ERROR,
            message       = f"Unhandled SOP action: {action}",
        )
        return TurnResult(
            response      = resp,
            sop_action    = action,
            context_state = ctx.state,
        )

    # ------------------------------------------------------------------
    # ASSERT handler
    # ------------------------------------------------------------------

    def _handle_assert(
        self,
        decision : SOPDecision,
        rf       : ResolvedFrame,
        vr       : ValidationResult,
        ctx      : OpenContext,
    ) -> TurnResult:
        cr = self._commit.execute(CommitRequest(
            resolved_frame = rf,
            validation     = vr,
            memory         = self._memory,
        ))

        exec_r = ExecutionResult(
            sop_decision  = decision,
            commit_result = cr,
        )
        response = self._composer.compose(exec_r, vr)

        if cr.status == "OK":
            if self._policy.should_update_active(cr, rf.frame):
                candidate = self._policy.candidate_for_active(rf.frame)
                if candidate:
                    self._ref_ctx.set_active(candidate)

            cell = self._close_turn("ASSERTED")
            return TurnResult(
                response      = response,
                sop_action    = SOPAction.ASSERT,
                commit_result = cr,
                closed_cell   = cell,
                context_state = None,
            )
        else:
            return TurnResult(
                response      = response,
                sop_action    = SOPAction.ASSERT,
                commit_result = cr,
                context_state = ctx.state,
            )

    # ------------------------------------------------------------------
    # LOOKUP handler
    # ------------------------------------------------------------------

    def _handle_lookup(
        self,
        decision : SOPDecision,
        rf       : ResolvedFrame,
        vr       : ValidationResult,
        ctx      : OpenContext,
    ) -> TurnResult:
        lr = self._lookup.lookup(self._build_lookup_request(rf))

        if lr.status == LookupStatus.NO_MATCH:
            webgate_decision = self._sop.escalate_to_webgate(
                "Local lookup returned NO_MATCH"
            )
            return self._handle_webgate(webgate_decision, rf, vr, ctx)

        exec_r = ExecutionResult(
            sop_decision  = decision,
            lookup_result = lr,
        )
        response = self._composer.compose(exec_r, vr)

        cell = self._close_turn("ANSWERED")
        closed = cell is not None
        return TurnResult(
            response      = response,
            sop_action    = SOPAction.LOOKUP,
            lookup_result = lr,
            closed_cell   = cell,
            context_state = None if closed else ctx.state,
        )

    # ------------------------------------------------------------------
    # CLARIFY handler
    # ------------------------------------------------------------------

    def _handle_clarify(
        self,
        decision : SOPDecision,
        vr       : ValidationResult,
        ctx      : OpenContext,
    ) -> TurnResult:
        exec_r = ExecutionResult(sop_decision=decision)
        response = self._composer.compose(exec_r, vr)

        return TurnResult(
            response      = response,
            sop_action    = SOPAction.CLARIFY,
            context_state = ctx.state,
        )

    # ------------------------------------------------------------------
    # WEBGATE handler
    # ------------------------------------------------------------------

    def _handle_webgate(
        self,
        decision : SOPDecision,
        rf       : ResolvedFrame,
        vr       : ValidationResult,
        ctx      : OpenContext,
    ) -> TurnResult:
        frame = rf.frame

        participant_str = (
            frame.participants[0].norm if frame.participants
            else (
                self._ref_ctx.active
                or self._ref_ctx.speaker
                or None
            )
        )
        predicate_str = frame.predicates[0].norm if frame.predicates else None
        version_str   = (
            frame.version_specifier.surface
            if frame.version_specifier else None
        )

        wg_req = WebGateRequest(
            participant = participant_str,
            predicate   = predicate_str,
            version     = version_str,
            raw_query   = frame.source_spans[0].surface if frame.source_spans else "",
        )
        wg_decision = self._webgate.check(wg_req)

        if wg_decision.decision == WebGateDecisionType.DENY:
            exec_r = ExecutionResult(
                sop_decision = SOPDecision(
                    action  = SOPAction.REJECT,
                    reason  = f"WebGate denied: {wg_decision.reason}",
                ),
            )
            response = self._composer.compose(exec_r, vr)
            return TurnResult(
                response      = response,
                sop_action    = SOPAction.WEBGATE,
                context_state = ctx.state,
            )

        wq_req = WebQueryRequest(
            query   = wg_req.raw_query,
            context = {
                "participant": participant_str,
                "predicate":   predicate_str,
            },
        )
        wq_result = self._webquery.query(wq_req)

        exec_r = ExecutionResult(
            sop_decision  = decision,
            webquery_data = wq_result.retrieved_data,
        )
        response = self._composer.compose(exec_r, vr)

        return TurnResult(
            response      = response,
            sop_action    = SOPAction.WEBGATE,
            context_state = ctx.state,
            closed_cell   = None,
        )

    # ------------------------------------------------------------------
    # REJECT handler
    # ------------------------------------------------------------------

    def _handle_reject(
        self,
        decision : SOPDecision,
        vr       : ValidationResult,
        ctx      : OpenContext,
    ) -> TurnResult:
        exec_r = ExecutionResult(sop_decision=decision)
        response = self._composer.compose(exec_r, vr)
        return TurnResult(
            response      = response,
            sop_action    = SOPAction.REJECT,
            context_state = ctx.state,
        )

    # ------------------------------------------------------------------
    # LookupRequest builder
    # ------------------------------------------------------------------

    def _build_lookup_request(self, rf: ResolvedFrame) -> LookupRequest:
        frame = rf.frame

        participant_norm: Optional[str] = None
        if frame.participants:
            participant_norm = frame.participants[0].norm
        elif frame.references:
            rr = rf.resolved.get(0)
            if rr and rr.referent:
                participant_norm = rr.referent.lower()
        if participant_norm is None:
            participant_norm = self._ref_ctx.active or ""

        predicate_norm: Optional[str] = (
            frame.predicates[0].norm.rstrip(".,!?;:")
            if frame.predicates else None
        )

        version_str: Optional[str] = (
            frame.version_specifier.surface
            if frame.version_specifier else None
        )

        return LookupRequest(
            participant = participant_norm,
            predicate   = predicate_norm,
            version     = version_str,
        )

    # ------------------------------------------------------------------
    # Context lifecycle helpers
    # ------------------------------------------------------------------

    def _close_turn(self, outcome: str) -> Optional[ClosedContextCell]:
        ctx = self._ctx_mgr.open
        if ctx is None:
            return None
        if not ctx.is_closeable():
            return None
        try:
            return self._ctx_mgr.close_turn(outcome)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Accessors for inspection / testing
    # ------------------------------------------------------------------

    @property
    def reference_context(self) -> ReferenceContext:
        return self._ref_ctx

    @property
    def context_manager(self) -> ContextManager:
        return self._ctx_mgr

    @property
    def memory(self) -> LocalMemory:
        return self._memory


# ============================================================
# INTERACTIVE ADVERSARIAL CLI WRAPPER
# ============================================================

def run_adversarial_cli() -> None:
    pipeline = SierraPipeline()
    
    print("=" * 60)
    print(" SIERRA CLM PIPELINE — ADVERSARIAL & INTERACTIVE CLI")
    print("=" * 60)
    print(" Comenzi disponibile:")
    print("   - 'exit' sau 'quit' : Încheie sesiunea.")
    print("   - 'state'           : Afișează starea curentă a memoriei și contextului.")
    print("   - Orice alt text    : Este procesat prin pipeline-ul M1-M5.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nsierra-user> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Sesiune întreruptă de utilizator.]")
            break

        if not user_input:
            continue
            
        low_input = user_input.lower()
        if low_input in ("exit", "quit"):
            print("sierra> La revedere! Închidere sesiune CLM.")
            break
            
        if low_input == "state":
            print(f"  [Debug State] ReferenceContext.speaker = {pipeline.reference_context.speaker}")
            print(f"  [Debug State] ReferenceContext.addressee = {pipeline.reference_context.addressee}")
            print(f"  [Debug State] ReferenceContext.active = {pipeline.reference_context.active}")
            print(f"  [Debug State] OpenContext State = {pipeline.context_manager.open}")
            print(f"  [Debug State] LocalMemory Facts = {len(pipeline.memory.knowledge)}")
            continue

        try:
            result = pipeline.process(user_input)
            
            msg = result.response.message if hasattr(result.response, 'message') else str(result.response)
            print(f"sierra> {msg}")
            
            print(f"  ├── [SOP Action]   : {result.sop_action.name if hasattr(result.sop_action, 'name') else result.sop_action}")
            print(f"  ├── [Context State]: {result.context_state}")
            if result.commit_result:
                print(f"  ├── [Commit Status]: {result.commit_result.status}")
            if result.lookup_result:
                print(f"  ├── [Lookup Data]  : {result.lookup_result}")
            if result.closed_cell:
                print(f"  └── [Cell Closed]  : Outcome -> {result.closed_cell.outcome}")
            else:
                print(f"  └── [Cell Status]  : Open / Active Context Maintained")

        except Exception as e:
            print(f"  [Eroare Pipeline] Execuția a întâmpinat o excepție: {e}")


if __name__ == "__main__":
    run_adversarial_cli()
