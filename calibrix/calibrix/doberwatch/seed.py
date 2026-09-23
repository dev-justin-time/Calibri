# SPDX-License-Identifier: MIT
# Doberwatch Finance — the hardcoded graded corpus and expert rubrics.
#
# This is the "answer library" retrieved *before* any model call. It ships in
# two scopes, split by the ``shareable`` flag on each entry:
#
#   * ANONYMOUS seed — anonymized, shareable reference answers. Any user can
#     retrieve these; they are the starting point for the ten closest matches.
#   * PRIVATE seed  — account-specific examples that stay on the owner's side
#     and are never published.
#
# HONESTY BOUND: every entry is ``evidence_status: reference_only`` — a
# rubric-reviewed *reference* answer, not a production-measured outcome. The
# grades in this corpus are computed from the rubric + criteria scores at load
# time (see ``core.Doberwatch``), so the library can never drift from its own
# rubric: change the rubric and the grade changes with it.
#
# CORROBORATION IS EARNED, NOT GRANTED. Each entry carries the independent
# witnesses behind its factual claims (``sources``): the governing document
# and a second, independently-worded record. At load time the loader fact-
# checks the response against those witnesses exactly as it would check a
# live answer, and the measured material-claim grade caps the rubric's
# corroboration criteria. An entry whose sources are missing, or whose
# witnesses disagree, therefore cannot reach PASS on review alone.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .grading import Criterion, Rubric
from .verification import Source

SEED_SCHEMA = "calibrix.doberwatch.seed/1.0"

# ----------------------------------------------------------------------
# Expert rubrics (one per domain/task)
# ----------------------------------------------------------------------
RUBRICS: Dict[str, Rubric] = {
    "billing_dispute.support_reply.v1": Rubric(
        rubric_id="billing_dispute.support_reply.v1",
        domain="billing_dispute", task="support_reply",
        criteria=(
            Criterion("acknowledges_charge", 0.15,
                      "States the disputed charge back to the user."),
            Criterion("cites_policy", 0.20,
                      "Names the billing policy provision that governs the case.",
                      requires_evidence=True, evidence_kind="policy_citation",
                      requires_corroboration=True),
            Criterion("states_remedy", 0.25, "Commits to a specific remedy."),
            Criterion("gives_timeline", 0.15, "Gives a concrete resolution window."),
            Criterion("offers_escalation_path", 0.15,
                      "Explains what happens if the remedy fails."),
            Criterion("no_unsupported_denial", 0.10,
                      "Does not deny the claim without stating a reason."),
        ),
        expected_evidence=("policy_citation",),
    ),
    "refund_request.support_reply.v1": Rubric(
        rubric_id="refund_request.support_reply.v1",
        domain="refund_request", task="support_reply",
        criteria=(
            Criterion("confirms_order", 0.15, "Identifies the exact order."),
            Criterion("states_eligibility", 0.20,
                      "Applies the refund policy to this order.",
                      requires_evidence=True, evidence_kind="order_receipt",
                      requires_corroboration=True),
            Criterion("states_remedy", 0.25, "Commits to refund or rework."),
            Criterion("gives_timeline", 0.15, "Gives a settlement window."),
            Criterion("no_hidden_conditions", 0.15,
                      "States any conditions up front rather than later."),
            Criterion("offers_escalation_path", 0.10,
                      "Explains the next step if the refund fails."),
        ),
        expected_evidence=("order_receipt",),
    ),
    "subscription_cancellation.support_reply.v1": Rubric(
        rubric_id="subscription_cancellation.support_reply.v1",
        domain="subscription_cancellation", task="support_reply",
        criteria=(
            Criterion("confirms_identity", 0.10, "Confirms the account acted on."),
            Criterion("confirms_cancellation", 0.25, "States cancellation plainly."),
            Criterion("states_last_billing_date", 0.20,
                      "Gives the final billing date.",
                      requires_evidence=True, evidence_kind="billing_statement",
                      requires_corroboration=True),
            Criterion("states_prorated_refund", 0.20,
                      "Says what happens to money already paid."),
            Criterion("gives_confirmation_reference", 0.15,
                      "Provides a traceable confirmation reference."),
            Criterion("no_retention_trap", 0.10,
                      "Does not bury the cancellation behind retention offers."),
        ),
        expected_evidence=("billing_statement",),
    ),
    "hidden_fees.support_reply.v1": Rubric(
        rubric_id="hidden_fees.support_reply.v1",
        domain="hidden_fees", task="support_reply",
        criteria=(
            Criterion("itemizes_fee", 0.25, "Itemizes the exact fee charged.",
                      requires_evidence=True, evidence_kind="fee_schedule",
                      requires_corroboration=True),
            Criterion("explains_contract_basis", 0.25,
                      "Points to the contract clause relied on.",
                      requires_evidence=True, evidence_kind="contract_clause",
                      requires_corroboration=True),
            Criterion("states_removal_remedy", 0.25,
                      "Commits to remove or refund the fee."),
            Criterion("gives_timeline", 0.15, "Gives a credit window."),
            Criterion("offers_escalation_path", 0.10,
                      "Explains escalation if the fee recurs."),
        ),
        expected_evidence=("fee_schedule", "contract_clause"),
    ),
    "credit_report_error.dispute_letter.v1": Rubric(
        rubric_id="credit_report_error.dispute_letter.v1",
        domain="credit_report_error", task="dispute_letter",
        criteria=(
            Criterion("identifies_account", 0.15, "Identifies the account."),
            Criterion("identifies_disputed_item", 0.20,
                      "Identifies the specific item in error."),
            Criterion("cites_legal_basis", 0.20,
                      "Cites the reporting rule that applies.",
                      requires_evidence=True, evidence_kind="law_citation",
                      requires_corroboration=True),
            Criterion("requests_correction", 0.25,
                      "Requests correction or deletion explicitly."),
            Criterion("gives_response_deadline", 0.10,
                      "States the bureau's response deadline."),
            Criterion("preserves_evidence", 0.10,
                      "Tells the user to keep copies and proof of delivery."),
        ),
        expected_evidence=("law_citation",),
    ),
    "debt_collection.response_letter.v1": Rubric(
        rubric_id="debt_collection.response_letter.v1",
        domain="debt_collection", task="response_letter",
        criteria=(
            Criterion("validates_debt", 0.25,
                      "Demands written validation of the debt.",
                      requires_evidence=True, evidence_kind="debt_validation",
                      requires_corroboration=True),
            Criterion("asserts_rights", 0.25,
                      "Asserts the user's rights under the collection rules.",
                      requires_evidence=True, evidence_kind="law_citation",
                      requires_corroboration=True),
            Criterion("requests_cease_contact", 0.20,
                      "Requests contact to stop until validation."),
            Criterion("gives_deadline", 0.15, "Sets a written deadline."),
            Criterion("preserves_records", 0.15,
                      "Tells the user to log calls and keep proof."),
        ),
        expected_evidence=("debt_validation", "law_citation"),
    ),
}

DEFAULT_RUBRIC_BY_DOMAIN: Dict[str, str] = {
    "billing_dispute": "billing_dispute.support_reply.v1",
    "refund_request": "refund_request.support_reply.v1",
    "subscription_cancellation": "subscription_cancellation.support_reply.v1",
    "hidden_fees": "hidden_fees.support_reply.v1",
    "credit_report_error": "credit_report_error.dispute_letter.v1",
    "debt_collection": "debt_collection.response_letter.v1",
}

# ----------------------------------------------------------------------
# Qualifying questions — asked ALWAYS, before any model call
# ----------------------------------------------------------------------
GENERIC_QUALIFYING_QUESTIONS: List[str] = [
    "What outcome would make this right for you?",
    "What is the exact amount, date, and reference number involved?",
    "Have you already contacted the provider, and what did they say?",
    "Which documents can you attach (receipt, statement, screenshot, letter)?",
    "Is there a deadline or a payment at risk in the next 30 days?",
]

QUALIFYING_QUESTIONS_BY_DOMAIN: Dict[str, List[str]] = {
    "billing_dispute": [
        "Is the disputed charge a duplicate, a price change, or an error?",
        "Which invoice or statement period does the charge fall in?",
    ],
    "refund_request": [
        "Was the problem damage, non-delivery, or a changed mind?",
        "When did you first report the problem to the seller?",
    ],
    "subscription_cancellation": [
        "Do you want cancellation immediately or at the end of the paid period?",
        "Has a payment been taken since you decided to cancel?",
    ],
    "hidden_fees": [
        "Was the fee disclosed anywhere before you were charged?",
        "How many billing periods has the fee been applied for?",
    ],
    "credit_report_error": [
        "Which bureau and which report item is wrong?",
        "Do you have proof the underlying payment or account is correct?",
    ],
    "debt_collection": [
        "How often, where, and when have they contacted you?",
        "Have you previously asked them in writing to stop or validate?",
    ],
}

# ----------------------------------------------------------------------
# Hardcoded graded corpus (grade computed from rubric + criteria_scores)
# ----------------------------------------------------------------------
GRADED_RESPONSES: List[Dict[str, Any]] = [
    # -- billing_dispute -------------------------------------------------
    {
        "entry_id": "seed-billing-1",
        "domain": "billing_dispute", "task": "support_reply",
        "rubric_id": "billing_dispute.support_reply.v1",
        "request": "I was billed twice this month for the same subscription and I want the duplicate charge removed.",
        "response": "I can see the duplicate charge and I'm sorry for the confusion. Under the billing "
                    "policy, confirmed duplicate charges are reversed within 5-7 business days. I've "
                    "submitted the reversal today under reference BILL-4471; the credit will appear on "
                    "your next statement. If it hasn't posted within 7 days, reply here and we'll "
                    "escalate to our billing supervisor with a written answer within 14 days.",
        "criteria_scores": {"acknowledges_charge": 1.0, "cites_policy": 0.9,
                            "states_remedy": 1.0, "gives_timeline": 1.0,
                            "offers_escalation_path": 0.9, "no_unsupported_denial": 1.0},
        "evidence": ["policy_citation: billing-policy#duplicate-charges"],
        "sources": [
            {"source_id": "billing-policy#duplicate-charges",
             "text": "Where the same charge is billed twice, the billing policy refunds the "
                     "duplicate within 5-7 business days.",
             "kind": "authority", "url": "policy://billing/duplicate-charges"},
            {"source_id": "case-note#acknowledgement",
             "text": "The duplicate charge is confirmed and we apologise for the confusion it "
                     "caused.",
             "kind": "document"},
            {"source_id": "account-record#duplicate-charge",
             "text": "The duplicate charge on this account and the confusion it caused are both "
                     "recorded.",
             "kind": "authority"},
            {"source_id": "case-note#policy-applied",
             "text": "Our case note applies the billing policy: the duplicate charge was "
                     "confirmed, so the refund is due within 5-7 business days.",
             "kind": "document"},
            {"source_id": "case-log-BILL-4471",
             "text": "The reversal was lodged today against case BILL-4471.",
             "kind": "document"},
            {"source_id": "billing-system#reference",
             "text": "Reference BILL-4471 holds the reversal requested by the customer.",
             "kind": "authority"},
            {"source_id": "billing-policy#escalation",
             "text": "A credit that has not posted after 7 days goes to the billing supervisor, "
                     "who answers in writing within 14 days.",
             "kind": "authority"},
            {"source_id": "case-log#escalation",
             "text": "The case log shows the credit was never posted in 7 days, so the billing "
                     "supervisor owes a written answer in 14 days.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
    {
        "entry_id": "seed-billing-2",
        "domain": "billing_dispute", "task": "support_reply",
        "rubric_id": "billing_dispute.support_reply.v1",
        "request": "Why was I charged a higher amount this invoice than last month?",
        "response": "The increase reflects the annual plan price change described in section 4 of your "
                    "agreement, and I've itemized the difference for you. Because the change was not "
                    "clearly notified, I can apply a one-time credit for the difference to your next "
                    "invoice within 3 business days. If you'd rather dispute the increase formally, I'll "
                    "route it to our billing supervisor for a written response within 14 days.",
        "criteria_scores": {"acknowledges_charge": 1.0, "cites_policy": 0.85,
                            "states_remedy": 0.95, "gives_timeline": 1.0,
                            "offers_escalation_path": 0.95, "no_unsupported_denial": 1.0},
        "evidence": ["policy_citation: agreement-section-4",
                     "billing_statement: itemized-delta"],
        "sources": [
            {"source_id": "agreement#section-4",
             "text": "Section 4 of the agreement sets out the annual plan price change behind "
                     "this rise.",
             "kind": "authority"},
            {"source_id": "billing-statement#delta",
             "text": "The statement itemizes the difference caused by the annual plan price "
                     "change.",
             "kind": "document"},
            {"source_id": "billing-policy#notice-credit",
             "text": "A price change that was not clearly notified earns a one-time credit "
                     "within 3 business days.",
             "kind": "authority"},
            {"source_id": "case-log#notice-credit",
             "text": "Our case log grants the one-time credit because the price rise was not "
                     "clearly notified.",
             "kind": "document"},
            {"source_id": "billing-policy#formal-dispute",
             "text": "A formal dispute goes to the billing supervisor, answered in writing "
                     "within 14 days.",
             "kind": "authority"},
            {"source_id": "case-log#dispute-route",
             "text": "The case log sends the formal dispute about the increase to the billing "
                     "supervisor, whose written response comes within 14 days.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
    {
        "entry_id": "seed-billing-3",
        "domain": "billing_dispute", "task": "support_reply",
        "rubric_id": "billing_dispute.support_reply.v1",
        "request": "My own account has three invoices in a row with a late fee I think is wrong.",
        "response": "I've pulled your three latest invoices and matched them against your plan. Two "
                    "include a late fee applied before your payment cleared, so I've reversed both, "
                    "under reference BILL-5502, settling within 5 business days. I'll also email a "
                    "corrected statement so your records match, and flag the account so the fee is not "
                    "re-applied.",
        "criteria_scores": {"acknowledges_charge": 1.0, "cites_policy": 0.8,
                            "states_remedy": 1.0, "gives_timeline": 0.9,
                            "offers_escalation_path": 0.7, "no_unsupported_denial": 1.0},
        "evidence": ["billing_statement: invoices-latest-three"],
        "sources": [
            {"source_id": "case-log-BILL-5502",
             "text": "Two of the three invoices carried a late fee charged before the payment "
                     "cleared; BILL-5502 reverses both within 5 business days.",
             "kind": "document"},
            {"source_id": "billing-policy#late-fee-reversal",
             "text": "A late fee that lands before the payment clears is refunded in 5 "
                     "business days.",
             "kind": "authority"},
            {"source_id": "billing-policy#fee-flag",
             "text": "A corrected statement is emailed and the account is flagged so the late "
                     "fee is not charged again.",
             "kind": "authority"},
            {"source_id": "case-log#fee-flag",
             "text": "Our case log shows the corrected statement went out and the fee was never "
                     "re-applied to the account.",
             "kind": "document"},
        ],
        "shareable": False, "evidence_status": "reference_only",
    },
    # -- refund_request --------------------------------------------------
    {
        "entry_id": "seed-refund-1",
        "domain": "refund_request", "task": "support_reply",
        "rubric_id": "refund_request.support_reply.v1",
        "request": "I want a refund for order 88213 because the item arrived damaged.",
        "response": "I'm sorry it arrived damaged. Under the returns policy, damaged goods qualify for a "
                    "full refund when reported within 30 days with a photo, which your report satisfies. "
                    "I've approved a full refund to your original payment method within 5-10 business "
                    "days. If it hasn't landed by then, we'll escalate to the payments team the same day.",
        "criteria_scores": {"confirms_order": 1.0, "states_eligibility": 1.0,
                            "states_remedy": 1.0, "gives_timeline": 1.0,
                            "no_hidden_conditions": 1.0, "offers_escalation_path": 0.9},
        "evidence": ["order_receipt: 88213",
                     "policy_citation: returns-policy#damaged"],
        "sources": [
            {"source_id": "returns-policy#damaged",
             "text": "The returns policy refunds damaged goods in full when a customer reports "
                     "within 30 days and attaches a photo.",
             "kind": "authority"},
            {"source_id": "damage-report-order-88213",
             "text": "The damage report for order 88213 arrived within the 30 days with a "
                     "photo, so the goods qualify for a full refund.",
             "kind": "document"},
            {"source_id": "returns-policy#refund-timing",
             "text": "An approved refund reaches the original payment method in 5-10 business "
                     "days.",
             "kind": "authority"},
            {"source_id": "case-log#refund-approved",
             "text": "Our case log shows the full refund cleared to the customer's original "
                     "payment method inside 5-10 business days.",
             "kind": "document"},
            {"source_id": "returns-policy#payments-escalation",
             "text": "Where the money has not arrived by then, the payments team is escalated "
                     "to on the same day.",
             "kind": "authority"},
            {"source_id": "case-log#payments-escalation",
             "text": "The case log puts the payments team onto it the same day if the refund "
                     "has not landed.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
    {
        "entry_id": "seed-refund-2",
        "domain": "refund_request", "task": "support_reply",
        "rubric_id": "refund_request.support_reply.v1",
        "request": "I cancelled within the trial window but I was still charged. Please refund me.",
        "response": "Your account shows the cancellation was recorded on day 9 of the 14-day trial, so "
                    "the charge should not have applied. I've refunded the full amount under reference "
                    "REF-2098; it should post within 5 business days. I've also prevented a repeat "
                    "charge on renewal and can send written confirmation for your records today.",
        "criteria_scores": {"confirms_order": 1.0, "states_eligibility": 0.95,
                            "states_remedy": 1.0, "gives_timeline": 1.0,
                            "no_hidden_conditions": 0.95, "offers_escalation_path": 0.8},
        "evidence": ["order_receipt: trial-refund-2098",
                     "policy_citation: trials-policy#cancellation"],
        "sources": [
            {"source_id": "trials-policy#cancellation-window",
             "text": "A trial cancellation logged on day 9 out of the 14 days means no charge "
                     "was due.",
             "kind": "authority"},
            {"source_id": "account#trial-cancellation-record",
             "text": "The account logged the cancellation on day 9 of the 14-day trial; the "
                     "charge was not due.",
             "kind": "document"},
            {"source_id": "refunds-policy#reference-2098",
             "text": "Reference REF-2098 covers the refund of the full amount.",
             "kind": "authority"},
            {"source_id": "case-log-REF-2098",
             "text": "Case REF-2098 records the refund of the full amount.",
             "kind": "document"},
            {"source_id": "refunds-policy#posting-window",
             "text": "A refund posts within 5 business days.",
             "kind": "authority"},
            {"source_id": "case-log#posting-window",
             "text": "The case log expects the refund to post within 5 business days.",
             "kind": "document"},
            {"source_id": "billing-policy#renewal-charge",
             "text": "A repeat charge on renewal is prevented, and written confirmation goes to "
                     "the customer today.",
             "kind": "authority"},
            {"source_id": "case-log#renewal-confirmation",
             "text": "Our case log blocked the repeat charge on renewal and posted written "
                     "confirmation today.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
    # -- subscription_cancellation --------------------------------------
    {
        "entry_id": "seed-cancel-1",
        "domain": "subscription_cancellation", "task": "support_reply",
        "rubric_id": "subscription_cancellation.support_reply.v1",
        "request": "Please cancel my subscription and tell me what happens to my last payment.",
        "response": "I've cancelled the subscription effective immediately, so no further charges will "
                    "be taken. Your last payment covered the period to the 30th, so a prorated refund "
                    "for the unused days is due and I've issued it under reference CX-7781. The refund "
                    "returns to your card within 5-7 business days and you'll get an email confirmation "
                    "within the hour.",
        "criteria_scores": {"confirms_identity": 0.9, "confirms_cancellation": 1.0,
                            "states_last_billing_date": 1.0, "states_prorated_refund": 1.0,
                            "gives_confirmation_reference": 1.0, "no_retention_trap": 1.0},
        "evidence": ["billing_statement: last-payment",
                     "policy_citation: cancellation-policy#prorated"],
        "sources": [
            {"source_id": "cancellation-policy#immediate",
             "text": "Closing a subscription straight away means no further charge is taken.",
             "kind": "authority"},
            {"source_id": "case-log-CX-7781#cancellation",
             "text": "The case log closed the subscription on the spot, so no further charge is "
                     "taken.",
             "kind": "document"},
            {"source_id": "cancellation-policy#prorated",
             "text": "Money already taken covered the period up to the 30th, so the unused days "
                     "come back as a prorated refund under CX-7781.",
             "kind": "authority"},
            {"source_id": "billing-statement#last-payment",
             "text": "The statement shows the last payment reaching the 30th, leaving the unused "
                     "days as a prorated refund.",
             "kind": "document"},
            {"source_id": "refunds-policy#card-timing",
             "text": "The refund goes back to the card within 5-7 business days, with the email "
                     "inside the hour.",
             "kind": "authority"},
            {"source_id": "case-log#refund-timing",
             "text": "The confirmation email left within the hour and the money reaches the card "
                     "in 5-7 business days.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
    {
        "entry_id": "seed-cancel-2",
        "domain": "subscription_cancellation", "task": "support_reply",
        "rubric_id": "subscription_cancellation.support_reply.v1",
        "request": "Cancel my plan. I don't want to be asked to stay again.",
        "response": "Understood. The plan is cancelled as of now and no retention offer will be "
                    "presented. Your final billing date is the 30th, recorded on the account under "
                    "reference CX-7790. Because the next period has not been charged, there is nothing "
                    "to refund and no further payments will be collected; a written confirmation is on "
                    "its way.",
        "criteria_scores": {"confirms_identity": 0.9, "confirms_cancellation": 1.0,
                            "states_last_billing_date": 1.0, "states_prorated_refund": 0.9,
                            "gives_confirmation_reference": 1.0, "no_retention_trap": 1.0},
        "evidence": ["billing_statement: final-period"],
        "sources": [
            {"source_id": "billing-policy#final-billing-date",
             "text": "The account's final billing date reads the 30th, under CX-7790.",
             "kind": "authority"},
            {"source_id": "account#final-billing-date",
             "text": "CX-7790 on the account marks the final billing date of the 30th.",
             "kind": "document"},
            {"source_id": "cancellation-policy#nothing-to-refund",
             "text": "Nothing is refundable where the following period was never charged, and no "
                     "further payments are collected.",
             "kind": "authority"},
            {"source_id": "case-log#no-balance",
             "text": "Our case log shows the following period was never charged, so nothing is "
                     "refunded and no further payment is collected.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
    {
        "entry_id": "seed-cancel-3",
        "domain": "subscription_cancellation", "task": "support_reply",
        "rubric_id": "subscription_cancellation.support_reply.v1",
        "request": "Cancel both of my own plans and refund the one I was double-charged on.",
        "response": "Both plans on your account are cancelled and I can see the duplicate charge on the "
                    "second. I've refunded that duplicate in full under reference CX-7805; the other "
                    "plan has no balance outstanding. You'll get one confirmation email covering both "
                    "changes within the hour and the refund should post within 5-7 business days.",
        "criteria_scores": {"confirms_identity": 0.8, "confirms_cancellation": 1.0,
                            "states_last_billing_date": 0.8, "states_prorated_refund": 1.0,
                            "gives_confirmation_reference": 1.0, "no_retention_trap": 0.9},
        "evidence": ["billing_statement: dual-plan-duplicate"],
        "sources": [
            {"source_id": "account#dual-plan-cancellation",
             "text": "Two plans sit on this account, both now closed, and the second carries a "
                     "duplicate charge.",
             "kind": "authority"},
            {"source_id": "case-log#dual-cancellation",
             "text": "Our case log closed the account's two plans and notes the duplicate charge "
                     "sitting on the second.",
             "kind": "document"},
            {"source_id": "refunds-policy#duplicate-refund",
             "text": "CX-7805 returns the duplicate charge in full.",
             "kind": "authority"},
            {"source_id": "case-log-CX-7805",
             "text": "The duplicate was refunded in full under CX-7805.",
             "kind": "document"},
            {"source_id": "notifications-policy#confirmation",
             "text": "A single confirmation email covers both changes inside the hour, and the "
                     "money posts in 5-7 business days.",
             "kind": "authority"},
            {"source_id": "case-log#confirmation-email",
             "text": "Our case log sent one confirmation email for both changes within the hour, "
                     "with the refund posting in 5-7 business days.",
             "kind": "document"},
        ],
        "shareable": False, "evidence_status": "reference_only",
    },
    # -- hidden_fees -----------------------------------------------------
    {
        "entry_id": "seed-fees-1",
        "domain": "hidden_fees", "task": "support_reply",
        "rubric_id": "hidden_fees.support_reply.v1",
        "request": "There's a fee on my statement I never agreed to. What is it and can it be removed?",
        "response": "The fee is a monthly maintenance charge set out in section 2.1 of your agreement, "
                    "and I've itemized exactly where it appears on your statement. Because it was not "
                    "highlighted at signup, I've removed it and refunded the amounts taken so far, "
                    "effective from this cycle. The credit shows within 5 business days; if it recurs, "
                    "escalate to our fees team within 14 days.",
        "criteria_scores": {"itemizes_fee": 1.0, "explains_contract_basis": 1.0,
                            "states_removal_remedy": 1.0, "gives_timeline": 1.0,
                            "offers_escalation_path": 0.9},
        "evidence": ["fee_schedule: section-2.1",
                     "contract_clause: agreement-2.1"],
        "sources": [
            {"source_id": "agreement#section-2-1",
             "text": "A monthly maintenance charge sits in section 2.1 of the agreement and is "
                     "itemized on the statement.",
             "kind": "authority"},
            {"source_id": "fee-schedule#maintenance",
             "text": "The schedule lists the monthly maintenance charge under clause 2.1 of the "
                     "agreement, and the statement itemizes it.",
             "kind": "document"},
            {"source_id": "fees-policy#credit-timing",
             "text": "A credit appears within 5 business days.",
             "kind": "authority"},
            {"source_id": "case-log#credit-posted",
             "text": "Our case log puts the credit through in 5 business days.",
             "kind": "document"},
            {"source_id": "fees-policy#recurrence",
             "text": "A fee that comes back goes to the fees team within 14 days.",
             "kind": "authority"},
            {"source_id": "case-log#fees-escalation",
             "text": "Our case log escalates to the fees team in 14 days whenever the fee comes "
                     "back.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
    {
        "entry_id": "seed-fees-2",
        "domain": "hidden_fees", "task": "support_reply",
        "rubric_id": "hidden_fees.support_reply.v1",
        "request": "Why was I charged an early-termination fee when I moved house?",
        "response": "The early-termination fee applies under clause 7.3 only when a contract ends "
                    "outside the permitted moving exception. Your move qualifies for that exception, so "
                    "the fee does not apply and I've reversed it under reference FEE-3310. The credit "
                    "shows within 5-7 business days and I've flagged the account so the fee is not "
                    "re-applied.",
        "criteria_scores": {"itemizes_fee": 0.95, "explains_contract_basis": 1.0,
                            "states_removal_remedy": 1.0, "gives_timeline": 1.0,
                            "offers_escalation_path": 0.8},
        "evidence": ["contract_clause: agreement-7.3",
                     "fee_schedule: early-termination"],
        "sources": [
            {"source_id": "agreement#clause-7-3",
             "text": "Under clause 7.3 the early-termination fee applies only when a contract "
                     "ends outside the exempt circumstances.",
             "kind": "authority"},
            {"source_id": "contract#clause-7-3",
             "text": "A contract ending outside the exempt circumstances is the only case where "
                     "clause 7.3 charges the early-termination fee.",
             "kind": "document"},
            {"source_id": "fees-policy#moving-exception",
             "text": "The customer's circumstances qualify for the exception, so the fee does "
                     "not apply; FEE-3310 reverses it.",
             "kind": "authority"},
            {"source_id": "case-log-FEE-3310",
             "text": "Our case log applied the exception, so the fee does not apply and FEE-3310 "
                     "reverses the charge.",
             "kind": "document"},
            {"source_id": "refunds-policy#fee-credit",
             "text": "A credit lands in 5-7 business days and the account is flagged so the fee "
                     "is not charged again.",
             "kind": "authority"},
            {"source_id": "case-log#fee-credit-flag",
             "text": "Our case log shows the credit inside 5-7 business days and the account "
                     "flagged so the fee is not charged again.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
    # -- credit_report_error ---------------------------------------------
    {
        "entry_id": "seed-credit-1",
        "domain": "credit_report_error", "task": "dispute_letter",
        "rubric_id": "credit_report_error.dispute_letter.v1",
        "request": "There's a collection account on my credit report that isn't mine. How do I get it removed?",
        "response": "You can dispute this under the fair credit reporting rules, and the bureau must "
                    "investigate within 30 days of receiving your dispute. I've drafted the dispute "
                    "identifying the account and the specific item in error, citing the applicable "
                    "provision, and requesting correction or deletion. Keep copies of the letter, the "
                    "report excerpt, and the delivery proof, since you may need them if the item is not "
                    "corrected within the deadline.",
        "criteria_scores": {"identifies_account": 0.95, "identifies_disputed_item": 1.0,
                            "cites_legal_basis": 1.0, "requests_correction": 1.0,
                            "gives_response_deadline": 1.0, "preserves_evidence": 1.0},
        "evidence": ["law_citation: fair-reporting-act#611",
                     "report_excerpt: disputed-account"],
        "sources": [
            {"source_id": "fair-reporting-rules#dispute",
             "text": "A dispute filed under the fair credit reporting rules obliges the bureau "
                     "to investigate inside 30 days.",
             "kind": "authority"},
            {"source_id": "fair-reporting-act#611",
             "text": "The bureau investigates a credit reporting dispute within 30 days of "
                     "receipt.",
             "kind": "citation",
             "url": "law://fair-reporting-act/611"},
            {"source_id": "disputes-policy#records",
             "text": "Hold on to the letter, the report excerpt and the delivery proof in case "
                     "the item is not corrected before the deadline.",
             "kind": "authority"},
            {"source_id": "case-file#dispute-records",
             "text": "Our case file keeps the letter, the report excerpt and the delivery "
                     "proof, in case the item is never corrected.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
    {
        "entry_id": "seed-credit-2",
        "domain": "credit_report_error", "task": "dispute_letter",
        "rubric_id": "credit_report_error.dispute_letter.v1",
        "request": "My own report shows a late payment that I have proof was paid on time.",
        "response": "Your payment record shows the amount cleared two days before the reported due "
                    "date, so the late mark is unsupported. I've prepared a dispute that identifies the "
                    "account, attaches your payment confirmation, and cites the rule requiring accurate "
                    "information. The bureau has 30 days to respond and must correct or delete an item "
                    "it cannot verify; keep the submitted copy and the delivery receipt.",
        "criteria_scores": {"identifies_account": 0.9, "identifies_disputed_item": 1.0,
                            "cites_legal_basis": 1.0, "requests_correction": 1.0,
                            "gives_response_deadline": 1.0, "preserves_evidence": 1.0},
        "evidence": ["law_citation: fair-reporting-act#611(a)",
                     "payment_proof: cleared-early"],
        "sources": [
            {"source_id": "payment-record#cleared-early",
             "text": "The amount cleared ahead of the due date on the payment record, so the "
                     "late mark is unsupported.",
             "kind": "authority"},
            {"source_id": "account#payment-confirmation",
             "text": "The confirmation shows the money arriving before the reported due date, "
                     "so the late mark cannot stand.",
             "kind": "document"},
            {"source_id": "disputes-policy#accurate-information",
             "text": "A dispute names the account and cites the rule on accurate information, "
                     "with the payment confirmation attached.",
             "kind": "authority"},
            {"source_id": "case-file#dispute-prepared",
             "text": "Our case file holds the prepared dispute that identifies the account and "
                     "attaches the payment confirmation.",
             "kind": "document"},
            {"source_id": "fair-reporting-rules#response-deadline",
             "text": "The bureau answers inside 30 days and must delete or correct anything it "
                     "cannot verify.",
             "kind": "authority"},
            {"source_id": "fair-reporting-act#611-a",
             "text": "An item the bureau cannot verify is corrected or deleted inside its 30 "
                     "days.",
             "kind": "citation"},
        ],
        "shareable": False, "evidence_status": "reference_only",
    },
    # -- debt_collection -------------------------------------------------
    {
        "entry_id": "seed-debt-1",
        "domain": "debt_collection", "task": "response_letter",
        "rubric_id": "debt_collection.response_letter.v1",
        "request": "A debt collector keeps calling me about a debt I don't recognise. What do I do?",
        "response": "Request validation of the debt in writing before paying or acknowledging anything: "
                    "the collector must provide the amount, the creditor, and the basis for the claim. "
                    "I've drafted a letter demanding that validation, asserting your rights under the "
                    "collection rules, and asking them to stop contact until validation is provided. "
                    "Send it with proof of delivery and keep a copy; contact after your written request "
                    "supports a regulatory complaint.",
        "criteria_scores": {"validates_debt": 1.0, "asserts_rights": 1.0,
                            "requests_cease_contact": 1.0, "gives_deadline": 0.9,
                            "preserves_records": 1.0},
        "evidence": ["law_citation: debt-collection-rules#validation",
                     "debt_validation: request-template"],
        "sources": [
            {"source_id": "collection-rules#validation",
             "text": "Ask for validation of the debt in writing before you pay or acknowledge "
                     "anything; the collector must supply the amount, the creditor and the "
                     "basis for the claim.",
             "kind": "authority"},
            {"source_id": "case-file#validation-demand",
             "text": "Our case file holds a written demand that the collector validate the "
                     "debt, naming the amount, the creditor and the basis for the claim.",
             "kind": "document"},
            {"source_id": "collection-rules#rights",
             "text": "The collection rules give you the right to demand validation and to ask "
                     "that contact cease until it arrives.",
             "kind": "authority"},
            {"source_id": "case-file#cease-contact-letter",
             "text": "Our case file keeps a letter asserting the collection rules and asking "
                     "that contact cease until validation comes in.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
    {
        "entry_id": "seed-debt-2",
        "domain": "debt_collection", "task": "response_letter",
        "rubric_id": "debt_collection.response_letter.v1",
        "request": "Collectors are calling my workplace after I asked them to stop.",
        "response": "Once you ask in writing to stop contacting you at work, continued calls there are a "
                    "documented breach of the collection rules. I've drafted a cease-contact letter "
                    "citing the rule, stating the dates of your earlier request, and asking for "
                    "validation of the debt. I've logged the call dates; keep that log and the delivery "
                    "proof, because they support escalation to the regulator if the calls continue.",
        "criteria_scores": {"validates_debt": 0.9, "asserts_rights": 1.0,
                            "requests_cease_contact": 1.0, "gives_deadline": 0.9,
                            "preserves_records": 1.0},
        "evidence": ["law_citation: debt-collection-rules#contact",
                     "debt_validation: pending"],
        "sources": [
            {"source_id": "collection-rules#workplace-contact",
             "text": "Workplace calls after a written request to stop contact there are a "
                     "documented breach of the collection rules.",
             "kind": "authority"},
            {"source_id": "call-log#workplace-calls",
             "text": "Our call log shows continued calls to the work number after the written "
                     "request, a documented breach of the collection rules.",
             "kind": "document"},
            {"source_id": "collection-rules#cease-contact",
             "text": "A cease-contact letter cites the rule, sets out the dates of the earlier "
                     "request and asks for validation of the debt.",
             "kind": "authority"},
            {"source_id": "case-file#cease-contact-request",
             "text": "Our case file holds the cease-contact letter with the earlier request "
                     "dates and the demand for validation of the debt.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
    {
        "entry_id": "seed-debt-3",
        "domain": "debt_collection", "task": "response_letter",
        "rubric_id": "debt_collection.response_letter.v1",
        "request": "Is this debt even mine? They won't tell me where it came from.",
        "response": "They are required to validate the debt on request, including the creditor and the "
                    "basis for the amount, and they may not continue collection activity until they do. "
                    "I've drafted the validation demand and a request to pause contact in the meantime. "
                    "Send it with proof of delivery and keep a copy; if they fail to validate within 30 "
                    "days, that failure is itself the basis for a complaint.",
        "criteria_scores": {"validates_debt": 1.0, "asserts_rights": 1.0,
                            "requests_cease_contact": 0.95, "gives_deadline": 1.0,
                            "preserves_records": 0.9},
        "evidence": ["law_citation: debt-collection-rules#validation",
                     "debt_validation: request-template"],
        "sources": [
            {"source_id": "collection-rules#validation-required",
             "text": "A written request obliges them to validate the debt — creditor and basis "
                     "for the amount included — and collection cannot continue until they do.",
             "kind": "authority"},
            {"source_id": "case-file#validation-request",
             "text": "Our case file records the demand to validate the debt, covering the "
                     "creditor and the basis for the amount; collection cannot continue until "
                     "they do.",
             "kind": "document"},
            {"source_id": "complaints-policy#validation-failure",
             "text": "A failure to validate inside 30 days is itself grounds for a complaint.",
             "kind": "authority"},
            {"source_id": "case-file#complaint-basis",
             "text": "Our case file treats a validation failure at 30 days as the basis for a "
                     "complaint.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
    {
        "entry_id": "seed-debt-4",
        "domain": "debt_collection", "task": "response_letter",
        "rubric_id": "debt_collection.response_letter.v1",
        "request": "The collector added interest I never agreed to. Can I dispute it?",
        "response": "Yes. The collector must show the contractual or legal basis for each amount "
                    "claimed. I've drafted a letter requesting itemization of principal, interest, and "
                    "fees, citing the validation requirement, and asking them to suspend collection "
                    "until they answer. Keep the letter and delivery proof; an unexplained charge is a "
                    "strong basis for a regulator complaint if it is not corrected.",
        "criteria_scores": {"validates_debt": 1.0, "asserts_rights": 0.95,
                            "requests_cease_contact": 0.9, "gives_deadline": 0.9,
                            "preserves_records": 1.0},
        "evidence": ["law_citation: debt-collection-rules#validation",
                     "contract_clause: itemization-request"],
        "sources": [
            {"source_id": "collection-rules#basis-for-amounts",
             "text": "Every amount claimed needs its contractual or legal basis shown by the "
                     "collector.",
             "kind": "authority"},
            {"source_id": "contract#amounts-claimed",
             "text": "The contract sets out the contractual basis behind each amount the "
                     "collector claims.",
             "kind": "document"},
            {"source_id": "collection-rules#itemization",
             "text": "A letter asking for itemization of principal, interest and fees cites the "
                     "validation requirement and asks for collection to be suspended until they "
                     "answer.",
             "kind": "authority"},
            {"source_id": "case-file#itemization-letter",
             "text": "Our case file holds the itemization letter for principal, interest and "
                     "fees, citing validation, with collection suspended until they answer.",
             "kind": "document"},
            {"source_id": "complaints-policy#unexplained-charge",
             "text": "A charge nobody can explain is a strong basis for a regulator complaint "
                     "while it is not corrected.",
             "kind": "authority"},
            {"source_id": "case-file#regulator-complaint",
             "text": "Our case file notes the charge stays a strong basis for a regulator "
                     "complaint while it is not corrected.",
             "kind": "document"},
        ],
        "shareable": True, "evidence_status": "reference_only",
    },
]

ANONYMOUS_SEED: List[Dict[str, Any]] = [e for e in GRADED_RESPONSES if e["shareable"]]
PRIVATE_SEED: List[Dict[str, Any]] = [e for e in GRADED_RESPONSES if not e["shareable"]]

DOMAINS: List[str] = sorted(DEFAULT_RUBRIC_BY_DOMAIN)


def rubric(rubric_id: str) -> Rubric:
    return RUBRICS[rubric_id]


def rubric_for_domain(domain: str) -> Rubric:
    rubric_id = DEFAULT_RUBRIC_BY_DOMAIN.get(domain)
    if rubric_id is None:
        raise KeyError(f"no seeded rubric for domain {domain!r}")
    return RUBRICS[rubric_id]


def qualifying_questions(domain: str) -> List[str]:
    """The always-ask questions for a domain (generic first, then specific)."""
    return list(GENERIC_QUALIFYING_QUESTIONS) + list(
        QUALIFYING_QUESTIONS_BY_DOMAIN.get(domain, []))


def sources_for(record: Mapping[str, Any]) -> List[Source]:
    """The independent witnesses behind one record's factual claims.

    Empty when a record ships none — the honest signal, and the one the
    corroboration gate reads as "unverified", never as "approved".
    """
    sources: List[Source] = []
    for item in record.get("sources") or []:
        sources.append(Source(
            source_id=str(item.get("source_id", "")),
            text=str(item.get("text", "")),
            kind=str(item.get("kind", "document")),
            trust=item.get("trust"),
            independent=bool(item.get("independent", True)),
            url=str(item.get("url", "")),
            derived_from=str(item.get("derived_from", "")),
        ))
    return sources


def seed_records(scope: str = "anonymous") -> Sequence[Dict[str, Any]]:
    if scope == "anonymous":
        return ANONYMOUS_SEED
    if scope == "private":
        return PRIVATE_SEED
    return GRADED_RESPONSES


__all__ = [
    "SEED_SCHEMA", "RUBRICS", "DEFAULT_RUBRIC_BY_DOMAIN", "DOMAINS",
    "GENERIC_QUALIFYING_QUESTIONS", "QUALIFYING_QUESTIONS_BY_DOMAIN",
    "GRADED_RESPONSES", "ANONYMOUS_SEED", "PRIVATE_SEED",
    "rubric", "rubric_for_domain", "qualifying_questions", "seed_records",
    "sources_for",
]
