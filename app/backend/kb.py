"""Bookly knowledge base + grounded retrieval.

Thesis in code: the agent may not answer a policy question from parametric
memory. It must retrieve a passage, and the passage id must be attached to the
answer. If retrieval scores below threshold we return NO passages, which the
orchestrator turns into an escalation rather than a guess.

Retrieval here is TF-IDF lexical scoring over seven chunks held in this file -
deliberately the thinnest thing that satisfies the contract above. Not BM25:
there is no k1 term-frequency saturation and no b length normalisation, which
are the two things that make BM25 what it is. Being precise about that matters,
because the honest claim is the interesting one.

What this stub cannot do is what pure lexical matching never can: it misses
paraphrase ("when will my stuff turn up"), morphology ("ebook" vs "Ebooks")
and typos, and it will occasionally retrieve confidently and wrongly. In
production this is a hybrid dense+lexical index with a cross-encoder reranker,
recall@k gated in the eval set, and re-indexing when the knowledge base is
published.

None of that changes anything above this module. The interface -
`search(query) -> list[Passage]` with scores and a confidence floor - is the
same signature a real index presents, and the grounding contract is enforced by
the caller, not the retriever. Retrieval quality is the piece you buy; the
refusal to answer without it is the piece worth building.
"""

from __future__ import annotations

import math
import re
from collections import Counter

CONFIDENCE_FLOOR = 0.28  # below this we refuse to answer and escalate

CHUNKS: list[dict] = [
    {
        "id": "KB-RET-01",
        "title": "Returns window",
        "text": (
            "Bookly accepts returns on physical books within 30 days of delivery. "
            "Bookly Plus members have 60 days. Items must be unread and in "
            "resalable condition. Refunds are issued to the original payment "
            "method within 5 working days of the return being scanned by the carrier."
        ),
    },
    {
        "id": "KB-RET-02",
        "title": "Non-returnable items",
        "text": (
            "Ebooks, audiobook credits, gift cards and personalised or signed "
            "editions are non-returnable once delivered or redeemed, except where "
            "the item arrived damaged or the wrong item was sent."
        ),
    },
    {
        "id": "KB-SHIP-01",
        "title": "Delivery times and costs",
        "text": (
            "UK standard delivery is 3 to 5 working days and is free on orders "
            "over 20 pounds, otherwise 2.99 pounds. Next-day delivery is 4.99 "
            "pounds when ordered before 6pm. Orders are dispatched from the "
            "Chelmsford warehouse Monday to Friday."
        ),
    },
    {
        "id": "KB-SHIP-02",
        "title": "Missing or late parcels",
        "text": (
            "A UK parcel is only treated as lost once 10 working days have passed "
            "since dispatch. Before that window the agent should share the latest "
            "carrier scan and set expectations. After the window a replacement or "
            "refund can be arranged."
        ),
    },
    {
        "id": "KB-ACC-01",
        "title": "Password reset",
        "text": (
            "Customers reset their password from the Sign in page using Forgot "
            "password. The emailed link is valid for 30 minutes. Support can "
            "trigger the same email but can never read, set or confirm a password."
        ),
    },
    {
        "id": "KB-CAN-01",
        "title": "Cancelling an order",
        "text": (
            "Cancellation policy: Bookly can cancel an order only while it is "
            "still processing and has not shipped. Cancelling refunds the full "
            "order total to the original payment method. Once an order has "
            "shipped or been delivered, cancellation is no longer available and "
            "a delivered book follows the returns window instead."
        ),
    },
    {
        "id": "KB-PAY-01",
        "title": "Refund timing",
        "text": (
            "Once a refund is issued, card refunds appear in 3 to 5 working days "
            "and PayPal refunds within 24 hours. Bookly cannot accelerate a refund "
            "that has already been sent to the payment provider."
        ),
    },
]

_BY_ID: dict[str, dict] = {}
for _c in CHUNKS:
    if _c["id"] in _BY_ID:
        raise RuntimeError(f"duplicate knowledge-base chunk id: {_c['id']}")
    _BY_ID[_c["id"]] = _c


def get(chunk_id: str) -> dict:
    """The chunk itself, by id. Raises on an id the knowledge base does not have."""
    try:
        return _BY_ID[chunk_id]
    except KeyError:
        raise KeyError(
            f"{chunk_id} is not a knowledge-base chunk. Known ids: "
            f"{', '.join(sorted(_BY_ID))}"
        ) from None


def ref(chunk_id: str) -> str:
    """A citable id, checked against the knowledge base at import time.

    Citations exist to make an answer verifiable, so a citation that does not
    resolve is worse than none at all - it is a confident pointer to nothing.
    Every id spoken elsewhere in the codebase comes through here, which means a
    renamed or deleted chunk fails at startup rather than shipping a phantom
    reference to a customer.
    """
    return get(chunk_id)["id"]


# The ids the policy plane cites. Declared here, next to the chunks they name,
# so the passage and the reference to it cannot drift apart.
RETURNS_WINDOW = ref("KB-RET-01")
NON_RETURNABLE = ref("KB-RET-02")
CANCELLATION = ref("KB-CAN-01")

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "for", "from", "how",
    "i", "in", "is", "it", "long", "me", "my", "of", "on", "or", "the", "to",
    "what", "when", "where", "you", "your", "does", "did", "with",
}


def _tok(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


_DOC_TOKENS = {c["id"]: Counter(_tok(c["title"] + " " + c["text"])) for c in CHUNKS}
_DF = Counter()
for _tf in _DOC_TOKENS.values():
    _DF.update(set(_tf))
_N = len(CHUNKS)


def search(query: str, k: int = 2) -> list[dict]:
    """Return scored passages above CONFIDENCE_FLOOR, best first. May be empty."""
    q = _tok(query)
    if not q:
        return []
    scored = []
    for chunk in CHUNKS:
        tf = _DOC_TOKENS[chunk["id"]]
        length = sum(tf.values()) or 1
        score = 0.0
        for term in set(q):
            if term not in tf:
                continue
            idf = math.log(1 + _N / (1 + _DF[term]))
            score += (tf[term] / length) * idf * 10
        # normalise by query length so short queries cannot trivially win
        score = score / math.sqrt(len(set(q)))
        if score > 0:
            scored.append({**chunk, "score": round(score, 3)})
    scored.sort(key=lambda c: c["score"], reverse=True)
    return [c for c in scored[:k] if c["score"] >= CONFIDENCE_FLOOR]
