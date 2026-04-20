import pytest
from httpx import AsyncClient, ASGITransport

from src.api.main import app

QUESTIONS = [
    "What is the recommended sleeping position for a newborn?",
    "At what age should solid foods be introduced?",
    "What vaccines are recommended at the 2-month visit?",
    "What are the developmental milestones for a 6-month-old?",
    "When should I call a doctor for an infant fever?",
    "How long should mothers breastfeed according to AAP?",
    "What makes a safe sleep environment for infants?",
    "What are signs of RSV in infants?",
    "When do babies typically start crawling?",
    "What foods should be avoided in the first year?",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("question", QUESTIONS)
async def test_query_returns_answer_and_sources(question: str):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/v1/query", json={"question": question})

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )

    body = response.json()
    assert body.get("answer"), f"Empty answer for: {question!r}"
    assert body.get("sources"), f"Empty sources for: {question!r}"
