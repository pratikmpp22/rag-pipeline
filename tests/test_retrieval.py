import pytest

from langchain_core.documents import Document

from src.retrieval import build_bm25_index, bm25_search, reciprocal_rank_fusion


@pytest.fixture
def sample_chunks():
    return [
        Document(page_content="Enterprise refund policy allows full refund within 30 days",
                 metadata={"source": "refund_policy.txt", "domain": "hr"}),
        Document(page_content="API rate limits for Pro plan are 10000 requests per minute",
                 metadata={"source": "api_reference.txt", "domain": "technical"}),
        Document(page_content="SSO supports SAML 2.0 with Okta and Azure AD",
                 metadata={"source": "security_compliance.txt", "domain": "technical"}),
        Document(page_content="Product integrations include Slack Jira GitHub Salesforce",
                 metadata={"source": "product_features.txt", "domain": "product"}),
        Document(page_content="Leave policy provides 20 days annual and 10 days sick leave",
                 metadata={"source": "hr_onboarding.txt", "domain": "hr"}),
    ]


def test_bm25_index_builds(sample_chunks):
    index, chunks = build_bm25_index(sample_chunks)
    assert index is not None
    assert len(chunks) == 5


def test_bm25_returns_k_results(sample_chunks):
    index, chunks = build_bm25_index(sample_chunks)
    results = bm25_search(index, chunks, "refund policy", k=3)
    assert len(results) == 3


def test_bm25_ranks_by_relevance(sample_chunks):
    index, chunks = build_bm25_index(sample_chunks)
    results = bm25_search(index, chunks, "refund policy enterprise", k=3)
    # The refund policy document should rank first
    assert "refund" in results[0].page_content.lower()


def test_rrf_merges_two_lists(sample_chunks):
    list1 = sample_chunks[:3]
    list2 = sample_chunks[2:]
    fused = reciprocal_rank_fusion([list1, list2], rrf_k=60)
    assert len(fused) == 5  # All unique docs


def test_rrf_boosts_item_in_both_lists():
    doc_a = Document(page_content="shared document about refunds",
                     metadata={"source": "a.txt"})
    doc_b = Document(page_content="unique document about billing",
                     metadata={"source": "b.txt"})
    doc_c = Document(page_content="unique document about API keys",
                     metadata={"source": "c.txt"})

    list1 = [doc_a, doc_b]
    list2 = [doc_c, doc_a]
    fused = reciprocal_rank_fusion([list1, list2], rrf_k=60)
    # doc_a appears in both lists, should rank first
    assert fused[0].page_content == doc_a.page_content


def test_multi_query_includes_original_question():
    # This tests the contract — actual LLM call is mocked in integration tests
    from src.retrieval import generate_query_variants
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Alt question 1\nAlt question 2")

    variants = generate_query_variants("What is the refund policy?", mock_llm, n=2)
    assert variants[0] == "What is the refund policy?"
    assert len(variants) >= 2
