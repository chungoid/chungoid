from pathlib import Path

from chungoid.utils.feedback_store import FeedbackStore, ProcessFeedback

def test_feedback_add_and_query(tmp_path: Path):
    store = FeedbackStore(project_root=tmp_path, chroma_mode="persistent")

    fb = ProcessFeedback(
        conversation_id="conv123",
        agent_id="agentZ",
        stage="planning",
        sentiment="👍",
        comment="Process felt smooth and intuitive",
    )

    store.add(fb)

    # Basic query by agent
    results = store.query(agent_id="agentZ")
    assert len(results) >= 1
    assert any(r.comment.startswith("Process felt") for r in results) 