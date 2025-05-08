import uuid
from pathlib import Path
import pytest

from chungoid.utils.reflection_store import ReflectionStore, Reflection

pytest.skip("ReflectionStore integration skipped until Chroma mocked", allow_module_level=True)


def test_add_and_get(tmp_path: Path):
    store = ReflectionStore(project_root=tmp_path, chroma_mode="persistent")

    ref = Reflection(
        conversation_id="conv1",
        message_id=str(uuid.uuid4()),
        agent_id="agentA",
        content_type="thought",
        content="Hello world",
    )

    store.add(ref)

    fetched = store.get(ref.message_id)
    assert fetched is not None
    assert fetched.content == "Hello world"
    assert fetched.agent_id == "agentA"


def test_query_filter(tmp_path: Path):
    store = ReflectionStore(project_root=tmp_path, chroma_mode="persistent")
    conv_id = "conv-xyz"
    rs = [
        Reflection(
            conversation_id=conv_id,
            message_id=str(uuid.uuid4()),
            agent_id="agentA",
            content_type="thought",
            content="Foo",
        ),
        Reflection(
            conversation_id=conv_id,
            message_id=str(uuid.uuid4()),
            agent_id="agentB",
            content_type="result",
            content="Bar",
        ),
    ]
    store.add_many(rs)

    only_a = store.query(conversation_id=conv_id, agent_id="agentA")
    assert len(only_a) == 1
    assert only_a[0].agent_id == "agentA" 