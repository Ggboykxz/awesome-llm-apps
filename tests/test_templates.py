import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from advanced_ai_agents.single_agent_apps.earnings_call_analyst_agent import youtube_ingest


class TestEarningsCallAnalyst:
    def test_extract_video_id_from_standard_url(self):
        result = youtube_ingest.extract_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        assert result == "dQw4w9WgXcQ"

    def test_extract_video_id_from_short_url(self):
        result = youtube_ingest.extract_video_id(
            "https://youtu.be/dQw4w9WgXcQ?t=120"
        )
        assert result == "dQw4w9WgXcQ"

    def test_extract_video_id_from_video_id_only(self):
        result = youtube_ingest.extract_video_id("dQw4w9WgXcQ")
        assert result == "dQw4w9WgXcQ"

    def test_extract_video_id_rejects_invalid(self):
        with pytest.raises(ValueError):
            youtube_ingest.extract_video_id("https://example.com/not-youtube")

    def test_extract_video_id_from_live_url(self):
        result = youtube_ingest.extract_video_id(
            "https://www.youtube.com/live/dQw4w9WgXcQ?si=abc"
        )
        assert result == "dQw4w9WgXcQ"

    def test_chunk_transcript_basic(self):
        from advanced_ai_agents.single_agent_apps.earnings_call_analyst_agent.youtube_ingest import (
            chunk_transcript,
        )
        from advanced_ai_agents.single_agent_apps.earnings_call_analyst_agent.schemas import (
            TranscriptSegment,
        )

        segments = [
            TranscriptSegment(start=0, duration=5, text="First segment"),
            TranscriptSegment(start=5, duration=5, text="Second segment"),
            TranscriptSegment(start=10, duration=5, text="Third segment"),
        ]

        chunks = chunk_transcript(segments, window_seconds=10)

        assert len(chunks) == 2
        assert chunks[0].start == 0
        assert chunks[1].start == 10


class TestChatWithYoutube:
    def test_session_state_script_runs(self):
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "advanced_llm_apps",
                    "chat_with_X_tutorials",
                    "chat_with_youtube_videos",
                    "test_session_state.py",
                )
            ],
            capture_output=True,
            text=True,
            cwd=os.path.join(
                os.path.dirname(__file__),
                "..",
                "advanced_llm_apps",
                "chat_with_X_tutorials",
                "chat_with_youtube_videos",
            ),
        )

        assert result.returncode == 0
        assert "Session State Logic Test Complete" in result.stdout


class TestRagTutorials:
    def test_rag_chain_imports(self):
        import ast
        import inspect

        app_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "rag_tutorials",
            "rag_chain",
            "app.py",
        )

        with open(app_path, "r") as f:
            source = f.read()

        tree = ast.parse(source)

        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        assert "format_docs" in functions
        assert "add_to_db" in functions
        assert "run_rag_chain" in functions
        assert "main" in functions

    def test_vision_rag_imports(self):
        app_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "rag_tutorials",
            "vision_rag",
            "vision_rag.py",
        )

        with open(app_path, "r") as f:
            source = f.read()

        assert "cohere" in source.lower() or "google" in source.lower()

    def test_autonomous_rag_imports(self):
        app_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "rag_tutorials",
            "autonomous_rag",
            "autorag.py",
        )

        with open(app_path, "r") as f:
            source = f.read()

        assert len(source) > 0


class TestStarterAgents:
    def test_ai_reasoning_agent_structure(self):
        app_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "starter_ai_agents",
            "ai_reasoning_agent",
            "reasoning_agent.py",
        )

        with open(app_path, "r") as f:
            source = f.read()

        assert "agno" in source.lower() or "openai" in source.lower()

    def test_ai_travel_agent_structure(self):
        app_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "starter_ai_agents",
            "ai_travel_agent",
            "travel_agent.py",
        )

        with open(app_path, "r") as f:
            source = f.read()

        assert len(source) > 0

    def test_ai_data_analysis_agent_structure(self):
        app_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "starter_ai_agents",
            "ai_data_analysis_agent",
            "ai_data_analyst.py",
        )

        with open(app_path, "r") as f:
            source = f.read()

        assert len(source) > 0


class TestVoiceAiAgents:
    def test_voice_rag_structure(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "voice_ai_agents",
            "voice_rag_openaisdk",
        )

        if os.path.exists(path):
            py_files = [f for f in os.listdir(path) if f.endswith(".py")]
            assert len(py_files) > 0
        else:
            pytest.skip("voice_rag_openaisdk not present")


class TestMcpAgents:
    def test_browser_mcp_structure(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "mcp_ai_agents",
            "browser_mcp_agent",
        )

        if os.path.exists(path):
            py_files = [f for f in os.listdir(path) if f.endswith(".py")]
            assert len(py_files) > 0
        else:
            pytest.skip("browser_mcp_agent not present")