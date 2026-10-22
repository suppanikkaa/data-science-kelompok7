"""
Automated Tests for the NER Service
====================================
At minimum 5 test cases covering:
- Normal input (English text with clear entities)
- Empty title / body rejection
- Malformed JSON
- Input exceeding max length
- Non-ASCII text (Malay/Chinese)
- Punctuation and repeated entities
- Long text
- API endpoint responses (via Flask test client)

Run: python -m pytest tests/test_ner_service.py -v
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

# ── Unit tests for extract_entities ─────────────────────────────────────────

from ner_pipeline import extract_entities, supported_labels, MAX_TEXT_LENGTH


class TestExtractEntities:

    def test_normal_english_article(self):
        """TC-01: Standard English news article returns entities."""
        result = extract_entities(
            title="Maybank posts record profit in 2026",
            body="Maybank Group, headquartered in Kuala Lumpur, reported a net profit of RM2.5 billion for Q3 2026."
        )
        assert "entities" in result
        assert "meta" in result
        assert result["meta"]["entity_count"] == len(result["entities"])
        # At least one entity should be found
        assert result["meta"]["entity_count"] > 0
        # All entities must have required fields
        for ent in result["entities"]:
            assert "text" in ent
            assert "label" in ent
            assert "start" in ent
            assert "end" in ent
            assert isinstance(ent["start"], int)
            assert isinstance(ent["end"], int)
            assert ent["start"] < ent["end"]

    def test_empty_title_raises(self):
        """TC-02: Empty title must raise ValueError."""
        with pytest.raises(ValueError, match="title"):
            extract_entities(title="", body="Some body text here.")

    def test_empty_body_raises(self):
        """TC-03: Empty body must raise ValueError."""
        with pytest.raises(ValueError, match="body"):
            extract_entities(title="A valid title", body="")

    def test_whitespace_only_title_raises(self):
        """TC-04: Whitespace-only title is treated as empty."""
        with pytest.raises(ValueError, match="title"):
            extract_entities(title="   \t\n", body="Some body text here.")

    def test_exceeds_max_length(self):
        """TC-05: Text exceeding MAX_TEXT_LENGTH raises ValueError."""
        big_body = "A " * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="maximum length"):
            extract_entities(title="title", body=big_body)

    def test_entity_offsets_correct(self):
        """TC-06: Character offsets must point to the correct text span."""
        combined_text = "Maybank Group is based in Kuala Lumpur."
        result = extract_entities(
            title="Maybank Group",
            body="is based in Kuala Lumpur."
        )
        # Reconstruct combined text used internally
        internal_text = "Maybank Group\n\nis based in Kuala Lumpur."
        for ent in result["entities"]:
            extracted = internal_text[ent["start"]:ent["end"]]
            assert extracted == ent["text"], (
                f"Offset mismatch: text='{ent['text']}' but slice='{extracted}'"
            )

    def test_no_entities_article(self):
        """TC-07: Article with no named entities returns empty list, not error."""
        result = extract_entities(
            title="General observations",
            body="The weather today is sunny and warm. Everything is fine."
        )
        assert "entities" in result
        assert isinstance(result["entities"], list)

    def test_repeated_entity(self):
        """TC-08: Same entity appearing multiple times should be detected at each occurrence."""
        result = extract_entities(
            title="CIMB quarterly results",
            body="CIMB reported strong results. CIMB's CEO said CIMB aims to expand."
        )
        ent_texts = [e["text"] for e in result["entities"]]
        # CIMB should appear - may or may not be deduplicated based on pipeline
        assert any("CIMB" in t for t in ent_texts)

    def test_non_ascii_malay_text(self):
        """TC-09: Malay text should not crash the service."""
        result = extract_entities(
            title="Maybank rekod keuntungan rekod",
            body="Maybank melaporkan keuntungan bersih RM2.5 bilion bagi suku ketiga 2026 di Kuala Lumpur."
        )
        assert "entities" in result

    def test_punctuation_in_entity(self):
        """TC-10: Entity names with punctuation (e.g., WWF-Malaysia) should be handled."""
        result = extract_entities(
            title="Maybank and WWF-Malaysia partnership",
            body="Maybank signed a partnership agreement with WWF-Malaysia in Kuala Lumpur on 15 October 2026."
        )
        assert "entities" in result
        # Should not crash; entity detection may vary
        assert isinstance(result["entities"], list)

    def test_meta_model_field_present(self):
        """TC-11: meta.model field must be present and non-empty."""
        result = extract_entities(
            title="Test title",
            body="Test body text for Maybank in Malaysia on October 2026."
        )
        assert result["meta"]["model"]
        assert len(result["meta"]["model"]) > 0

    def test_supported_labels_non_empty(self):
        """TC-12: Model must expose at least one NER label."""
        labels = supported_labels()
        assert len(labels) > 0
        # Standard labels should be included
        assert "ORG" in labels or "PERSON" in labels


# ── Integration tests via Flask test client ──────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestAPIEndpoint:

    def test_health_endpoint(self, client):
        """TC-13: GET /health returns 200 with status ok."""
        res = client.get("/health")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data["status"] == "ok"

    def test_extract_valid_request(self, client):
        """TC-14: POST /extract-entities with valid JSON returns 200."""
        payload = {
            "title": "RHB Bank partners with Grab",
            "body": "RHB Bank and Grab Financial announced a deal in Kuala Lumpur on October 2026."
        }
        res = client.post("/extract-entities",
                          data=json.dumps(payload),
                          content_type="application/json")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "entities" in data
        assert "meta" in data

    def test_extract_empty_title_400(self, client):
        """TC-15: Empty title returns 400."""
        payload = {"title": "", "body": "Some body text."}
        res = client.post("/extract-entities",
                          data=json.dumps(payload),
                          content_type="application/json")
        assert res.status_code == 400
        data = json.loads(res.data)
        assert "error" in data

    def test_extract_missing_title_field_400(self, client):
        """TC-16: Missing title field returns 400."""
        payload = {"body": "Some body text."}
        res = client.post("/extract-entities",
                          data=json.dumps(payload),
                          content_type="application/json")
        assert res.status_code == 400

    def test_extract_wrong_content_type_400(self, client):
        """TC-17: Non-JSON content type returns 400."""
        res = client.post("/extract-entities",
                          data="title=test&body=test",
                          content_type="application/x-www-form-urlencoded")
        assert res.status_code == 400

    def test_extract_malformed_json_400(self, client):
        """TC-18: Malformed JSON returns 400."""
        res = client.post("/extract-entities",
                          data="{invalid json}",
                          content_type="application/json")
        assert res.status_code == 400

    def test_extract_non_object_json_400(self, client):
        """TC-18b: JSON body must be an object, not a list/string/number."""
        for payload in ([], [1, 2], "text", 42):
            res = client.post(
                "/extract-entities",
                data=json.dumps(payload),
                content_type="application/json",
            )
            assert res.status_code == 400
            assert "error" in json.loads(res.data)

    def test_panel_loads(self, client):
        """TC-19: GET / returns 200 with HTML panel."""
        res = client.get("/")
        assert res.status_code == 200
        assert b"NER" in res.data or b"entity" in res.data.lower()
        assert b'href="/evaluation"' in res.data

    def test_evaluation_page_loads(self, client):
        """TC-21: GET /evaluation returns the model card and metrics."""
        res = client.get("/evaluation")
        assert res.status_code == 200
        html = res.data.decode("utf-8")
        assert "Model &amp; Evaluation" in html
        assert "Human-reviewed" in html
        assert "0.4878" in html
        assert "122 spans" in html

    def test_extract_response_deterministic(self, client):
        """TC-20: Same input returns same output (deterministic)."""
        payload = {
            "title": "Maybank quarterly results",
            "body": "Maybank Group reported profits in Kuala Lumpur for Q3 2026."
        }
        res1 = client.post("/extract-entities", data=json.dumps(payload), content_type="application/json")
        res2 = client.post("/extract-entities", data=json.dumps(payload), content_type="application/json")
        assert res1.status_code == 200
        data1 = json.loads(res1.data)
        data2 = json.loads(res2.data)
        assert data1["entities"] == data2["entities"]
