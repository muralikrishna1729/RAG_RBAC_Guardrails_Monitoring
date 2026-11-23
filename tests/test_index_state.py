import tempfile
import unittest
from pathlib import Path

from app.utils.index_state import (
    DEFAULT_COLLECTION_NAME,
    get_active_collection_name,
    get_index_generation,
    write_index_generation,
)


class IndexStateTests(unittest.TestCase):
    def test_publishes_generation_and_collection_together(self):
        with tempfile.TemporaryDirectory() as directory:
            generation = write_index_generation(
                directory,
                collection_name="rag_test_collection",
                generation="generation-123",
            )
            self.assertEqual(generation, "generation-123")
            self.assertEqual(get_index_generation(directory), "generation-123")
            self.assertEqual(get_active_collection_name(directory), "rag_test_collection")

    def test_reads_legacy_plain_text_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / ".index_generation"
            marker.write_text("legacy-generation", encoding="utf-8")
            self.assertEqual(get_index_generation(directory), "legacy-generation")
            self.assertEqual(get_active_collection_name(directory), DEFAULT_COLLECTION_NAME)

    def test_missing_marker_uses_legacy_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(get_index_generation(directory), "legacy")
            self.assertEqual(get_active_collection_name(directory), DEFAULT_COLLECTION_NAME)


if __name__ == "__main__":
    unittest.main()
