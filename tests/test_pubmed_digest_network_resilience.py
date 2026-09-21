import contextlib
import io
import sys
import types
import unittest
from unittest import mock

try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = mock.Mock
    sys.modules["openai"] = openai_stub

import pubmed_digest


class OpenWithRetryTests(unittest.TestCase):
    @mock.patch.object(pubmed_digest.time, "sleep")
    @mock.patch.object(pubmed_digest.urllib.request, "urlopen")
    def test_retries_timeout_error(self, mock_urlopen, mock_sleep):
        response = object()
        mock_urlopen.side_effect = [TimeoutError("read timed out"), response]

        result = pubmed_digest.open_with_retry(mock.Mock())

        self.assertIs(result, response)
        self.assertEqual(mock_urlopen.call_count, 2)
        mock_sleep.assert_called_once_with(1.0)


class FetchNewPapersTests(unittest.TestCase):
    @mock.patch.object(pubmed_digest.time, "sleep")
    @mock.patch.object(pubmed_digest, "paper_from_summary", return_value="pubmed-paper")
    @mock.patch.object(pubmed_digest, "fetch_biorxiv_entry_map")
    @mock.patch.object(pubmed_digest, "fetch_summaries", return_value={"123": {"title": "Paper"}})
    @mock.patch.object(
        pubmed_digest,
        "build_candidate_pmids",
        return_value=(["pubmed:123"], {"stages": []}),
    )
    def test_skips_biorxiv_hydration_without_biorxiv_candidates(
        self,
        _mock_build_candidates,
        _mock_fetch_summaries,
        mock_fetch_biorxiv,
        _mock_paper_from_summary,
        _mock_sleep,
    ):
        papers, metadata = pubmed_digest.fetch_new_papers(
            conn=mock.Mock(),
            query="taste",
            topic_label="custom",
            days_back=7,
            retmax=20,
            full_text_limit=1000,
            candidate_pool_size=50,
            sources={"pubmed", "biorxiv"},
        )

        self.assertEqual(papers, ["pubmed-paper"])
        self.assertEqual(metadata["papers_fetched"], 1)
        mock_fetch_biorxiv.assert_not_called()

    @mock.patch.object(pubmed_digest.time, "sleep")
    @mock.patch.object(pubmed_digest, "paper_from_summary", return_value="pubmed-paper")
    @mock.patch.object(
        pubmed_digest,
        "fetch_biorxiv_entry_map",
        side_effect=TimeoutError("read timed out"),
    )
    @mock.patch.object(pubmed_digest, "fetch_summaries", return_value={"123": {"title": "Paper"}})
    @mock.patch.object(
        pubmed_digest,
        "build_candidate_pmids",
        return_value=(["pubmed:123", "biorxiv:10.1101/example"], {"stages": []}),
    )
    def test_continues_when_biorxiv_hydration_times_out(
        self,
        _mock_build_candidates,
        _mock_fetch_summaries,
        mock_fetch_biorxiv,
        _mock_paper_from_summary,
        _mock_sleep,
    ):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            papers, metadata = pubmed_digest.fetch_new_papers(
                conn=mock.Mock(),
                query="taste",
                topic_label="custom",
                days_back=7,
                retmax=20,
                full_text_limit=1000,
                candidate_pool_size=50,
                sources={"pubmed", "biorxiv"},
            )

        self.assertEqual(papers, ["pubmed-paper"])
        self.assertEqual(metadata["papers_fetched"], 1)
        mock_fetch_biorxiv.assert_called_once()
        self.assertIn("continuing without bioRxiv papers", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
