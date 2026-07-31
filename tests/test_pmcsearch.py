import unittest
from unittest.mock import Mock

import httpx

from pmcortex.pmcsearch import PMCSearch


def api_response(payload: object) -> Mock:
    """Create a successful mocked HTTP response with a JSON payload."""
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = payload
    return response


class TestPMCSearch(unittest.TestCase):
    def test_search_returns_dataframe_with_pmcids_and_titles(self) -> None:
        search = PMCSearch()
        search.client = Mock()
        search.client.post.side_effect = [
            api_response(
                {
                    "esearchresult": {
                        "idlist": ["3438321", "2693326"],
                    }
                }
            ),
            api_response(
                {
                    "result": {
                        "uids": ["3438321", "2693326"],
                        "3438321": {"title": "Title A"},
                        "2693326": {"title": "Title B"},
                    }
                }
            ),
        ]

        result = search.search("breast cancer", max_results=2)

        self.assertEqual(list(result.columns), ["pmcid", "title"])
        self.assertEqual(
            result.to_dict("records"),
            [
                {"pmcid": "PMC3438321", "title": "Title A"},
                {"pmcid": "PMC2693326", "title": "Title B"},
            ],
        )
        search.client.post.assert_any_call(
            search.ESEARCH_URL,
            data={
                "db": "pmc",
                "term": "breast cancer",
                "retmax": "2",
                "retmode": "json",
                "sort": "relevance",
            },
        )
        search.client.post.assert_any_call(
            search.ESUMMARY_URL,
            data={
                "db": "pmc",
                "id": "3438321,2693326",
                "retmode": "json",
                "version": "2.0",
            },
        )
        search.close()

    def test_search_returns_fewer_rows_when_fewer_hits_exist(self) -> None:
        search = PMCSearch()
        search.client = Mock()
        search.client.post.side_effect = [
            api_response({"esearchresult": {"idlist": ["123"]}}),
            api_response(
                {
                    "result": {
                        "uids": ["123"],
                        "123": {"title": "Only result"},
                    }
                }
            ),
        ]

        result = search.search("rare query", max_results=100)

        self.assertEqual(
            result.to_dict("records"),
            [{"pmcid": "PMC123", "title": "Only result"}],
        )
        search.close()

    def test_search_returns_empty_dataframe_without_summary_request(self) -> None:
        search = PMCSearch()
        search.client = Mock()
        search.client.post.return_value = api_response(
            {"esearchresult": {"idlist": []}}
        )

        result = search.search("no matches", max_results=25)

        self.assertEqual(list(result.columns), ["pmcid", "title"])
        self.assertTrue(result.empty)
        search.client.post.assert_called_once()
        search.close()

    def test_search_batches_summary_requests(self) -> None:
        uids = [str(uid) for uid in range(1, 202)]
        first_batch = uids[:200]
        second_batch = uids[200:]
        search = PMCSearch()
        search.client = Mock()
        search.client.post.side_effect = [
            api_response({"esearchresult": {"idlist": uids}}),
            api_response(
                {
                    "result": {
                        "uids": first_batch,
                        **{
                            uid: {"title": f"Title {uid}"}
                            for uid in first_batch
                        },
                    }
                }
            ),
            api_response(
                {
                    "result": {
                        "uids": second_batch,
                        "201": {"title": "Title 201"},
                    }
                }
            ),
        ]

        result = search.search("large result set", max_results=201)

        self.assertEqual(len(result), 201)
        self.assertEqual(search.client.post.call_count, 3)
        self.assertEqual(result.iloc[-1].to_dict(), {
            "pmcid": "PMC201",
            "title": "Title 201",
        })
        search.close()

    def test_search_rate_limits_api_requests(self) -> None:
        now = 0.0
        sleep_calls: list[float] = []

        def fake_time() -> float:
            return now

        def fake_sleep(seconds: float) -> None:
            nonlocal now
            sleep_calls.append(seconds)
            now += seconds

        search = PMCSearch(
            max_requests_per_second=2.9,
            time_fn=fake_time,
            sleep_fn=fake_sleep,
        )
        search.client = Mock()
        search.client.post.return_value = api_response(
            {"esearchresult": {"idlist": []}}
        )

        search.search("first")
        search.search("second")

        self.assertEqual(search.client.post.call_count, 2)
        self.assertEqual(len(sleep_calls), 1)
        self.assertGreaterEqual(sleep_calls[0], 1 / 2.9)
        search.close()

    def test_search_retries_remote_protocol_error(self) -> None:
        now = 0.0
        sleep_calls: list[float] = []

        def fake_time() -> float:
            return now

        def fake_sleep(seconds: float) -> None:
            nonlocal now
            sleep_calls.append(seconds)
            now += seconds

        search = PMCSearch(
            max_attempts=2,
            retry_backoff_s=1.0,
            time_fn=fake_time,
            sleep_fn=fake_sleep,
        )
        search.client = Mock()
        search.client.post.side_effect = [
            httpx.RemoteProtocolError("Server disconnected"),
            api_response({"esearchresult": {"idlist": []}}),
        ]

        result = search.search("cancer")

        self.assertTrue(result.empty)
        self.assertEqual(search.client.post.call_count, 2)
        self.assertEqual(sleep_calls, [1.0])
        search.close()

    def test_search_does_not_retry_permanent_client_error(self) -> None:
        search = PMCSearch(max_attempts=3)
        request = httpx.Request("POST", "https://example.test/")
        response = httpx.Response(404, request=request)
        search.client = Mock()
        search.client.post.return_value = response

        with self.assertRaises(httpx.HTTPStatusError):
            search.search("missing")

        search.client.post.assert_called_once()
        search.close()

    def test_search_validates_query_and_max_results(self) -> None:
        search = PMCSearch()
        try:
            with self.assertRaises(ValueError):
                search.search(" ")
            with self.assertRaises(TypeError):
                search.search("cancer", max_results=1.5)
            with self.assertRaises(TypeError):
                search.search("cancer", max_results=True)
            with self.assertRaises(ValueError):
                search.search("cancer", max_results=0)
            with self.assertRaises(ValueError):
                search.search("cancer", max_results=10_001)
        finally:
            search.close()

    def test_constructor_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            PMCSearch(max_requests_per_second=0)
        with self.assertRaises(ValueError):
            PMCSearch(max_attempts=0)
        with self.assertRaises(ValueError):
            PMCSearch(retry_backoff_s=-1)


if __name__ == "__main__":
    unittest.main()
