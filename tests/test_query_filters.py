import unittest
from tempfile import TemporaryDirectory

import pandas as pd

from pmcortex.query_filters import QueryFilterPipeline


class TestQueryFilterPipeline(unittest.TestCase):
    def test_constructor_loads_raw_csv_files_without_eager_annotations(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)

        self.assertNotIn("hits_parsed", pipeline.annotated_df.columns)
        self.assertNotIn("n_sources", pipeline.annotated_df.columns)
        self.assertNotIn("self_citation", pipeline.annotated_df.columns)
        self.assertNotIn("author_spoiling", pipeline.annotated_df.columns)
        self.assertNotIn("explicit_spoiler", pipeline.annotated_df.columns)
        self.assertNotIn("hits_parsed", pipeline.filtered_df.columns)
        self.assertNotIn("self_citation", pipeline.filtered_df.columns)
        self.assertNotIn("explicit_spoiler", pipeline.filtered_df.columns)

    def test_expensive_annotations_are_materialized_only_when_needed(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_min_hit_queries(1)
            pipeline.filter_no_et_al_queries()

            self.assertNotIn("hits_parsed", pipeline.filtered_df.columns)
            self.assertNotIn("self_citation", pipeline.filtered_df.columns)
            self.assertNotIn("explicit_spoiler", pipeline.filtered_df.columns)
            self.assertNotIn("n_sources", pipeline.filtered_df.columns)

            pipeline.filter_no_explicit_spoilers()
            self.assertIn("explicit_spoiler", pipeline.filtered_df.columns)
            self.assertIn("author_spoiling", pipeline.filtered_df.columns)
            self.assertNotIn("self_citation", pipeline.filtered_df.columns)
            self.assertNotIn("n_sources", pipeline.filtered_df.columns)

            pipeline.filter_no_self_citations()
            self.assertIn("hits_parsed", pipeline.filtered_df.columns)
            self.assertIn("self_citation", pipeline.filtered_df.columns)
            self.assertNotIn("n_sources", pipeline.filtered_df.columns)

            pipeline.reset().filter_min_source_count(2)
            self.assertIn("n_sources", pipeline.filtered_df.columns)

        sparse_source_count = pipeline.filtered_df.loc[
            pipeline.filtered_df["query"] == "Sparse references.",
            "n_sources",
        ]
        self.assertTrue(sparse_source_count.empty)

    def test_filters_can_be_applied_stepwise(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_min_hit_queries(1)
            pipeline.filter_no_et_al_queries()
            pipeline.filter_no_explicit_spoilers()
            pipeline.filter_no_self_citations()
            pipeline.filter_min_source_count(2)

        remaining_queries = pipeline.filtered_df["query"].tolist()
        self.assertEqual(remaining_queries, ["Clean query."])

    def test_filter_min_hit_queries_accepts_n_other_than_one(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_min_hit_queries(2)

        remaining_queries = pipeline.filtered_df["query"].tolist()
        self.assertEqual(remaining_queries, ["Ambiguous citation."])

    def test_filter_min_source_count_filters_by_article_reference_count(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_min_source_count(2)

        remaining_queries = pipeline.filtered_df["query"].tolist()
        self.assertEqual(
            remaining_queries,
            [
                "Clean query.",
                "Smith et al. reported the effect.",
                "Smith reported the effect.",
                "Ambiguous citation.",
                "Independent statement.",
            ],
        )
        self.assertEqual(
            pipeline.filtered_df.loc[
                pipeline.filtered_df["query"] == "Clean query.",
                "n_sources",
            ].item(),
            3,
        )

    def test_filter_min_source_count_uses_unique_pmids_per_source(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_duplicate_source_reference_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_min_source_count(2)

        remaining_queries = pipeline.filtered_df["query"].tolist()
        self.assertEqual(remaining_queries, ["Unique enough."])
        self.assertEqual(
            pipeline.filtered_df.loc[
                pipeline.filtered_df["query"] == "Unique enough.",
                "n_sources",
            ].item(),
            2,
        )

    def test_filter_min_source_count_ignores_missing_string_pmids(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_missing_pmid_source_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_min_source_count(2)

        remaining_queries = pipeline.filtered_df["query"].tolist()
        self.assertEqual(remaining_queries, ["Complete references."])

    def test_filter_max_strangeness_filters_queries_by_configurable_threshold(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_max_strangeness(0.3)

        remaining_queries = pipeline.filtered_df["query"].tolist()
        self.assertNotIn("!!! ??? ###", remaining_queries)
        self.assertIn("Clean query.", remaining_queries)
        self.assertIn("Ambiguous citation.", remaining_queries)
        self.assertIn("strangeness", pipeline.filtered_df.columns)

    def test_filter_query_length_supports_min_and_max_word_thresholds(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_min_query_length(2).filter_max_query_length(2)

        remaining_queries = pipeline.filtered_df["query"].tolist()
        self.assertEqual(
            remaining_queries,
            [
                "Clean query.",
                "Ambiguous citation.",
                "Independent statement.",
                "Sparse references.",
            ],
        )
        self.assertIn("query_length", pipeline.filtered_df.columns)

    def test_filter_no_anaphoric_starts_filters_demonstrative_openings(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_cleanup_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_no_anaphoric_starts()

        remaining_queries = pipeline.filtered_df["query"].tolist()
        self.assertNotIn("This finding suggests an effect.", remaining_queries)
        self.assertNotIn("Those experiments confirmed the result.", remaining_queries)
        self.assertNotIn("The same mechanism was observed.", remaining_queries)
        self.assertIn("Clean biological statement.", remaining_queries)

    def test_filter_no_reference_framing_starts_filters_reference_dependent_openings(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_cleanup_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_no_reference_framing_starts()

        remaining_queries = pipeline.filtered_df["query"].tolist()
        self.assertNotIn("According to prior work, the pathway remained active.", remaining_queries)
        self.assertNotIn("As shown in Figure 2, cells were resistant.", remaining_queries)
        self.assertNotIn("As reported in earlier studies, resistance increased.", remaining_queries)
        self.assertIn("Clean biological statement.", remaining_queries)

    def test_filter_self_contained_queries_filters_queries_that_depend_on_context(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_cleanup_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_self_contained_queries()

        remaining_queries = pipeline.filtered_df["query"].tolist()
        self.assertNotIn("This finding suggests an effect.", remaining_queries)
        self.assertNotIn("It can lead to certain complications.", remaining_queries)
        self.assertNotIn("They were more resistant to treatment.", remaining_queries)
        self.assertNotIn("The latter mechanism remained active.", remaining_queries)
        self.assertNotIn("According to prior work, the pathway remained active.", remaining_queries)
        self.assertNotIn("Figure 2 shows that cells were resistant.", remaining_queries)
        self.assertNotIn("Cells were resistant in Figure 2.", remaining_queries)
        self.assertIn("Clean biological statement.", remaining_queries)
        self.assertIn("Overall, the pathway remained active.", remaining_queries)

    def test_filter_no_summary_starts_filters_discourse_summary_openings(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_cleanup_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_no_summary_starts()

        remaining_queries = pipeline.filtered_df["query"].tolist()
        self.assertNotIn("Taken together, the results support a feedback loop.", remaining_queries)
        self.assertNotIn("In summary, the treatment reduced growth.", remaining_queries)
        self.assertNotIn("Overall, the pathway remained active.", remaining_queries)
        self.assertIn("Clean biological statement.", remaining_queries)

    def test_cleanup_queries_removes_leading_connectives_and_invalidates_annotations(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_cleanup_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            pipeline.filter_max_strangeness(1.0)
            pipeline.cleanup_queries()

        remaining_queries = pipeline.filtered_df["query"].tolist()
        self.assertIn("The pathway remained active.", remaining_queries)
        self.assertIn("Cells were more resistant.", remaining_queries)
        self.assertIn("A second observation followed.", remaining_queries)
        self.assertIn("This finding suggests an effect.", remaining_queries)
        self.assertNotIn("strangeness", pipeline.filtered_df.columns)

    def test_select_shared_reference_subset_reuses_decoys_across_queries(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_selection_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            selected_df, selected_refs = pipeline.select_shared_reference_subset(2)

        self.assertEqual(selected_refs, ["1", "2"])
        self.assertTrue(selected_df["n_selected_decoys"].eq(2).all())
        self.assertEqual(
            selected_df["selected_decoys"].tolist(),
            [["1", "2"], ["1", "2"], ["1", "2"]],
        )

    def test_select_shared_reference_subset_excludes_hits_from_decoys(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_same_source_decoy_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            selected_df, selected_refs = pipeline.select_shared_reference_subset(2)

        self.assertEqual(selected_refs, ["10", "11", "12"])
        self.assertTrue(selected_df["n_selected_decoys"].eq(2).all())
        self.assertEqual(selected_df["selected_decoys"].tolist(), [["11", "12"], ["10", "12"]])
        self.assertNotIn("10", selected_df.iloc[0]["selected_decoys"])
        self.assertNotIn("11", selected_df.iloc[1]["selected_decoys"])

    def test_select_shared_reference_subset_limits_selection_to_relevant_pmcids(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_selection_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)
            original_filtered_df = pipeline.filtered_df.copy(deep=True)
            selected_df, selected_refs = pipeline.select_shared_reference_subset(
                2,
                relevant_pmcids=["PMC1", "PMC3"],
            )

        self.assertEqual(selected_df["source_pmcid"].tolist(), ["PMC1", "PMC3"])
        self.assertEqual(selected_refs, ["1", "2"])
        self.assertTrue(selected_df["n_selected_decoys"].eq(2).all())
        self.assertEqual(
            selected_df["selected_decoys"].tolist(),
            [["1", "2"], ["1", "2"]],
        )
        self.assertTrue(pipeline.filtered_df.equals(original_filtered_df))
        self.assertNotIn("hits_parsed", pipeline.filtered_df.columns)
        self.assertNotIn("selected_decoys", pipeline.filtered_df.columns)

    def test_select_shared_reference_subset_raises_when_n_exceeds_available_decoys(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self._write_selection_example_csvs(tmpdir)

            pipeline = QueryFilterPipeline(tmpdir)

            with self.assertRaises(ValueError):
                pipeline.select_shared_reference_subset(3)

    @staticmethod
    def _write_example_csvs(tmpdir: str) -> None:
        df_contexts = pd.DataFrame(
            [
                {
                    "query": "Clean query.",
                    "hits": "[1001]",
                    "n_hits": 1,
                    "source_pmcid": "PMC1",
                },
                {
                    "query": "Smith et al. reported the effect.",
                    "hits": "[1002]",
                    "n_hits": 1,
                    "source_pmcid": "PMC1",
                },
                {
                    "query": "Smith reported the effect.",
                    "hits": "[1003]",
                    "n_hits": 1,
                    "source_pmcid": "PMC1",
                },
                {
                    "query": "Ambiguous citation.",
                    "hits": "[1004, 1005]",
                    "n_hits": 2,
                    "source_pmcid": "PMC1",
                },
                {
                    "query": "Independent statement.",
                    "hits": "[2001]",
                    "n_hits": 1,
                    "source_pmcid": "PMC2",
                },
                {
                    "query": "Sparse references.",
                    "hits": "[3001]",
                    "n_hits": 1,
                    "source_pmcid": "PMC3",
                },
                {
                    "query": "!!! ??? ###",
                    "hits": "[4001]",
                    "n_hits": 1,
                    "source_pmcid": "PMC4",
                },
            ]
        )
        df_contexts.to_csv(f"{tmpdir}/extracted_contexts.csv", index=False)

        df_sources = pd.DataFrame(
            [
                {"source_pmcid": "PMC1", "pmid": "1001", "authors": "Adams A, Baker B"},
                {"source_pmcid": "PMC1", "pmid": "1002", "authors": "Smith J, Baker B"},
                {"source_pmcid": "PMC1", "pmid": "1003", "authors": "Cooper C"},
                {"source_pmcid": "PMC2", "pmid": "2001", "authors": "Lane A"},
                {"source_pmcid": "PMC2", "pmid": "2002", "authors": "Miller M"},
                {"source_pmcid": "PMC2", "pmid": "2003", "authors": "Jones J"},
                {"source_pmcid": "PMC3", "pmid": "3001", "authors": "Solo S"},
                {"source_pmcid": "PMC4", "pmid": "4001", "authors": "Odd O"},
            ]
        )
        df_sources.to_csv(f"{tmpdir}/extracted_sources.csv", index=False)

        df_metainfo = pd.DataFrame(
            [
                {
                    "abstract": "a",
                    "authors": "['Taylor T', 'Jordan J']",
                    "pmcid": "PMC1",
                    "pmid": "1",
                    "title": "t1",
                },
                {
                    "abstract": "b",
                    "authors": "['Lane A', 'Parker P']",
                    "pmcid": "PMC2",
                    "pmid": "2",
                    "title": "t2",
                },
                {
                    "abstract": "c",
                    "authors": "['Other O']",
                    "pmcid": "PMC3",
                    "pmid": "3",
                    "title": "t3",
                },
                {
                    "abstract": "d",
                    "authors": "['Odd O']",
                    "pmcid": "PMC4",
                    "pmid": "4",
                    "title": "t4",
                },
            ]
        )
        df_metainfo.to_csv(f"{tmpdir}/extracted_metadata.csv", index=False)

    @staticmethod
    def _write_same_source_decoy_example_csvs(tmpdir: str) -> None:
        df_contexts = pd.DataFrame(
            [
                {"query": "Q1", "hits": "[10]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "Q2", "hits": "[11]", "n_hits": 1, "source_pmcid": "PMC1"},
            ]
        )
        df_contexts.to_csv(f"{tmpdir}/extracted_contexts.csv", index=False)

        df_sources = pd.DataFrame(
            [
                {"source_pmcid": "PMC1", "pmid": "10", "authors": "A A"},
                {"source_pmcid": "PMC1", "pmid": "11", "authors": "B B"},
                {"source_pmcid": "PMC1", "pmid": "12", "authors": "C C"},
            ]
        )
        df_sources.to_csv(f"{tmpdir}/extracted_sources.csv", index=False)

        df_metainfo = pd.DataFrame(
            [
                {"abstract": "a", "authors": "['A A']", "pmcid": "PMC1", "pmid": "1", "title": "t1"},
            ]
        )
        df_metainfo.to_csv(f"{tmpdir}/extracted_metadata.csv", index=False)

    @staticmethod
    def _write_selection_example_csvs(tmpdir: str) -> None:
        df_contexts = pd.DataFrame(
            [
                {"query": "Q1", "hits": "[10]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "Q2", "hits": "[11]", "n_hits": 1, "source_pmcid": "PMC2"},
                {"query": "Q3", "hits": "[12]", "n_hits": 1, "source_pmcid": "PMC3"},
            ]
        )
        df_contexts.to_csv(f"{tmpdir}/extracted_contexts.csv", index=False)

        df_sources = pd.DataFrame(
            [
                {"source_pmcid": "PMC1", "pmid": "1", "authors": "A A"},
                {"source_pmcid": "PMC1", "pmid": "2", "authors": "B B"},
                {"source_pmcid": "PMC1", "pmid": "10", "authors": "C C"},
                {"source_pmcid": "PMC2", "pmid": "1", "authors": "A A"},
                {"source_pmcid": "PMC2", "pmid": "2", "authors": "B B"},
                {"source_pmcid": "PMC2", "pmid": "11", "authors": "D D"},
                {"source_pmcid": "PMC3", "pmid": "1", "authors": "A A"},
                {"source_pmcid": "PMC3", "pmid": "2", "authors": "B B"},
                {"source_pmcid": "PMC3", "pmid": "12", "authors": "E E"},
            ]
        )
        df_sources.to_csv(f"{tmpdir}/extracted_sources.csv", index=False)

        df_metainfo = pd.DataFrame(
            [
                {"abstract": "a", "authors": "['A A']", "pmcid": "PMC1", "pmid": "11", "title": "t1"},
                {"abstract": "b", "authors": "['B B']", "pmcid": "PMC2", "pmid": "12", "title": "t2"},
                {"abstract": "c", "authors": "['C C']", "pmcid": "PMC3", "pmid": "13", "title": "t3"},
            ]
        )
        df_metainfo.to_csv(f"{tmpdir}/extracted_metadata.csv", index=False)

    @staticmethod
    def _write_cleanup_example_csvs(tmpdir: str) -> None:
        df_contexts = pd.DataFrame(
            [
                {"query": "However, the pathway remained active.", "hits": "[1001]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "Therefore, cells were more resistant.", "hits": "[1002]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "In addition, a second observation followed.", "hits": "[1003]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "This finding suggests an effect.", "hits": "[1004]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "Those experiments confirmed the result.", "hits": "[1005]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "The same mechanism was observed.", "hits": "[1006]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "Clean biological statement.", "hits": "[1007]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "According to prior work, the pathway remained active.", "hits": "[1008]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "As shown in Figure 2, cells were resistant.", "hits": "[1009]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "As reported in earlier studies, resistance increased.", "hits": "[1010]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "Taken together, the results support a feedback loop.", "hits": "[1011]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "In summary, the treatment reduced growth.", "hits": "[1012]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "Overall, the pathway remained active.", "hits": "[1013]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "It can lead to certain complications.", "hits": "[1014]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "They were more resistant to treatment.", "hits": "[1015]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "The latter mechanism remained active.", "hits": "[1016]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "Figure 2 shows that cells were resistant.", "hits": "[1017]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "Cells were resistant in Figure 2.", "hits": "[1018]", "n_hits": 1, "source_pmcid": "PMC1"},
            ]
        )
        df_contexts.to_csv(f"{tmpdir}/extracted_contexts.csv", index=False)

        df_sources = pd.DataFrame(
            [
                {"source_pmcid": "PMC1", "pmid": "1001", "authors": "Adams A"},
                {"source_pmcid": "PMC1", "pmid": "1002", "authors": "Baker B"},
                {"source_pmcid": "PMC1", "pmid": "1003", "authors": "Cooper C"},
                {"source_pmcid": "PMC1", "pmid": "1004", "authors": "Dover D"},
                {"source_pmcid": "PMC1", "pmid": "1005", "authors": "Evans E"},
                {"source_pmcid": "PMC1", "pmid": "1006", "authors": "Fisher F"},
                {"source_pmcid": "PMC1", "pmid": "1007", "authors": "Green G"},
                {"source_pmcid": "PMC1", "pmid": "1008", "authors": "Hall H"},
                {"source_pmcid": "PMC1", "pmid": "1009", "authors": "Irwin I"},
                {"source_pmcid": "PMC1", "pmid": "1010", "authors": "Jones J"},
                {"source_pmcid": "PMC1", "pmid": "1011", "authors": "King K"},
                {"source_pmcid": "PMC1", "pmid": "1012", "authors": "Lane L"},
                {"source_pmcid": "PMC1", "pmid": "1013", "authors": "Moore M"},
                {"source_pmcid": "PMC1", "pmid": "1014", "authors": "North N"},
                {"source_pmcid": "PMC1", "pmid": "1015", "authors": "Olsen O"},
                {"source_pmcid": "PMC1", "pmid": "1016", "authors": "Perez P"},
                {"source_pmcid": "PMC1", "pmid": "1017", "authors": "Quinn Q"},
                {"source_pmcid": "PMC1", "pmid": "1018", "authors": "Reed R"},
            ]
        )
        df_sources.to_csv(f"{tmpdir}/extracted_sources.csv", index=False)

        df_metainfo = pd.DataFrame(
            [
                {"abstract": "a", "authors": "['Taylor T']", "pmcid": "PMC1", "pmid": "1", "title": "t1"},
            ]
        )
        df_metainfo.to_csv(f"{tmpdir}/extracted_metadata.csv", index=False)

    @staticmethod
    def _write_duplicate_source_reference_csvs(tmpdir: str) -> None:
        df_contexts = pd.DataFrame(
            [
                {"query": "Duplicate heavy.", "hits": "[5001]", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "Unique enough.", "hits": "[6001]", "n_hits": 1, "source_pmcid": "PMC2"},
            ]
        )
        df_contexts.to_csv(f"{tmpdir}/extracted_contexts.csv", index=False)

        df_sources = pd.DataFrame(
            [
                {"source_pmcid": "PMC1", "pmid": "5001", "authors": "Author A"},
                {"source_pmcid": "PMC1", "pmid": "5001", "authors": "Author A"},
                {"source_pmcid": "PMC2", "pmid": "6001", "authors": "Author B"},
                {"source_pmcid": "PMC2", "pmid": "6002", "authors": "Author C"},
            ]
        )
        df_sources.to_csv(f"{tmpdir}/extracted_sources.csv", index=False)

        df_metainfo = pd.DataFrame(
            [
                {"abstract": "a", "authors": "['Main A']", "pmcid": "PMC1", "pmid": "1", "title": "t1"},
                {"abstract": "b", "authors": "['Main B']", "pmcid": "PMC2", "pmid": "2", "title": "t2"},
            ]
        )
        df_metainfo.to_csv(f"{tmpdir}/extracted_metadata.csv", index=False)

    @staticmethod
    def _write_missing_pmid_source_csvs(tmpdir: str) -> None:
        df_contexts = pd.DataFrame(
            [
                {"query": "Complete references.", "hits": "['7001']", "n_hits": 1, "source_pmcid": "PMC1"},
                {"query": "Missing reference data.", "hits": "['8001']", "n_hits": 1, "source_pmcid": "PMC2"},
            ]
        )
        df_contexts.to_csv(f"{tmpdir}/extracted_contexts.csv", index=False)

        df_sources = pd.DataFrame(
            [
                {"source_pmcid": "PMC1", "pmid": "7001", "authors": "Author A"},
                {"source_pmcid": "PMC1", "pmid": "7002", "authors": "Author B"},
                {"source_pmcid": "PMC2", "pmid": None, "authors": "Author C"},
                {"source_pmcid": "PMC2", "pmid": "8001", "authors": "Author D"},
            ]
        )
        df_sources.to_csv(f"{tmpdir}/extracted_sources.csv", index=False)

        df_metainfo = pd.DataFrame(
            [
                {"abstract": "a", "authors": "['Main A']", "pmcid": "PMC1", "pmid": "1", "title": "t1"},
                {"abstract": "b", "authors": "['Main B']", "pmcid": "PMC2", "pmid": "2", "title": "t2"},
            ]
        )
        df_metainfo.to_csv(f"{tmpdir}/extracted_metadata.csv", index=False)


if __name__ == "__main__":
    unittest.main()
