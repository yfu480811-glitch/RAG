from tracerag.retrieval import rrf_fuse


def test_rrf_prefers_items_present_in_both_lists() -> None:
    list1 = ["a", "b", "c"]
    list2 = ["x", "b", "y"]

    fused = rrf_fuse(list1, list2, k=60)
    ranked_ids = [x[0] for x in fused]

    assert ranked_ids[0] == "b"
    assert "a" in ranked_ids and "x" in ranked_ids
