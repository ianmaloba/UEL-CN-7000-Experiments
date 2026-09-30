from analysis.aggregate_model_coverage import _pair_status


def test_pair_status_keeps_generation_and_infrastructure_outcomes_distinct():
    assert _pair_status("complete", "complete") == "complete_pair"
    assert _pair_status("complete", "incomplete_response") == "mixed_generation_pair"
    assert _pair_status("incomplete_response", "empty_response") == "incomplete_pair"
    assert _pair_status("provider_error", "complete") == "infrastructure_error_pair"
    assert _pair_status("missing", "complete") == "missing_pair"
