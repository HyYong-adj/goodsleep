from psg_only.metrics import evaluate_predictions


def test_perfect_predictions_have_perfect_scores():
    rows = []
    for index in range(8):
        label = index % 4
        rows.append({"subject_id": "s1", "epoch_index": index, "target": label, "prediction": label, "valid": True})
    result = evaluate_predictions(rows)
    assert result["status"] == "complete"
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_missing_class_is_invalid():
    rows = [{"subject_id": "s1", "epoch_index": i, "target": 0, "prediction": 0, "valid": True} for i in range(3)]
    assert evaluate_predictions(rows)["status"] == "invalid"


def test_temporal_metrics_do_not_bridge_missing_epochs():
    rows = [
        {"subject_id": "s1", "epoch_index": index * 2, "target": index, "prediction": index, "valid": True}
        for index in range(4)
    ]
    metrics = evaluate_predictions(rows)
    assert metrics["status"] == "complete"
    assert metrics["transition_macro_f1"] is None
    assert metrics["transition_rate_error"] is None


def test_metrics_report_majority_baseline():
    rows = [
        {"subject_id": "s1", "epoch_index": index, "target": index % 4, "prediction": index % 4, "valid": True}
        for index in range(8)
    ]
    assert evaluate_predictions(rows)["majority_class_baseline"]["class_name"] == "Wake"
