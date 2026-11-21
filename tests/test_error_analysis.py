"""TIDE-style matching used for Table T11."""
import trace_experiments as te

GT = [{"cls": "car", "box": [0, 0, 100, 100]},
      {"cls": "van", "box": [200, 200, 260, 260]},
      {"cls": "bus", "box": [400, 400, 500, 500]}]


def types(preds):
    return [e["type"] for e in te.match_image(GT, preds)]


def test_true_positive_and_duplicate():
    t = types([{"cls": "car", "conf": .9, "box": [2, 2, 98, 98]},
               {"cls": "car", "conf": .8, "box": [0, 0, 100, 100]}])
    assert t.count("TP") == 1 and t.count("Duplicate") == 1


def test_classification_error_is_not_also_missed():
    t = types([{"cls": "bus", "conf": .7, "box": [200, 200, 260, 262]}])
    assert "Classification" in t
    assert t.count("Missed") == 2              # car and bus missed, van explained by the cls error


def test_background_false_positive():
    assert "Background" in types([{"cls": "car", "conf": .5, "box": [700, 700, 720, 720]}])


def test_localization_error():
    assert "Localization" in types([{"cls": "car", "conf": .6, "box": [50, 50, 150, 150]}])


def test_size_buckets_follow_coco():
    assert te.size_bucket([0, 0, 31, 31]) == "small"
    assert te.size_bucket([0, 0, 50, 50]) == "medium"
    assert te.size_bucket([0, 0, 100, 100]) == "large"
