"""Cohort-level metric aggregation helper."""


def combine(videos):
    events = [e for row in videos.values() for e in row["windows"]["0.5"]]
    return {"timely": sum(e["correct_recovered"] for e in events),
            "first": sum(e["category"] == "correct_first" for e in events),
            "fp": sum(row["false_reports"] for row in videos.values()),
            "visible_mean_iou": sum(row["visible_iou_sum"] for row in videos.values()) /
            sum(row["visible"] for row in videos.values())}
