import json
from pathlib import Path

import numpy as np
from discopat.metrics import compute_iomean

# EVAL_SETS = ["test", "private_test"]
EVAL_SETS = ["train"]

def match_gts_and_preds(
    groundtruths: list,
    predictions: list,
    scores: list,
    threshold: float
) -> tuple[int, list[tuple[float, float]]]:
    """Match GTs and preds on an image in the dataset.

    Args:
        groundtruths: list of groundtruths, boxes [x1, y1, x2, y2],
        predictions: list of predictions, boxes [x1, y1, x2, y2],
        scores: list of confidence score, same length as predictions
        threshold: threshold for the localization metric,
        localization_criterion: metric used to assess the fit between GTs and preds.

    Returns:
        A report for the considered image, containing:
            - The total number of groundtruths,
            - For each pred, a tuple (score, is_tp).

    """

    # Sort predictions by score descending
    predictions = np.array(predictions)
    scores = np.array(scores)
    order = np.argsort(-scores)
    predictions = predictions[order]
    scores = scores[order]

    # Track matches
    gt_matched = np.zeros(len(groundtruths), dtype=bool)
    tps = np.zeros(len(predictions))

    for i, pred in enumerate(predictions):
        # Find best matching GT
        loc_scores = [compute_iomean(pred, gt) for gt in groundtruths]
        best_gt = int(np.argmax(loc_scores))
        best_loc = loc_scores[best_gt]
        if best_loc >= threshold and not gt_matched[best_gt]:
            tps[i] = 1
            gt_matched[best_gt] = True

    return len(groundtruths), np.array(list(zip(scores, tps))).reshape(-1, 2)


def compute_ap(
    predictions,
    targets,
    threshold: float
) -> float:
    """Compute the Average Precision (AP) for a given localization threshold.

    Args:
        model: the neural network to be evaluated,
        data_loader: the evaluation dataloader,
        threshold: localization threshold,
        localization_criterion: metric used to match groundtruths and preds.

    Returns:
        The AP.

    """
    num_groundtruths = 0
    big_tp_vector = np.empty((0, 2))
    for frame in targets:
        target = targets[frame]
        if frame not in predictions:
            num_groundtruths += len(target)
            continue
        output = predictions[frame]
        num_gts, tp_vector = match_gts_and_preds(
            groundtruths=target["boxes"],
            predictions=output["boxes"],
            scores=output["scores"],
            threshold=threshold
        )
        num_groundtruths += num_gts
        big_tp_vector = np.concat((big_tp_vector, tp_vector))
    if num_groundtruths == 0:
        return 0

    # Sort the TP vector by decreasing prediction score over the whole dataset
    big_tp_vector = big_tp_vector[np.argsort(-big_tp_vector[:, 0])]

    # Cumulative sums
    tp_cumulative = np.cumsum(big_tp_vector[:, 1])
    fp_cumulative = np.cumsum(1 - big_tp_vector[:, 1])

    # Prepend zeros for the case score_threshold=1
    tp_cum = np.concatenate([[0], tp_cumulative])
    fp_cum = np.concatenate([[0], fp_cumulative])

    recall = tp_cum / num_groundtruths
    precision = tp_cum / (tp_cum + fp_cum + 1e-10)

    # Ensure precision is non-increasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Compute area under curve (AP)
    return np.trapezoid(precision, recall)


def read_xml(path):
    from xml.etree.ElementTree import parse

    tree = parse(path)
    root = tree.getroot()
    annotations = {}
    for element in root.findall("image"):
        if "index" in element.attrib:
            frame_index = element.attrib["index"].split("-")[1]
        else:
            frame_index = element.attrib["name"].split(".")[0]
        annotations[frame_index] = {
            "boxes": [
                (
                    float(bbox.attrib["xtl"]),
                    float(bbox.attrib["ytl"]),
                    float(bbox.attrib["xbr"]),
                    float(bbox.attrib["ybr"]),
                ) for bbox in element.findall("box")
            ],
            "scores": [
                float(bbox.attrib.get("score", 1.0))
                for bbox in element.findall("box")
            ]
        }
    return annotations


def main(reference_dir, prediction_dir, output_dir):
    scores = {}
    for eval_set in EVAL_SETS:
        print(f'Scoring {eval_set}')

        predictions = read_xml(prediction_dir / f'{eval_set}_predictions.xml')
        targets = read_xml(reference_dir / f'{eval_set}_labels.xml')

        scores[eval_set] = float(compute_ap(
            predictions, targets,
            threshold=0.5
        ))

    # Add train and test times in the score
    json_durations = (prediction_dir / 'metadata.json').read_text()
    durations = json.loads(json_durations)
    scores.update(**durations)
    print(scores)

    # Write output scores
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'scores.json').write_text(json.dumps(scores))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scoring program for codabench"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="/app/input/ref",
        help="",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default="/app/input/res",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )

    args = parser.parse_args()

    main(
        Path(args.reference_dir),
        Path(args.prediction_dir),
        Path(args.output_dir)
    )
