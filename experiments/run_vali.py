from binoculars.detector import Binoculars
from binoculars.detector import BINOCULARS_ACCURACY_THRESHOLD as THRESHOLD
from experiments.utils import convert_to_pandas, save_experiment

import os
import argparse
import datetime

import torch
from datasets import Dataset, logging as datasets_logging
import numpy as np
from sklearn import metrics


def main(args):
    # Initialize Binoculars (experiments in paper use the "accuracy" mode threshold wherever applicable)
    bino = Binoculars(mode="accuracy", max_token_observed=args.tokens_seen)

    # Load dataset
    ds = Dataset.from_json(f"{args.dataset_path}")

    # Set (non) default values
    args.dataset_name = args.dataset_name or args.dataset_path.rstrip("/").split("/")[-2]
    machine_sample_key = (
            args.machine_sample_key
            or [x for x in list(ds.features.keys())[::-1] if "generated_text" in x][0]
    )
    args.machine_text_source = args.machine_text_source or machine_sample_key.rstrip("_generated_text_wo_prompt")

    # Set job name, experiment path and create directory
    args.job_name = (
            args.job_name
            or f"{args.dataset_name}-{args.machine_text_source}-{args.tokens_seen}-tokens"
            .strip().replace(' ', '-')
    )
    #breakpoint()
    args.experiment_path = f"results/vali/{args.job_name}"
    os.makedirs(f"{args.experiment_path}", exist_ok=True)

    # Score human and machine generated text
    print(f"Scoring human text")
    human_scores = ds.map(
        lambda batch: {"score": bino.compute_score(batch[args.human_sample_key])},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=ds.column_names
    )

    print(f"Scoring machine text")
    machine_scores = ds.map(
        lambda batch: {"score": bino.compute_score(batch[args.machine_sample_key])},
        batched=True,
        batch_size=args.batch_size,
        remove_columns=ds.column_names
    )

    score_df = convert_to_pandas(human_scores, machine_scores)
    # 閾値を求めるプログラムなのでaccuracyモードの閾値は用いない
    # score_df["pred"] = np.where(score_df["score"] < THRESHOLD, 1, 0) scoreカラムの値がTHRESHOLDより小さい場合にpredカラムに1を、そうでない場合に0を設定

    # F1スコアが最大となる閾値を求める
    thresholds = np.arange(0.0, 1.5, 0.0001)
    best_threshold = None
    best_f1_score = 0

    for threshold in thresholds:
        score_df["pred"] = np.where(score_df["score"] < threshold, 1, 0)

        # Compute metrics
        f1_score = metrics.f1_score(score_df["class"], score_df["pred"])
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = threshold
        
    print(f"Best Threshold by F1 Score: {best_threshold}, Best F1 Score: {best_f1_score}")

    # 偽陽性(人間 のテキストが誤ってAIとみなされる)率が0.01%未満となる閾値を求める
    score = -1 * score_df["score"]
    fpr, tpr, thresholds = metrics.roc_curve(y_true=score_df["class"], y_score=score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    tpr_at_fpr_0_2 = np.interp(0.2 / 100, fpr, tpr)
    threshold_at_fpr_0_2 = thresholds[np.where(fpr < 0.002)[0][-1]]
    print(f"Best Threshold by low FPR: {threshold_at_fpr_0_2}")

    """
    # Compute metrics
    f1_score = metrics.f1_score(score_df["class"], score_df["pred"])
    score = -1 * score_df["score"]  # We negative scale the scores to make the class 1 (machine) the positive class
    fpr, tpr, thresholds = metrics.roc_curve(y_true=score_df["class"], y_score=score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    
    # Interpolate the TPR at FPR = 0.01%, this is a fixed point in roc curve
    tpr_at_fpr_0_01 = np.interp(0.01 / 100, fpr, tpr)
    """

    # Save experiment
    save_experiment(args, score_df, fpr, tpr, best_f1_score, roc_auc, tpr_at_fpr_0_2, best_threshold, threshold_at_fpr_0_2)


if __name__ == "__main__":
    print("=" * 60, "START", "=" * 60)

    # Set logging at the CRITICAL level to avoid seeing loaded datasets from cache
    datasets_logging.set_verbosity_error()

    parser = argparse.ArgumentParser(
        description="Run (default) Binoculars on a dataset and compute/plot relevant metrics.",
    )

    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, help="Path to the jsonl file")
    parser.add_argument("--dataset_name", type=str, default=None, help="name of the dataset")
    parser.add_argument("--human_sample_key", type=str, help="key for the human-generated text")
    parser.add_argument("--machine_sample_key", type=str, default=None,
                        help="key for the machine-generated text")
    parser.add_argument("--machine_text_source", type=str, default=None,
                        help="name of model used to generate machine text")

    # Scoring arguments ここで判別に用いるトークン数を変える
    parser.add_argument("--tokens_seen", type=int, default=32, help="Number of tokens seen by the model")

    # Computational arguments ここで一度にスコアを計算する文章の数を変える
    parser.add_argument("--batch_size", type=int, default=32)

    # Job arguments
    parser.add_argument("--job_name", type=str, default=None)

    args = parser.parse_args()

    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Type: {torch.cuda.get_device_name(0)}")

    args.start_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    main(args)

    print("=" * 60, "END", "=" * 60)
