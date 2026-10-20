"""Shared downstream scoring for the measurement-v5 rebenchmark.

This library is the single owner of record/subject reductions and exact
finite-sample class-prior baselines. It does not read datasets or run artifacts.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import binom
from sklearn.metrics import balanced_accuracy_score, f1_score

__all__ = ["concordance", "score_predictions", "classification_baselines", "regression_train_mean_baseline"]


def concordance(y: np.ndarray, prediction: np.ndarray) -> float | None:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.size < 2:
        return None
    denominator = y.var() + prediction.var() + (y.mean() - prediction.mean()) ** 2
    return float(2 * np.mean((y - y.mean()) * (prediction - prediction.mean())) / denominator) if denominator > 0 else None


def score_predictions(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray,
                      subjects: Sequence[str], task: str, num_classes: int,
                      record_ids: Sequence[str] | None = None) -> dict[str, Any]:
    subjects = np.asarray(subjects, dtype=str)
    per_subject = []
    for subject in sorted(set(subjects)):
        chosen = subjects == subject
        if task == "refed_regression":
            records = np.asarray(record_ids if record_ids is not None else subjects, dtype=str)
            record_scores = []
            for record in sorted(set(records[chosen])):
                selected = chosen & (records == record)
                dimensions = []
                for dimension in range(2):
                    valid = mask[selected, dimension].astype(bool)
                    dimensions.append(concordance(target[selected, dimension][valid], prediction[selected, dimension][valid]))
                # A missing dimension stays an explicit failed record endpoint.
                value = float(np.mean(dimensions)) if all(value is not None for value in dimensions) else None
                record_scores.append(dict(record_id=record, score=value, ccc_by_dimension=dimensions,
                                          sample_count=int(selected.sum())))
            valid_records = [row["score"] for row in record_scores if row["score"] is not None]
            value = float(np.mean(valid_records)) if valid_records else None
            per_subject.append(dict(subject_id=subject, score=value, records=record_scores,
                                    record_count=len(record_scores), defined_record_count=len(valid_records), sample_count=int(chosen.sum())))
        else:
            valid = mask[chosen].astype(bool)
            value = f1_score(target[chosen][valid], prediction[chosen][valid], labels=np.arange(num_classes), average="macro", zero_division=0)
            per_subject.append(dict(subject_id=subject, score=float(value), sample_count=int(valid.sum())))
    if not per_subject:
        raise ValueError("no evaluation subjects")
    valid_subjects = [row["score"] for row in per_subject if row["score"] is not None]
    diagnostic_primary=float(np.mean(valid_subjects)) if valid_subjects else None
    coverage_complete=len(valid_subjects)==len(per_subject) and all(
        row.get("defined_record_count",0)==row.get("record_count",0) for row in per_subject)
    result = dict(primary=diagnostic_primary if coverage_complete else None,
                  diagnostic_defined_primary=diagnostic_primary,coverage_complete=coverage_complete,
                  decision_eligible=coverage_complete,
                  primary_name="subject_equal_mean_ccc" if task == "refed_regression" else "subject_equal_macro_f1",
                  subjects=per_subject, subject_count=len(per_subject), defined_subject_count=len(valid_subjects), sample_count=len(target))
    if task != "refed_regression":
        valid = mask.astype(bool)
        result["pooled_balanced_accuracy"] = float(balanced_accuracy_score(target[valid], prediction[valid]))
    return result



def classification_baselines(training_targets,arrays,num_classes):
    """Train majority and exact finite-sample expectation under train priors."""
    train=np.asarray(training_targets,dtype=int)
    counts=np.bincount(train,minlength=num_classes)
    priors=counts/counts.sum();majority=int(counts.argmax())
    target=np.asarray(arrays["target"]);mask=np.asarray(arrays["mask"],dtype=bool)
    subjects=np.asarray(arrays["subject_id"])
    rows=[]
    for subject in sorted(set(subjects)):
        selected=(subjects==subject)&mask
        y=target[selected];n=len(y)
        if n==0:raise ValueError("baseline subject has no valid evaluation labels")
        fixed=float(f1_score(y,np.full(n,majority),labels=np.arange(num_classes),average="macro",zero_division=0))
        expected=[];m=np.arange(n+1,dtype=np.float64)
        for label,prior in enumerate(priors):
            positive=int((y==label).sum())
            fraction=np.divide(2*positive*m,n*(positive+m),out=np.zeros_like(m),where=positive+m>0)
            expected.append(float(np.dot(fraction,binom.pmf(m,n,float(prior)))))
        rows.append(dict(subject_id=str(subject),sample_count=n,majority_macro_f1=fixed,
                         prior_random_expected_macro_f1=float(np.mean(expected))))
    majority_score=float(np.mean([row["majority_macro_f1"] for row in rows]))
    prior_score=float(np.mean([row["prior_random_expected_macro_f1"] for row in rows]))
    return dict(majority_class=majority,train_class_counts=counts.tolist(),train_class_priors=priors.tolist(),
                majority_subject_equal_macro_f1=majority_score,
                prior_random_exact_expected_subject_equal_macro_f1=prior_score,
                b0=max(majority_score,prior_score),subjects=rows,
                prior_expectation="exact binomial finite-sample expectation; no Monte Carlo draws")


def regression_train_mean_baseline(train_mean,train_valid_point_counts,arrays):
    """Native MAE/RMSE: valid points → coordinates → records → subjects."""
    mean=np.asarray(train_mean,dtype=np.float64).reshape(2)
    counts=np.asarray(train_valid_point_counts,dtype=np.int64).reshape(2)
    if not np.isfinite(mean).all() or np.any(counts<1):
        raise ValueError("REFED mean baseline needs two finite training-fitted coordinates and support counts")
    target=np.asarray(arrays["target"],dtype=np.float64);mask=np.asarray(arrays["mask"],dtype=bool)
    subjects=np.asarray(arrays["subject_id"],dtype=str);records=np.asarray(arrays["record_id"],dtype=str)
    if target.ndim!=3 or target.shape[1]!=2 or target.shape!=mask.shape:
        raise ValueError("REFED baseline target and mask must preserve [window,2,time]")
    if not np.isfinite(target[mask]).all():raise ValueError("nonfinite valid REFED baseline target")
    rows=[]
    for subject in sorted(set(subjects)):
        subject_records=[]
        for record in sorted(set(records[subjects==subject])):
            selected=(subjects==subject)&(records==record)
            maes=[];rmses=[];support=[]
            for dimension in range(2):
                valid=mask[selected,dimension];values=target[selected,dimension][valid]
                support.append(int(values.size))
                residual=values-mean[dimension]
                maes.append(float(np.abs(residual).mean()) if values.size else None)
                rmses.append(float(np.sqrt(np.square(residual).mean())) if values.size else None)
            defined=all(value is not None for value in maes)
            subject_records.append(dict(record_id=record,mae=float(np.mean(maes)) if defined else None,
                rmse=float(np.mean(rmses)) if defined else None,mae_by_dimension=maes,rmse_by_dimension=rmses,
                valid_point_counts=support,sample_count=int(selected.sum())))
        defined=bool(subject_records) and all(row["mae"] is not None for row in subject_records)
        rows.append(dict(subject_id=subject,mae=float(np.mean([row["mae"] for row in subject_records])) if defined else None,
                         rmse=float(np.mean([row["rmse"] for row in subject_records])) if defined else None,
                         records=subject_records,record_count=len(subject_records),
                         defined_record_count=sum(row["mae"] is not None for row in subject_records)))
    complete=bool(rows) and all(row["mae"] is not None for row in rows)
    return dict(available=True,type="outer_training_coordinate_mean",b0=0.,train_mean=mean.tolist(),
                train_valid_point_counts=counts.tolist(),coverage_complete=complete,subjects=rows,
                mae=float(np.mean([row["mae"] for row in rows])) if complete else None,
                rmse=float(np.mean([row["rmse"] for row in rows])) if complete else None,
                error_reduction="native valid points within each record/coordinate; coordinate mean, record mean, then subject mean",
                sample_count=len(target),subject_count=len(rows))
