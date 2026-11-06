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
                      record_ids: Sequence[str] | None = None, *,
                      metric_version: str = "v1") -> dict[str, Any]:
    if metric_version not in {"v1", "v2"}:
        raise ValueError("known downstream metric version required")
    target, prediction, mask = np.asarray(target), np.asarray(prediction), np.asarray(mask)
    if metric_version == "v2":
        if target.shape != prediction.shape or target.shape != mask.shape or mask.dtype != np.bool_:
            raise ValueError("v2 predictions, targets and boolean mask must have identical support")
        if not np.isfinite(target[mask]).all() or not np.isfinite(prediction[mask]).all():
            raise ValueError("nonfinite values on scored support")
        if task == "refed_regression" and (target.ndim != 3 or target.shape[1] != 2):
            raise ValueError("REFED must preserve [window,2,time]")
    subjects = np.asarray(subjects, dtype=str)
    if metric_version == "v2":
        if subjects.shape != (len(target),) or (record_ids is not None and np.asarray(record_ids).shape != subjects.shape):
            raise ValueError("subject and record identities must match the prediction rows")
        if task != "refed_regression":
            if target.ndim != 1 or num_classes < 2 or any(not mask[subjects == s].any() for s in np.unique(subjects)):
                raise ValueError("classification requires a valid target for every subject")
            for values in (target[mask], prediction[mask]):
                if np.any(values != values.astype(int)) or np.any((values < 0) | (values >= num_classes)):
                    raise ValueError("classification values must use the complete fixed class vocabulary")
    per_subject = []
    for subject in sorted(set(subjects)):
        chosen = subjects == subject
        if task == "refed_regression":
            records = np.asarray(record_ids if record_ids is not None else subjects, dtype=str)
            record_scores = []
            for record in sorted(set(records[chosen])):
                selected = chosen & (records == record)
                dimensions, maes, rmses, constant = [], [], [], []
                for dimension in range(2):
                    valid = mask[selected, dimension].astype(bool)
                    y, p = target[selected, dimension][valid], prediction[selected, dimension][valid]
                    is_constant = bool(y.size >= 2 and np.all(y == y[0]))
                    constant.append(is_constant)
                    dimensions.append(None if metric_version == "v2" and is_constant else concordance(y, p))
                    residual = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
                    maes.append(float(np.abs(residual).mean()) if residual.size else None)
                    rmses.append(float(np.sqrt(np.square(residual).mean())) if residual.size else None)
                # A missing dimension stays an explicit failed record endpoint.
                eligible = [value for value, flat in zip(dimensions, constant)
                            if not (metric_version == "v2" and flat)]
                value = float(np.mean(eligible)) if eligible and all(value is not None for value in eligible) else None
                record_scores.append(dict(record_id=record, score=value, ccc_by_dimension=dimensions,
                                          mae_by_dimension=maes, rmse_by_dimension=rmses,
                                          mae=float(np.mean(maes)) if all(v is not None for v in maes) else None,
                                          rmse=float(np.mean(rmses)) if all(v is not None for v in rmses) else None,
                                          sample_count=int(selected.sum())))
                if metric_version == "v2":
                    record_scores[-1]["constant_target_by_dimension"] = constant
                    record_scores[-1]["dynamic_ccc_eligible_by_dimension"] = [not value for value in constant]
            valid_records = [row["score"] for row in record_scores if row["score"] is not None]
            value = float(np.mean(valid_records)) if valid_records else None
            coordinate_ccc = []
            if metric_version == "v2":
                for dimension in range(2):
                    values = [row["ccc_by_dimension"][dimension] for row in record_scores
                              if row["dynamic_ccc_eligible_by_dimension"][dimension]]
                    coordinate_ccc.append(float(np.mean(values)) if values and all(v is not None for v in values) else None)
                value = float(np.mean(coordinate_ccc)) if all(v is not None for v in coordinate_ccc) else None
            per_subject.append(dict(subject_id=subject, score=value, records=record_scores,
                                    record_count=len(record_scores), defined_record_count=len(valid_records), sample_count=int(chosen.sum())))
            for error in ("mae", "rmse"):
                values = [row[error] for row in record_scores]
                per_subject[-1][error] = float(np.mean(values)) if all(v is not None for v in values) else None
                if metric_version == "v2":
                    per_subject[-1][error + "_by_dimension"] = [
                        float(np.mean([row[error + "_by_dimension"][c] for row in record_scores]))
                        if all(row[error + "_by_dimension"][c] is not None for row in record_scores) else None
                        for c in range(2)]
            if metric_version == "v2":
                per_subject[-1].update(ccc_by_dimension=coordinate_ccc,
                    dynamic_ccc_record_counts=[sum(row["dynamic_ccc_eligible_by_dimension"][c] for row in record_scores) for c in range(2)],
                    constant_target_record_counts=[sum(row["constant_target_by_dimension"][c] for row in record_scores) for c in range(2)])
        else:
            valid = mask[chosen].astype(bool)
            value = f1_score(target[chosen][valid], prediction[chosen][valid], labels=np.arange(num_classes), average="macro", zero_division=0)
            per_subject.append(dict(subject_id=subject, score=float(value), sample_count=int(valid.sum())))
            per_subject[-1]["accuracy"] = float(np.mean(target[chosen][valid] == prediction[chosen][valid])) if valid.any() else None
    if not per_subject:
        raise ValueError("no evaluation subjects")
    valid_subjects = [row["score"] for row in per_subject if row["score"] is not None]
    diagnostic_primary=float(np.mean(valid_subjects)) if valid_subjects else None
    coverage_complete=len(valid_subjects)==len(per_subject) and all(
        row.get("defined_record_count",0)==row.get("record_count",0) for row in per_subject)
    if metric_version == "v2" and task == "refed_regression":
        coverage_complete = len(valid_subjects) == len(per_subject) and all(
            row["mae"] is not None and row["rmse"] is not None for row in per_subject)
    result = dict(primary=diagnostic_primary if coverage_complete else None,
                  diagnostic_defined_primary=diagnostic_primary,coverage_complete=coverage_complete,
                  decision_eligible=coverage_complete,
                  primary_name="subject_equal_mean_ccc" if task == "refed_regression" else "subject_equal_macro_f1",
                  subjects=per_subject, subject_count=len(per_subject), defined_subject_count=len(valid_subjects), sample_count=len(target))
    if task != "refed_regression":
        valid = mask.astype(bool)
        result["pooled_balanced_accuracy"] = float(balanced_accuracy_score(target[valid], prediction[valid]))
        result["pooled_accuracy"] = float(np.mean(target[valid] == prediction[valid]))
        result["subject_equal_accuracy"] = float(np.mean([row["accuracy"] for row in per_subject])) if all(row["accuracy"] is not None for row in per_subject) else None
    else:
        for error in ("mae", "rmse"):
            values = [row[error] for row in per_subject]
            result[error] = float(np.mean(values)) if all(v is not None for v in values) else None
        if metric_version == "v2":
            result["primary_name"] = "subject_equal_mean_dynamic_ccc"
            for metric in ("ccc", "mae", "rmse"):
                result[metric + "_by_dimension"] = [
                    float(np.mean([row[metric + "_by_dimension"][c] for row in per_subject]))
                    if all(row[metric + "_by_dimension"][c] is not None for row in per_subject) else None
                    for c in range(2)]
            result["constant_target_record_counts"] = [sum(row["constant_target_record_counts"][c] for row in per_subject) for c in range(2)]
    if metric_version == "v2":
        result["metric_version"] = metric_version
        result["subject_sd"] = float(np.std(valid_subjects, ddof=1)) if coverage_complete and len(valid_subjects) > 1 else None
        if task != "refed_regression":
            values = [row["accuracy"] for row in per_subject]
            result["accuracy_subject_sd"] = float(np.std(values, ddof=1)) if all(v is not None for v in values) and len(values) > 1 else None
        else:
            for metric in ("ccc", "mae", "rmse"):
                result[metric + "_subject_sd_by_dimension"] = [
                    float(np.std([row[metric + "_by_dimension"][c] for row in per_subject], ddof=1))
                    if len(per_subject) > 1 and all(row[metric + "_by_dimension"][c] is not None for row in per_subject) else None
                    for c in range(2)]
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
                         majority_accuracy=float(np.mean(y == majority)),
                         prior_random_expected_accuracy=float(np.mean(priors[y])),
                         prior_random_expected_macro_f1=float(np.mean(expected))))
    majority_score=float(np.mean([row["majority_macro_f1"] for row in rows]))
    prior_score=float(np.mean([row["prior_random_expected_macro_f1"] for row in rows]))
    return dict(majority_class=majority,train_class_counts=counts.tolist(),train_class_priors=priors.tolist(),
                majority_subject_equal_macro_f1=majority_score,
                prior_random_exact_expected_subject_equal_macro_f1=prior_score,
                majority_subject_equal_accuracy=float(np.mean([row["majority_accuracy"] for row in rows])),
                prior_random_expected_subject_equal_accuracy=float(np.mean([row["prior_random_expected_accuracy"] for row in rows])),
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
