from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    return {
        "roc_auc":               round(roc_auc_score(y_test, y_prob), 4),
        "accuracy":              round(accuracy_score(y_test, y_pred), 4),
        "f1":                    round(f1_score(y_test, y_pred), 4),
        "precision":             round(precision_score(y_test, y_pred), 4),
        "recall":                round(recall_score(y_test, y_pred), 4),
        "confusion_matrix":      confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "roc_fpr":               fpr,
        "roc_tpr":               tpr,
        "pr_precision":          precision_curve,
        "pr_recall":             recall_curve,
        "y_prob":                y_prob,
        "y_pred":                y_pred,
    }


def compare(all_results: dict) -> None:
    print("\n=== Model Comparison ===")
    print(f"{'Model':<25} {'ROC-AUC':>10} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 77)
    for name, res in all_results.items():
        print(f"{name:<25} {res['roc_auc']:>10.4f} {res['accuracy']:>10.4f} "
              f"{res['f1']:>10.4f} {res['precision']:>10.4f} {res['recall']:>10.4f}")
    print()
