"""Compare five winner classifiers and test opening rules across bot tiers."""
import argparse
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from simulate import FEATURES, RULES, SIZES, STRATEGIES


EVALUATION_SEED = 513
KEYS = ["players", "tier", "seed"]
MODELS = {
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(25, weights="distance")),
    "Naive Bayes": GaussianNB(),
    "CART": DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=EVALUATION_SEED),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, min_samples_leaf=8, random_state=EVALUATION_SEED, n_jobs=2
    ),
    "ANN": make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=0.01, max_iter=400,
            early_stopping=True, random_state=EVALUATION_SEED,
        ),
    ),
}


def check_and_partition(rows):
    if set(map(tuple, rows[["players", "tier"]].drop_duplicates().to_numpy())) != set(product(SIZES, RULES)):
        raise ValueError("The dataset must include all four scenarios")
    if rows.duplicated([*KEYS, "seat"]).any():
        raise ValueError("More than one row was found for a seat in a game")
    games = rows.groupby(KEYS, sort=True)
    if not games.size().eq(games.size().index.get_level_values("players")).all():
        raise ValueError("Each game must have one row for each seat")
    for size in SIZES:
        by_size = rows[rows["players"] == size]
        columns = ["seed", "seat", "settlement_1", "settlement_2", "road_1", "road_2", *FEATURES]
        basic = by_size[by_size["tier"] == "basic"][columns].sort_values(["seed", "seat"])
        advanced = by_size[by_size["tier"] == "advanced"][columns].sort_values(["seed", "seat"])
        if not basic.reset_index(drop=True).equals(advanced.reset_index(drop=True)):
            raise ValueError("Basic and Advanced games must have identical matched openings")
    if not games["winner_seat"].nunique(dropna=False).eq(1).all():
        raise ValueError("Winner labels disagree within a game")
    completed = rows.dropna(subset=["winner_seat"]).copy()
    completed["winner_seat"] = completed["winner_seat"].astype(int)
    if not completed["won"].eq(completed["winner_seat"].eq(completed["seat"])).all():
        raise ValueError("Player and game winner labels disagree")
    if not completed.groupby(KEYS)["won"].sum().eq(1).all():
        raise ValueError("Each completed game must have exactly one winner")
    seeds = sorted(rows["seed"].unique())
    train, test = train_test_split(seeds, test_size=0.25, random_state=EVALUATION_SEED)
    return completed, set(train), set(test)


def game_features(rows):
    matrix = rows.pivot(index="seed", columns="seat", values=list(FEATURES)).sort_index()
    matrix.columns = [f"seat{seat}_{feature}" for feature, seat in matrix.columns]
    winner = rows.groupby("seed")["winner_seat"].first().reindex(matrix.index)
    return matrix, winner


def plot_confusions(matrices, size, tier, output):
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained")
    figure.suptitle(f"{size} players | {tier.title()} bots | Held-out games",
                    fontsize=17, fontweight="bold")
    maximum = max(int(matrix.max()) for matrix in matrices.values())
    for axis, (name, matrix) in zip(axes.flat, matrices.items()):
        heatmap = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for actual, predicted in np.ndindex(matrix.shape):
            n = int(matrix[actual, predicted])
            axis.text(predicted, actual, str(n), ha="center", va="center",
                      color="white" if n > 0.55 * maximum else "#0c3154",
                      fontweight="bold")
        axis.set(
            title=name, xlabel="Predicted winning seat", ylabel="Actual winning seat",
            xticks=range(size), yticks=range(size),
            xticklabels=range(1, size + 1), yticklabels=range(1, size + 1),
        )
    figure.colorbar(heatmap, ax=list(axes.flat), label="Number of games", shrink=0.75)
    figure.savefig(output / f"confusion_{size}p_{tier}.png", dpi=170)
    plt.close(figure)


def paired_stats(truth, predicted, reference):
    test_correct = np.asarray(predicted) == np.asarray(truth)
    reference_correct = np.asarray(reference) == np.asarray(truth)
    diff = test_correct.astype(int) - reference_correct.astype(int)
    gains = int(np.count_nonzero(diff == 1))
    losses = int(np.count_nonzero(diff == -1))
    p_value = binomtest(gains, gains + losses, p=0.5).pvalue if gains + losses else 1.0
    rng = np.random.default_rng(EVALUATION_SEED)
    sampled = rng.multinomial(len(diff), [losses / len(diff), 1 - (gains + losses) / len(diff),
                                          gains / len(diff)], size=10000)
    low, high = np.quantile((sampled[:, 2] - sampled[:, 0]) / len(diff), [0.025, 0.975])
    return {
        "lift": diff.mean(), "lift_ci_low": low, "lift_ci_high": high,
        "p_value": p_value,
    }


def score_openings(rows, winner, size, tier, baseline):
    scores = []
    flat = rows[rows["seed"].isin(winner.index)].sort_values(["seed", "seat"]).copy()
    # Fixed random ordering breaks feature ties independently of seats and outcomes.
    rng = np.random.default_rng(EVALUATION_SEED)
    flat["tie_order"] = rng.random(len(flat))
    for rule, feature in STRATEGIES.items():
        ordered = flat.sort_values([feature, "tie_order"], ascending=False)
        pick = ordered.drop_duplicates("seed").set_index("seed")["seat"].reindex(winner.index)
        scores.append({
            "players": size, "tier": tier, "rule": rule, "test_games": len(winner),
            "accuracy": accuracy_score(winner, pick), "baseline_accuracy": accuracy_score(winner, baseline),
            **paired_stats(winner, pick, baseline),
        })
    return scores


def evaluate(input_path, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(input_path)
    completed, train_seeds, test_seeds = check_and_partition(rows)
    pd.DataFrame({"seed": sorted(train_seeds | test_seeds)}).assign(
        split=lambda table: np.where(table["seed"].isin(train_seeds), "train", "test")
    ).to_csv(output / "split.csv", index=False)

    summaries, metrics, reports, confusions, predictions, opening_scores = [], [], [], [], [], []
    forests, holdouts = {}, {}
    for size, tier in product(SIZES, RULES):
        scenario = completed[(completed["players"] == size) & (completed["tier"] == tier)]
        attempts = rows.loc[(rows["players"] == size) & (rows["tier"] == tier), "seed"].nunique()
        x, y = game_features(scenario)
        train = x.index.isin(train_seeds)
        test = x.index.isin(test_seeds)
        x_train, y_train, x_test, y_test = x.loc[train], y.loc[train], x.loc[test], y.loc[test]
        labels = list(range(1, size + 1))
        if set(y_train) != set(labels) or not len(y_test):
            raise ValueError(f"Insufficient valid games for {size} players, {tier}")
        baseline_seat = int(y_train.value_counts().sort_index().idxmax())
        baseline = np.full(len(y_test), baseline_seat)
        info = {"players": size, "tier": tier}
        summaries.append({
            **info, "attempts": attempts, "completed": len(y), "no_winner": attempts - len(y),
            "train_games": len(y_train), "test_games": len(y_test), "baseline_seat": baseline_seat,
            "uniform_chance": 1 / size,
        })
        model_predictions = {"Seat baseline": baseline}
        for name, model in MODELS.items():
            fitted = clone(model).fit(x_train, y_train)
            model_predictions[name] = fitted.predict(x_test)
            if name == "Random Forest":
                forests[size, tier] = fitted
        holdouts[size, tier] = (x_test, y_test, baseline)
        matrices = {}
        for name, winner_hat in model_predictions.items():
            model_info = {**info, "model": name}
            report = classification_report(
                y_test, winner_hat, labels=labels, output_dict=True, zero_division=0
            )
            metrics.append({
                **model_info, "test_games": len(y_test), "accuracy": report["accuracy"],
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                **paired_stats(y_test, winner_hat, baseline),
            })
            reports.extend({**model_info, "label": label, **values}
                           for label, values in report.items() if isinstance(values, dict))
            matrix = confusion_matrix(y_test, winner_hat, labels=labels)
            assert matrix.sum() == len(y_test)
            matrices[name] = matrix
            confusions.extend({**model_info, "actual": labels[i], "predicted": labels[j],
                               "count": int(matrix[i, j])} for i, j in np.ndindex(matrix.shape))
            predictions.extend({**model_info, "seed": seed, "actual": actual, "predicted": predicted}
                               for seed, actual, predicted in zip(y_test.index, y_test, winner_hat))
        plot_confusions(matrices, size, tier, output)
        opening_scores.extend(score_openings(scenario, y_test, size, tier, baseline))
        print(f"Evaluated {size} players, {tier} ({len(y_test)} held-out games)", flush=True)

    transfer = []
    for size, target in product(SIZES, RULES):
        source = next(tier for tier in RULES if tier != target)
        x_test, y_test, baseline = holdouts[size, target]
        winner_hat = forests[size, source].predict(x_test)
        transfer.append({
            "players": size, "trained_on": source, "tested_on": target, "test_games": len(y_test),
            "accuracy": accuracy_score(y_test, winner_hat),
            "baseline_accuracy": accuracy_score(y_test, baseline),
            **paired_stats(y_test, winner_hat, baseline),
        })

    model_metrics = pd.DataFrame(metrics)
    comparisons = model_metrics["model"] != "Seat baseline"
    model_metrics.loc[comparisons, "p_bonferroni"] = np.minimum(
        model_metrics.loc[comparisons, "p_value"] * comparisons.sum(), 1.0
    )
    model_metrics = pd.DataFrame(metrics)
    comparisons = model_metrics["model"] != "Seat baseline"
    model_metrics.loc[comparisons, "p_bonferroni"] = np.minimum(
        model_metrics.loc[comparisons, "p_value"] * comparisons.sum(), 1.0
    )
    rules = pd.DataFrame(opening_scores)
    rules["p_bonferroni"] = np.minimum(rules["p_value"] * len(rules), 1.0)
    rules["beats_baseline"] = (rules["lift"] > 0) & (rules["p_bonferroni"] < 0.05)
    for csv_name, data in {
        "game_counts": summaries, "model_metrics": model_metrics, "classification_report": reports,
        "confusion_counts": confusions, "test_predictions": predictions,
        "opening_rule_results": rules, "cross_tier_results": transfer,
    }.items():
        pd.DataFrame(data).to_csv(output / f"{csv_name}.csv", index=False, float_format="%.6f")
    from summarize import write_report
    write_report(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/openings.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    evaluate(args.input, args.output)

