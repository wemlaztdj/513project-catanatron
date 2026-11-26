"""Write the human-readable report from the generated experiment tables."""
import pandas as pd


def percent(value):
    return f"{value:.1%}"


def points(value):
    return f"{100 * value:+.1f} pp"


def scenario(row):
    return f"{row.players} players, {row.tier.title()}"


def write_report(folder):
    counts = pd.read_csv(folder / "game_counts.csv")
    models = pd.read_csv(folder / "model_metrics.csv")
    openings = pd.read_csv(folder / "opening_rule_results.csv")
    transfer = pd.read_csv(folder / "cross_tier_results.csv")
    baseline_rows = models[models.model == "Seat baseline"]
    classifiers = models[models.model != "Seat baseline"]
    best = classifiers.sort_values(["accuracy", "macro_f1"], ascending=False).drop_duplicates(
        ["players", "tier"]
    )
    supported = classifiers[(classifiers.lift > 0) & (classifiers.p_bonferroni < 0.05)]
    universal = openings.groupby("rule")["beats_baseline"].all()
    universal_rules = list(universal[universal].index)
    title = "Can we predict the winner before the first roll?"
    lines = [
        "# Catan winner prediction", "", f"## {title}", "",
        f"Catanatron ran **{counts.attempts.sum():,} games**, with "
        f"**{counts.attempts.min():,} attempts in each scenario**. "
        f"The engine returned a winner in **{counts.completed.sum():,} games**. "
        "The predictors describe the visible opening after every seat placed two settlements "
        "and two roads, immediately before the first roll. The target is the eventual winning seat.",
        "",
    ]
    if supported.empty:
        lines.append(
            "Some models can score above the seat-only baseline in an individual test sample. "
            "The adjusted comparisons provide no statistically reliable gain for any of the five classifiers."
        )
    else:
        lines.append(f"**{len(supported)} of {len(classifiers)} classifier comparisons** showed a "
                     "statistically reliable improvement over the seat-only baseline after correction "
                     "for multiple comparisons.")
    lines += [
        "", "### Best observed classifier per scenario", "",
        "| Scenario | Test games | Best classifier | Accuracy | Seat baseline | Lift | Macro F1 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for baseline in baseline_rows.itertuples():
        row = best[(best.players == baseline.players) & (best.tier == baseline.tier)].iloc[0]
        lines.append(f"| {scenario(baseline)} | {row.test_games} | {row.model} | "
                     f"{percent(row.accuracy)} | {percent(baseline.accuracy)} | "
                     f"{points(row.lift)} | {row.macro_f1:.3f} |")
    lines += [
        "", "The best observed model was selected descriptively from the same held-out results "
        "shown here. Uniform guesses have expected accuracy of 50% with two seats and 25% "
        "with four. The seat-only baseline always predicts the seat with the most training wins.",
        "", "### Test-set accuracy and classification reports", "",
        "| Scenario | Model | Accuracy | Macro precision | Macro recall | Macro F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in models.itertuples():
        lines.append(f"| {scenario(row)} | {row.model} | {percent(row.accuracy)} | "
                     f"{row.macro_precision:.3f} | {row.macro_recall:.3f} | {row.macro_f1:.3f} |")
    lines += [
        "", "The [complete classification reports](classification_report.csv) include precision, "
        "recall, F1, and support for each winning seat, plus macro and weighted averages. "
        "A metric with a zero denominator is recorded as zero. The [model metrics](model_metrics.csv) "
        "and [test predictions](test_predictions.csv) provide the underlying comparisons.",
        "", "### Confusion matrices", "",
        "Rows show the actual winning seat, columns show the predicted winning seat, "
        "and diagonal cells count correct predictions. The color scale is shared within each scenario.",
    ]
    for row in counts.itertuples():
        chart = f"confusion_{row.players}p_{row.tier}.png"
        lines += ["", f"![Confusion matrices for {scenario(row)}]({chart})"]
    lines += [
        "", "## Is there a universally good starting position?", "",
        "A starting-position rule selects the seat with the highest value of a fixed opening "
        "feature. For this experiment, the rule must outperform the seat-only baseline in all four "
        "scenarios after correcting for 28 comparisons. This is an operational definition of "
        "robustness across the tested settings.",
        "",
    ]
    if universal_rules:
        lines.append(f"Rules meeting that criterion: {', '.join(universal_rules)}.")
    else:
        lines.append("**None of the seven rules met that criterion.**")
    balanced = openings[openings.rule == "balanced"]
    lines.append(
        f"The balanced score improved on the baseline in {sum(balanced.lift > 0)} of {len(balanced)} "
        f"test samples and cleared the adjusted significance threshold in {sum(balanced.beats_baseline)}. "
        "This experiment covers two post-setup bot policies and one opening-placement policy."
    )
    lines += ["", "| Opening rule | 2P Basic | 2P Advanced | 4P Basic | 4P Advanced |",
              "|---|---:|---:|---:|---:|"]
    for rule, group in openings.groupby("rule", sort=False):
        ordered = group.set_index(["players", "tier"]).reindex(
            [(2, "basic"), (2, "advanced"), (4, "basic"), (4, "advanced")]
        )
        entries = [points(row.lift) + (" *" if row.beats_baseline else "") for row in ordered.itertuples()]
        lines.append(f"| {rule.replace('_', ' ').title()} | " + " | ".join(entries) + " |")
    lines += [
        "", "Values are accuracy differences from the seat-only baseline in percentage points. "
        "An asterisk marks a positive difference with adjusted p < 0.05. "
        "See [opening rule results](opening_rule_results.csv) for paired intervals and adjusted p-values.",
        "", "### Cross-tier transfer", "",
        "For each player count, a Random Forest trained on one tier predicts winners in "
        "the other tier on seeds excluded from training. This measures whether learned relationships "
        "transfer between the two policies when the opening setup process is held constant.",
        "", "| Players | Train tier | Test tier | Test games | Accuracy | Seat baseline | Lift |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in transfer.itertuples():
        lines.append(f"| {row.players} | {row.trained_on.title()} | {row.tested_on.title()} | {row.test_games} | "
                     f"{percent(row.accuracy)} | {percent(row.baseline_accuracy)} | {points(row.lift)} |")
    lines += [
        "", "[Cross-tier results](cross_tier_results.csv) include paired intervals. Transfer is descriptive.",
        "", "## Experimental design", "",
        "- **Basic** uses Catanatron Weighted Random. **Advanced** uses Catanatron Value Function, "
        "a greedy heuristic with 5% random exploration at decisions with multiple choices.",
        "- The two tiers share the same starting positions within each player count and seed. A common "
        "opening policy favors productive, diverse intersections probabilistically and picks a legal "
        "starting road at random. The match compares post-setup bot policies and does not estimate "
        "the effect of differences in their own placement choices.",
        "- Catanatron uses its default board and ten victory points. The two-player scenario uses "
        "the engine's standard mechanics with two seats. The source population is simulated games.",
        "- The experiment reserves the same 25% of unique seeds in all scenarios. Every completed "
        "game contributes one example to its own scenario model. Games reaching the turn limit "
        "without a winner are counted and excluded from classification. See [game counts](game_counts.csv).",
        "- Hyperparameters were fixed before the full evaluation. Models use ordered seat-level "
        "opening features, with identifiers, node IDs, the RNG seed, and post-roll outcomes excluded. "
        "The KNN and ANN scalers are fit on training games.",
        "- Differences are paired within game. Lift intervals are 95% game-level bootstrap "
        "intervals. Tests use exact McNemar p-values and Bonferroni correction separately for "
        "the 20 classifier comparisons and the 28 opening-rule comparisons.",
        "- A high-scoring position can still lose. Opening features and winner associations "
        "are observational within this setup policy. The project does not estimate causal effects "
        "of moving a settlement or guarantee performance against humans.",
        "", "See the [project README](../README.md) for the run commands and feature definitions.", "",
    ]
    (folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

