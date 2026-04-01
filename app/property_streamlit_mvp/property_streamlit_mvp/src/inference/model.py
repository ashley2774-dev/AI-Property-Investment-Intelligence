from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuleBasedModel:
    classes_: tuple[str, str, str] = ("reject", "review", "recommended")

    def predict_proba(self, X):
        rows = []
        for _, row in X.iterrows():
            score = 0.0
            score += 2.5 if row.get("dscr", 0) >= 1.25 else 0.0
            score += 2.0 if row.get("monthly_cash_flow", 0) > 0 else -2.5
            score += 1.5 if row.get("roi", 0) >= 0.08 else 0.0
            score += 1.0 if row.get("gross_yield", 0) >= 0.09 else 0.0
            score += 1.0 if row.get("bond_to_rent", 99) <= 0.75 else -1.0
            score += 0.5 if row.get("opex_to_rent", 99) <= 0.35 else -0.5

            if score >= 4.0:
                probs = [0.08, 0.20, 0.72]
            elif score >= 1.5:
                probs = [0.18, 0.62, 0.20]
            else:
                probs = [0.76, 0.18, 0.06]
            rows.append(probs)
        return rows

    def predict(self, X):
        probas = self.predict_proba(X)
        labels = []
        for p in probas:
            labels.append(self.classes_[max(range(len(p)), key=lambda i: p[i])])
        return labels
