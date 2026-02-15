"""
Evaluate BERT phishing model accuracy on a labeled set of phishing vs legitimate messages.
Run from Hackathon folder: python -m src.eval_phishing_accuracy
"""

from __future__ import annotations

# Label: 1 = phishing, 0 = legitimate
PHISHING_EXAMPLES = [
    "Dear Customer, Your account has been suspended. You must verify your identity within 24 hours or lose access permanently. Click here to verify: https://secure-bank-verify.com/login.",
    "CONGRATULATIONS! You've been selected to receive 5 Bitcoin. To claim, send 0.1 BTC to the address below for verification. Offer expires in 48 hours.",
    "Hi, this is IT Support. We're seeing suspicious activity on your account. To secure it, please reply with your current password and the 2FA code you just received. We'll fix it from our end.",
    "Your package is at the depot. A delivery fee of $2.99 is required. Pay within 12 hours or the package will be returned to sender. Click here to pay.",
    "You've won our exclusive giveaway! To release your prize we need a small processing fee ($50). This is standard policy. Send payment within 24 hours to secure your winnings.",
]

LEGITIMATE_EXAMPLES = [
    "The Federal Reserve announced a 0.25% interest rate hike today. Markets reacted with modest gains. Economists had widely expected the move.",
    "According to the health ministry, vaccination rates rose 5% in the last quarter. The data was published in the weekly epidemiological bulletin.",
    "The company reported revenue of $2.1B, in line with analyst estimates. CEO Jane Smith commented on the earnings call.",
    "A spokesperson for the agency confirmed the policy change. The full report is available on the agency's official website.",
    "According to the latest survey, 62% of respondents said they support the measure. The margin of error is 3%.",
    "Hi, your order #8842 has shipped. Track it at our website using the link in your account. No reply needed.",
    "Reminder: Your next team standup is Tuesday at 10am. Agenda and Zoom link are in the calendar invite.",
]


def main() -> None:
    from .local_model import is_available, predict_phishing

    if not is_available():
        print("Local phishing model is not available (transformers/torch or model load failed).")
        return

    # (text, true_label): 1 = phishing, 0 = legitimate
    samples: list[tuple[str, int]] = []
    for t in PHISHING_EXAMPLES:
        samples.append((t, 1))
    for t in LEGITIMATE_EXAMPLES:
        samples.append((t, 0))

    preds: list[int] = []
    confidences: list[float] = []
    for text, _ in samples:
        verdict, conf = predict_phishing(text)
        # Supported = phishing (1), Refuted = legitimate (0), Unknown = treat as wrong
        if verdict == "Supported":
            preds.append(1)
        elif verdict == "Refuted":
            preds.append(0)
        else:
            preds.append(-1)  # unknown
        confidences.append(conf)

    labels = [s[1] for s in samples]
    n = len(labels)
    correct = sum(1 for i in range(n) if preds[i] == labels[i])
    unknown = sum(1 for p in preds if p == -1)
    accuracy = correct / n if n else 0.0

    # Precision/recall/F1 for phishing class (label 1)
    tp = sum(1 for i in range(n) if labels[i] == 1 and preds[i] == 1)
    fp = sum(1 for i in range(n) if labels[i] == 0 and preds[i] == 1)
    fn = sum(1 for i in range(n) if labels[i] == 1 and preds[i] != 1)
    precision_phish = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_phish = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_phish = 2 * precision_phish * recall_phish / (precision_phish + recall_phish) if (precision_phish + recall_phish) > 0 else 0.0

    print("=" * 60)
    print("Phishing model accuracy evaluation")
    print("=" * 60)
    print(f"Total samples:  {n}  (phishing: {sum(labels)}, legitimate: {n - sum(labels)})")
    print(f"Correct:       {correct}")
    if unknown:
        print(f"Unknown:        {unknown}")
    print(f"Accuracy:       {accuracy:.2%}")
    print()
    print("Phishing class (Supported = phishing):")
    print(f"  Precision:   {precision_phish:.2%}")
    print(f"  Recall:      {recall_phish:.2%}")
    print(f"  F1:          {f1_phish:.2%}")
    print("=" * 60)

    # Per-sample breakdown (optional, compact)
    print("\nPer-sample (first 5 phishing, first 3 legitimate):")
    for i, (text, label) in enumerate(samples[:8]):
        pred = preds[i]
        conf = confidences[i]
        ok = "OK" if pred == label else "FAIL"
        pred_str = "phishing" if pred == 1 else "legit" if pred == 0 else "unknown"
        true_str = "phishing" if label == 1 else "legit"
        short = (text[:55] + "...") if len(text) > 55 else text
        print(f"  {ok} [{true_str}] pred={pred_str} conf={conf:.2f}  {short}")


if __name__ == "__main__":
    main()
