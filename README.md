# Feedback Intelligence
A local AI analytics portfolio project by **Babich Analytics**. Import labeled customer feedback, predict sentiment and topic with two trained text classifiers, inspect trends, and evaluate errors on a separate test set.

## Run on Windows
Requires Python 3.10+; no third-party packages, GPU, model download, or paid API.

Download/extract this repository, open Command Prompt in its folder, then run:
```cmd
python app.py
```
Open **http://127.0.0.1:8003**. Keep the command window open. Ctrl+C stops the server.

## Features
- CSV validation; topic and sentiment prediction using multinomial Naive Bayes.
- Search, topic/sentiment filters, review flags, monthly negative counts, JSON export.
- Held-out accuracy, macro F1, class precision/recall/F1, and confusion matrices.
- Local-only browser dashboard; feedback stays on your machine.

## Dataset and evaluation
`sample_feedback.csv` contains **240 fictional English-language records**: 192 training and 48 test. `make_sample.py` reproduces it. Separate phrase families are assigned to each split and exact duplicate texts are rejected. Only training rows influence the model. Reference labels and split membership are supplied in the CSV, not inferred by the app.

This is a transparent educational baseline, not a production-ready sentiment model. Both splits are formulaic synthetic data; shared vocabulary and sentiment patterns can inflate results even when phrase families differ. Real customer feedback needs independently labeled, deduplicated data, preferably split by customer/source and time. Evaluate mixed sentiment, sarcasm, spelling errors, class imbalance, and multiple topics. A small synthetic test set cannot justify automatic business decisions.

The dashboard displays **uncalibrated model scores**, not verified confidence or a probability of being correct. A score below 0.80 for either classifier flags human review; this threshold is illustrative and must be validated on representative data. Above-threshold predictions can still be wrong. The app does not train on human corrections or generate replies.

## Bring your own labeled CSV
Columns: `id,date,text,topic,sentiment,split`.
- Date: YYYY-MM-DD; unique IDs and distinct nonempty text.
- Topic: Pricing, Service, Quality, Delivery (one label per record).
- Sentiment: Positive, Neutral, Negative.
- Split: train or test; training must contain all classes.
- Limit: 10,000 records, 10,000 characters per text.

```cmd
python app.py --csv your_feedback.csv
python app.py --threshold 0.9
python app.py --csv your_feedback.csv --report results.json
python -m unittest -v
```
Use appropriately authorized, anonymized data. Keep real feedback outside the repository. The provided `.gitignore` excludes common private-data directories and exports; it cannot identify every sensitive file.

## Explain this in an interview
The business question is which issues recur and whether automated classification is dependable enough to help analysts triage feedback. Explain label creation, the training/test separation, model errors, and why a review threshold requires validation. AI assisted the implementation; run, inspect, modify, and explain the code before presenting it as evidence of your skills.

## Next milestones
Real labeled evaluation data; saved human corrections; multi-label topics; score calibration; comparison against an independently evaluated local transformer baseline.
