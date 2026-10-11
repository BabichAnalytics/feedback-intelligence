"""Local feedback classification and evaluation. Python standard library only."""
import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
TOPICS = ('Pricing', 'Service', 'Quality', 'Delivery')
SENTIMENTS = ('Positive', 'Neutral', 'Negative')
FIELDS = ('id', 'date', 'text', 'topic', 'sentiment', 'split')


def tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


class NaiveBayes:
    """Laplace-smoothed multinomial NB; posterior scores are not calibrated."""
    def fit(self, rows, label):
        self.classes = sorted({r[label] for r in rows})
        if len(self.classes) < 2:
            raise ValueError('Training data needs at least two classes.')
        self.docs = Counter(r[label] for r in rows)
        self.words = {c: Counter() for c in self.classes}
        self.vocab = set()
        for row in rows:
            words = tokens(row['text'])
            self.words[row[label]].update(words)
            self.vocab.update(words)
        if not self.vocab:
            raise ValueError('Training text contains no words.')
        self.totals = {c: sum(self.words[c].values()) for c in self.classes}
        return self

    def predict(self, text):
        counts = Counter(t for t in tokens(text) if t in self.vocab)
        logs = {}
        for c in self.classes:
            logs[c] = math.log(self.docs[c] / sum(self.docs.values()))
            for word, count in counts.items():
                logs[c] += count * math.log((self.words[c][word] + 1) /
                                           (self.totals[c] + len(self.vocab)))
        peak = max(logs.values())
        weights = {c: math.exp(s - peak) for c, s in logs.items()}
        total = sum(weights.values())
        winner = max(weights, key=weights.get)
        return winner, weights[winner] / total, bool(counts)


def load_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('CSV must contain: ' + ', '.join(FIELDS))
        rows = list(reader)
    if not rows:
        raise ValueError('CSV is empty.')
    if len(rows) > 10000:
        raise ValueError('Demo limit: 10,000 rows.')
    ids, texts = set(), set()
    for row in rows:
        if any(row.get(f) is None for f in FIELDS):
            raise ValueError('Incomplete CSV row.')
        row['text'] = row['text'].strip()
        if not row['id'].strip() or row['id'] in ids:
            raise ValueError('Each ID must be nonempty and unique.')
        if not row['text'] or len(row['text']) > 10000:
            raise ValueError('Text must contain 1–10,000 characters.')
        normalized = ' '.join(tokens(row['text']))
        if normalized in texts:
            raise ValueError('Duplicate text found; remove duplicates before splitting.')
        if row['topic'] not in TOPICS or row['sentiment'] not in SENTIMENTS:
            raise ValueError('Unrecognized topic or sentiment label.')
        if row['split'] not in ('train', 'test'):
            raise ValueError('Split must be train or test.')
        date.fromisoformat(row['date'])
        ids.add(row['id'])
        texts.add(normalized)
    if not any(r['split'] == 'train' for r in rows) or not any(r['split'] == 'test' for r in rows):
        raise ValueError('Both train and test rows are required.')
    for field, classes in [('topic', TOPICS), ('sentiment', SENTIMENTS)]:
        if {r[field] for r in rows if r['split'] == 'train'} != set(classes):
            raise ValueError('Training rows must cover every ' + field + ' class.')
    return rows


def evaluate(records, field, labels):
    predicted = 'predicted_' + field
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for row in records:
        matrix[row[field]][row[predicted]] += 1
    correct = sum(matrix[c][c] for c in labels)
    per_class = []
    for c in labels:
        tp = matrix[c][c]
        support = sum(matrix[c].values())
        predicted_count = sum(matrix[a][c] for a in labels)
        precision = tp / predicted_count if predicted_count else 0
        recall = tp / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class.append(dict(label=c, precision=precision, recall=recall, f1=f1, support=support))
    return dict(accuracy=correct / len(records), correct=correct, total=len(records),
                macro_f1=sum(r['f1'] for r in per_class) / len(labels),
                confusion_matrix=matrix, per_class=per_class)


def analyze(rows, threshold=0.8):
    train = [r for r in rows if r['split'] == 'train']
    models = {f: NaiveBayes().fit(train, f) for f in ('topic', 'sentiment')}
    results = []
    for row in rows:
        result = dict(row)
        known = True
        for field, model in models.items():
            label, score, has_words = model.predict(row['text'])
            result['predicted_' + field] = label
            result[field + '_score'] = score
            known = known and has_words
        result['review'] = not known or min(result['topic_score'], result['sentiment_score']) < threshold
        results.append(result)
    test = [r for r in results if r['split'] == 'test']
    return dict(records=results, train_count=len(train), test_count=len(test), threshold=threshold,
                evaluation={f: evaluate(test, f, labels) for f, labels in
                            [('topic', TOPICS), ('sentiment', SENTIMENTS)]},
                provenance='Fictional English-language demonstration data. Scores are uncalibrated model outputs, not probabilities of correctness.')


class Handler(BaseHTTPRequestHandler):
    payload = None

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/':
            body, mime = (ROOT / 'dashboard.html').read_bytes(), 'text/html; charset=utf-8'
        elif path == '/api/report':
            body, mime = json.dumps(self.payload, ensure_ascii=False).encode(), 'application/json; charset=utf-8'
        elif path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
            return
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'sample_feedback.csv')
    parser.add_argument('--port', type=int, default=8003)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--report', type=Path, help='Write JSON report and exit instead of starting server')
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error('threshold must be greater than 0 and at most 1')
    try:
        payload = analyze(load_rows(args.csv), args.threshold)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Unable to load feedback: {error}\n')
    if args.report:
        args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        print('Report saved:', args.report)
        return
    Handler.payload = payload
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Feedback Intelligence: http://127.0.0.1:{args.port}', flush=True)
    print('Press Ctrl+C to stop. Local demonstration; no paid APIs.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
