import tempfile
import unittest
from pathlib import Path
from app import ROOT, NaiveBayes, analyze, evaluate, load_rows

class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.rows = load_rows(ROOT / 'sample_feedback.csv')
    def test_no_train_test_text_overlap(self):
        train = {r['text'].lower() for r in self.rows if r['split']=='train'}
        test = {r['text'].lower() for r in self.rows if r['split']=='test'}
        self.assertFalse(train & test)
        self.assertEqual(len(test),48)
    def test_training_ignores_test_labels(self):
        first = analyze(self.rows)
        changed = [dict(r) for r in self.rows]
        for r in changed:
            if r['split']=='test': r['sentiment']='Negative'
        second = analyze(changed)
        self.assertEqual([r['predicted_sentiment'] for r in first['records']],
                         [r['predicted_sentiment'] for r in second['records']])
    def test_metrics_known_case(self):
        rows=[{'topic':'a','predicted_topic':'a'},{'topic':'a','predicted_topic':'b'}]
        m=evaluate(rows,'topic',['a','b'])
        self.assertEqual(m['accuracy'],0.5)
        self.assertEqual(m['confusion_matrix']['a']['b'],1)
        self.assertAlmostEqual(m['macro_f1'],1/3)
    def test_unknown_text_requires_review(self):
        result=analyze(self.rows+[dict(id='x',date='2026-10-01',text='zzzzz qqqqq',topic='Pricing',sentiment='Neutral',split='test')])
        self.assertTrue(result['records'][-1]['review'])
    def test_invalid_csv_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.csv';p.write_text('text\nhello\n')
            with self.assertRaises(ValueError):load_rows(p)
    def test_duplicate_text_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.csv';text=(ROOT/'sample_feedback.csv').read_text()
            line=text.splitlines()[1].replace('FB-0001','FB-9999',1)
            p.write_text(text+line+'\n')
            with self.assertRaises(ValueError):load_rows(p)
    def test_report_totals_and_scores(self):
        report=analyze(self.rows)
        self.assertEqual(report['train_count'],192)
        for metric in report['evaluation'].values():
            self.assertEqual(metric['total'],48)
            self.assertEqual(sum(sum(row.values()) for row in metric['confusion_matrix'].values()),48)
        for r in report['records']:
            self.assertTrue(0<=r['sentiment_score']<=1)
            self.assertTrue(0<=r['topic_score']<=1)

if __name__=='__main__':unittest.main()
