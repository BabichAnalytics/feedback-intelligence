"""Create reproducible fictional data; test phrase families never appear in training."""
import csv
from pathlib import Path

TRAIN = {
'Pricing': ['The price of the subscription', 'The cost of our monthly plan', 'The fee on my invoice', 'The charge for the package'],
'Service': ['The customer support agent', 'The help desk representative', 'The service team response', 'The staff assistance'],
'Quality': ['The product build quality', 'The material durability', 'The item craftsmanship', 'The equipment reliability'],
'Delivery': ['The parcel delivery time', 'The shipping schedule', 'The package arrival', 'The courier shipment']}
TEST = {
'Pricing': ['Our renewal pricing', 'The extra billing fee'],
'Service': ['My experience with customer care', 'The response from the help team'],
'Quality': ['The performance of this product', 'The reliability of the replacement item'],
'Delivery': ['The time taken to ship my order', 'The delivery of our replacement parcel']}
ENDINGS = {
'Positive': ['was excellent and exceeded expectations.', 'was great and made me very happy.', 'was wonderful and I am satisfied.', 'was fantastic; I would recommend it.'],
'Neutral': ['was as described in the information provided.', 'was what I expected with nothing unusual.', 'was average and neither good nor bad.', 'was standard for this type of purchase.'],
'Negative': ['was terrible and left me disappointed.', 'was bad and caused frustrating problems.', 'was awful and I am unhappy.', 'was unacceptable; I would not recommend it.']}
HELDOUT = {'Positive': ['was good and I appreciate it.', 'was impressive and helpful.'],
'Neutral': ['was ordinary with no strong opinion.', 'was typical and met the basic description.'],
'Negative': ['was disappointing and needs improvement.', 'was poor and I regret the purchase.']}


def generate():
    rows = []
    for split, starts, ends in [('train', TRAIN, ENDINGS), ('test', TEST, HELDOUT)]:
        for topic, phrases in starts.items():
            for sentiment, endings in ends.items():
                for phrase in phrases:
                    for ending in endings:
                        number = len(rows) + 1
                        rows.append(dict(id=f'FB-{number:04}', date=f'2026-{1+(number-1)%9:02}-{1+(number-1)%27:02}',
                                         text=f'{phrase} {ending}', topic=topic, sentiment=sentiment, split=split))
    with (Path(__file__).parent / 'sample_feedback.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'date', 'text', 'topic', 'sentiment', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Created {len(rows)} fictional feedback records.')

if __name__ == '__main__':
    generate()
