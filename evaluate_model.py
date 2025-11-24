# evaluate_model.py

from evaluate import compute_score
from debug_utils import callout

def evaluate_test(samples):
    callout("Computing final test score...")
    return compute_score(generated_samples=samples, set_name='test')
