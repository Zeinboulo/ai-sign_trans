import os
import subprocess
import sys
import numpy as np


ROOT = os.path.dirname(os.path.dirname(__file__))
GEN = os.path.join(ROOT, "scripts", "generate_sample.py")
SAMPLES_DIR = os.path.join(ROOT, "samples")
KEYPOINTS = os.path.join(SAMPLES_DIR, "sample_keypoints.npy")
MANIFEST = os.path.join(ROOT, "splits", "sample_manifest.csv")


def test_generate_and_load_sample(tmp_path):
    # Run the generator
    subprocess.check_call([sys.executable, GEN])

    assert os.path.exists(KEYPOINTS), "Sample keypoints file was not created"
    assert os.path.exists(MANIFEST), "Sample manifest was not created"

    arr = np.load(KEYPOINTS)
    assert arr.ndim == 2
    assert arr.shape[0] > 0 and arr.shape[1] > 0
