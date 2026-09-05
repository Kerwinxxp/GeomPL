"""CPU-only subprocess tests: real smoke checks, fake model/network boundaries."""
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

GEO = r"""
import sys, types, runpy
from pathlib import Path
case, script = sys.argv[1:]
revision = 'a' * 40
torch = types.ModuleType('torch')
torch.cuda = types.SimpleNamespace(is_available=lambda: case != 'cuda', get_device_name=lambda _: 'RTX 4090')
sys.modules['torch'] = torch
hub = types.ModuleType('huggingface_hub')
hub.snapshot_download = lambda *a, **kw: str(Path('/cache/snapshots') / (('b'*40) if case == 'revision' else revision))
sys.modules['huggingface_hub'] = hub
geo = types.ModuleType('belief_elicit.georanker_belief')
values = {'direction': [[1.,2.],[1.,2.]], 'nan': [[float('nan'),0.],[1.,0.]], 'drift': [[2.,1.],[2.1,1.]], 'length': [[2.],[2.]]}.get(case, [[2.,1.],[2.,1.]])
def score(*args, **kwargs):
    if case == 'exception': raise RuntimeError('model load failed')
    assert geo.BASE.endswith(revision), 'model is not pinned to a snapshot'
    return values.pop(0)
geo.score_rewards = score
sys.modules['belief_elicit.georanker_belief'] = geo
sys.argv = [script, 'Qwen/Qwen2-VL-7B-Instruct', revision]
runpy.run_path(script, run_name='__main__')
"""

LAMA = r"""
import sys, types, runpy
from PIL import Image
case, script = sys.argv[1:]
torch = types.ModuleType('torch')
torch.cuda = types.SimpleNamespace(is_available=lambda: case != 'cuda', get_device_name=lambda _: 'RTX 4090')
torch.device = lambda x: x
sys.modules['torch'] = torch
module = types.ModuleType('simple_lama_inpainting')
class Model:
    def __init__(self, device):
        assert device == 'cuda'
    def __call__(self, image, mask):
        assert mask.getextrema() == (0,255)
        if case == 'forward': raise RuntimeError('CUDA forward failed')
        return Image.new('RGB', (1,1)) if case == 'shape' else image.copy()
module.SimpleLama = Model
sys.modules['simple_lama_inpainting'] = module
runpy.run_path(script, run_name='__main__')
"""

SAM = r"""
import sys, types, runpy, contextlib
import numpy as np
from pathlib import Path
from PIL import Image
case, script = sys.argv[1:]
revision = 'a'*40
image = Image.open('data/sample_images/261517384_292417efcc_117_60558526@N00.jpg')
torch = types.ModuleType('torch')
torch.no_grad = contextlib.nullcontext
sys.modules['torch'] = torch
hub = types.ModuleType('huggingface_hub')
hub.snapshot_download = lambda *a, **kw: '/cache/snapshots/' + ('b'*40 if case=='revision' else revision)
sys.modules['huggingface_hub'] = hub
class Tensor:
    def detach(self): return self
    def cpu(self): return self
    def numpy(self):
        array = np.zeros((image.height, image.width))
        array[0,0] = float('nan') if case=='nan' else 1
        if case=='full': array[:] = 1
        if case=='empty': array[:] = 0
        if case=='shape': array = np.ones((2,2))
        return array
class Inputs(dict):
    def to(self, device):
        assert device == 'cuda'
        return self
class Processor:
    @classmethod
    def from_pretrained(cls, path, local_files_only):
        assert Path(path).name == revision and local_files_only
        return cls()
    def __call__(self, images, text, return_tensors):
        assert text == 'Japanese banner' and return_tensors == 'pt'
        return Inputs()
    def post_process_instance_segmentation(self, output, threshold, target_sizes):
        return [{'masks': [] if case=='no-mask' else [Tensor()]}]
class Model(Processor):
    def to(self, device):
        assert device == 'cuda'
        return self
    def eval(self): return self
    def __call__(self, **kw):
        if case=='forward': raise RuntimeError('CUDA forward failure')
        return object()
module = types.ModuleType('transformers')
module.Sam3Model, module.Sam3Processor = Model, Processor
sys.modules['transformers'] = module
sys.argv = [script, 'facebook/sam3', revision]
runpy.run_path(script, run_name='__main__')
"""

class SmokeTests(unittest.TestCase):
    def run_case(self, code, script, case, succeeds=False):
        result = subprocess.run([sys.executable, '-c', code, case, str(HERE/script)], cwd=ROOT, capture_output=True, text=True)
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('PASS', result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertNotIn('PASS', result.stdout)

    def test_georanker_checks_and_process_exit(self):
        for case in ['ok','cuda','revision','direction','nan','drift','length','exception']:
            with self.subTest(case=case): self.run_case(GEO, 'Strict-GeoRanker-Smoke.py', case, case=='ok')

    def test_lama_checks_and_process_exit(self):
        for case in ['ok','cuda','forward','shape']:
            with self.subTest(case=case): self.run_case(LAMA, 'Strict-LaMa-Smoke.py', case, case=='ok')

    def test_sam_checks_and_process_exit(self):
        for case in ['ok','revision','no-mask','empty','full','shape','nan','forward']:
            with self.subTest(case=case): self.run_case(SAM, 'Strict-Sam3-Smoke.py', case, case=='ok')

    def test_prefetch_pins_offline_reference(self):
        import runpy
        import tempfile
        import types
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            revision = 'a'*40
            root = Path(directory)
            snapshot = root/'snapshots'/revision
            snapshot.mkdir(parents=True)
            def download(model_id, revision, local_files_only=False):
                if local_files_only:
                    return str(root/'snapshots'/(root/'refs'/revision).read_text())
                return str(snapshot)
            hub = types.ModuleType('huggingface_hub')
            hub.snapshot_download = download
            with patch.dict(sys.modules, {'huggingface_hub':hub}), patch.object(sys, 'argv', ['prefetch', 'model', revision]):
                runpy.run_path(str(HERE/'Prefetch-Pinned-Model.py'), run_name='__main__')
            self.assertEqual((root/'refs'/'main').read_text(), revision)
            self.assertEqual(list((root/'refs').iterdir()), [root/'refs'/'main'])

if __name__ == '__main__': unittest.main()
