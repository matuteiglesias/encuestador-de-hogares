import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'science/2026-09-11/build_predictive_welfare_release.py'
spec=importlib.util.spec_from_file_location('wp04_predictive_builder',SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class WP04PredictiveRunProofTests(unittest.TestCase):
    def test_genuine_release_manifest_yields_bounded_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'household-welfare-predictive-2024-q3-abc'
            p.mkdir()
            manifest={'release_id':p.name,'artifact_type':module.ARTIFACT_TYPE,
                      'status':'research_commissioned_with_transport_caveats'}
            (p/'manifest.json').write_text(json.dumps(manifest))
            before=module.sha256(p/'manifest.json')
            path=module.record_successful_release_run(p,'2026-10-25T01:00:00Z','2026-10-25T01:02:00Z')
            proof=json.loads(path.read_text())
            self.assertEqual(proof['manifest_sha256'],before)
            self.assertEqual(module.sha256(p/'manifest.json'),before)
            self.assertEqual(proof['run_id'],p.name)
            self.assertNotIn('dataset_rows',path.read_text())
            with self.assertRaises(FileExistsError):
                module.record_successful_release_run(p,'2026-10-25T01:00:00Z','2026-10-25T01:02:00Z')
    def test_wrong_release_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'one'
            p.mkdir()
            (p/'manifest.json').write_text(json.dumps({'release_id':'two','artifact_type':module.ARTIFACT_TYPE}))
            with self.assertRaises(ValueError):
                module.record_successful_release_run(p,'2026-10-25T01:00:00Z','2026-10-25T01:02:00Z')
