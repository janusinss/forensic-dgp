"""Adapt the retained namespace-safe importer to the exact new sealed PIN."""
import ast
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'scripts/import_cctv_dgp_live_conditioning_v1_r1_return.py'
    prep=json.loads((ROOT/'outputs/cctv_dgp_conditioning_learning_v1_r2_preparation.json').read_text())
    text=source.read_text(encoding='utf-8')
    edits=[('live_conditioning_v1_r1','conditioning_learning_v1_r2'),
        ('cctv-dgp-live-conditioning-v1-r1','cctv-dgp-conditioning-learning-v1-r2'),
        ('754591c33b56e97d0b9f63336a3219ca12ddb316ddf6dcc2d883a34fda3c2d5f',prep['protocol_sha256']),
        ("assert exported['complete'] and exported['diagnostic_success_not_implied']","assert exported['complete'] and exported['training_success_not_implied']"),
        ("assert exported['optimizer_updates'] == 0","assert exported['accepted_updates'] in [None,0,1,2]"),
        ('len(members) <= 900','len(members) <= 2400'),
        ("assert manifest['optimizer_updates'] == 0","assert manifest['accepted_updates'] == exported['accepted_updates']"),
        ("complete_diagnostic_reported=manifest['complete_diagnostic']","complete_pilot_reported=manifest['complete_pilot'], accepted_updates_reported=manifest['accepted_updates']"),
        ("'complete_diagnostic_reported'","'complete_pilot_reported'")]
    for before,after in edits:
        assert before in text,before
        text=text.replace(before,after)
    ast.parse(text,feature_version=(3,10))
    target=ROOT/'scripts/import_cctv_dgp_conditioning_learning_v1_r2_return.py'
    with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    print(dict(prospective_safe_importer_saved=True,sealed_protocol_sha256=prep['protocol_sha256']))


if __name__=='__main__':main()
