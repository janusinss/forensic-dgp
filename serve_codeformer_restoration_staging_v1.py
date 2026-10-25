"""Opt-in loopback U96 staging launcher; existing app/defaults stay unchanged.

--verify-only rehashes reviewed sources/model files without socket or worker.
The caller supplies the independently reviewed staging manifest; this launcher
never manufactures source approval and never starts training or a historical run.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import socket

import research_brush_process_restoration_v2 as L

ROOT = L.ROOT
ROUTE = '/restoration-workflow'
SOURCES = ('codeformer_restoration_workflow_v1.py', 'codeformer_restoration_web_v1.py',
           'templates/codeformer_restoration_workflow_v1.html', 'static/codeformer_restoration_workflow_v1.js',
           'serve_codeformer_restoration_staging_v1.py', 'codeformer_app_delivery_v1.py',
           'research/protected_surroundings_brush_v1.py')
FORMAT = 'opt-in-U96-restoration-staging-source-v1'
SCOPE = {'deployment': 'opt_in_U96_staging', 'restoration_selection': 'incumbent_U96',
         'preparation': 'optional-reviewed-manual-Protect-Apply', 'automatic_preparation': False,
         'completion_development': False, 'model_qualified': False, 'main_app_default_changed': False,
         'training_optimizer_VM_calls': 0, 'maximum_forwards_per_request': 1,
         'worker_request_deadline_seconds': 360, 'minimum_available_RAM_bytes': 2684354560}


def validate_staging_contract(value, expected_old, bound, read):
    """Pure injected manifest validator; no socket, worker or model import."""
    L.B.need(type(value) is dict and value.get('format') == FORMAT and value.get('complete') is True
             and value.get('scope') == SCOPE and all(type(value['scope'][key]) is type(item) for key, item in SCOPE.items())
             and value.get('old_manifests') == expected_old,
             'Exact opt-in staging scope and unchanged three-manifest lineage required.')
    rows = value.get('sources')
    L.B.need(type(rows) is list and len(rows) == len(SOURCES)
             and {row['path'] for row in rows} == set(SOURCES), 'Exact seven-source staging namespace required.')
    for row in rows:
        bound(row, 1024 ** 2)
    for row in expected_old.values():
        bound(row, 1024 ** 2)
    review = read(value['source_review'])
    L.B.need(review.get('complete') is True and review.get('source_review_pass') is True
             and review.get('sources') == rows and review.get('old_manifests') == expected_old
             and review.get('scope') == SCOPE
             and all(type(review['scope'][key]) is type(item) for key, item in SCOPE.items()),
             'Independent exact current staging source review required.')
    return {'sources': copy.deepcopy(rows), 'source_review': copy.deepcopy(value['source_review']),
            'old_manifests': copy.deepcopy(expected_old), 'scope': copy.deepcopy(SCOPE)}


class StagingSources:
    def __init__(self, path, expected_sha, lifecycle):
        self.record = L.B.P.pin(path, 1024 ** 2)
        L.B.need(self.record['sha256'] == expected_sha, 'Exact staging manifest SHA required.')
        self.old = {'parent': lifecycle['parent_manifest'], 'brush': lifecycle['brush_manifest'],
                    'lifecycle': lifecycle['manifest']}
        self.initial = None
        identity = self.check()
        self.initial = copy.deepcopy(identity)

    def check(self):
        lifecycle = L.source_check()
        L.B.need(self.old == {'parent': lifecycle['parent_manifest'], 'brush': lifecycle['brush_manifest'],
                             'lifecycle': lifecycle['manifest']}, 'Configured process lineage changed.')
        value = L.B.P.read(self.record, 1024 ** 2)
        contract = validate_staging_contract(value, self.old, L.B.P.bound, L.B.P.read)
        reviewed = L.B.P.manifest()  # File hashes only; no model load or neural call.
        models = copy.deepcopy(reviewed['models'])
        identity = {'staging_manifest': copy.deepcopy(self.record), **contract, 'models': models,
                    'process_lifecycle': 'one-disposable-worker-per-request-v2'}
        L.B.need(self.initial is None or identity == self.initial, 'Staging source/model identity changed; restart refused.')
        return identity


def configure(args):
    lifecycle = L.configure_sources(args.lifecycle_manifest, args.lifecycle_manifest_sha,
        args.manifest, args.manifest_sha, args.source_manifest, args.source_manifest_sha)
    return StagingSources(args.staging_manifest, args.staging_manifest_sha, lifecycle)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'source-manifest', 'lifecycle-manifest', 'staging-manifest'):
        result.add_argument('--' + name, type=Path, required=True)
        result.add_argument('--' + name + '-sha', required=True)
    result.add_argument('--receipt', type=Path)
    result.add_argument('--port', type=int, default=0)
    result.add_argument('--verify-only', action='store_true')
    return result


def fresh_receipt(path):
    L.B.need(path is not None, 'Actual startup requires a fresh scratch JSON receipt.')
    target = L.ordinary(path)
    L.B.need(target.is_relative_to(ROOT / 'scratch') and target.suffix == '.json' and not target.exists(),
             'Fresh workspace scratch JSON startup receipt required.')
    return target


def main():
    args = parser().parse_args()
    L.B.need(type(args.port) is int and 0 <= args.port <= 65535, 'Use port0 or valid loopback port.')
    sources = configure(args)
    identity = sources.check()
    if args.verify_only:
        print(json.dumps({'complete': True, 'staging_source_model_verification_pass': True,
            'source_identity': identity, 'socket_server_worker_neural_training_VM_calls': 0,
            'scope': SCOPE}), flush=True)
        return
    receipt = fresh_receipt(args.receipt)
    # Reviewed modules are imported only after their current source pins pass.
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, RedirectResponse, Response
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    from codeformer_restoration_workflow_v1 import RestorationController
    from codeformer_restoration_web_v1 import create_router
    worker = L.ProcessSelectedRestorer()
    controller = RestorationController(worker, sources.check, identity, identity['models'],
                                       deployment='opt_in_U96_staging')
    app = FastAPI(title='Opt-in U96 restoration workflow staging')
    app.include_router(create_router(controller, sources.check))
    app.mount('/static', StaticFiles(directory=str(ROOT / 'static')), name='static')
    @app.get(ROUTE, response_class=HTMLResponse)
    def page():
        current = sources.check()
        template = next(row for row in current['sources'] if row['path'] == 'templates/codeformer_restoration_workflow_v1.html')
        contents = L.B.P.bound(template, 1024 ** 2).read_text(encoding='utf8')
        sources.check()
        return HTMLResponse(contents)
    @app.get('/')
    def home(): return RedirectResponse(ROUTE)
    @app.get('/favicon.ico')
    def favicon(): return Response(status_code=204)
    sock = socket.socket()
    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        sock.bind(('127.0.0.1', args.port))
        identity = sources.check()
        status = worker.measured_status()
        L.B.need(status['neural_forward_attempts'] == status['neural_forward_returns'] == 0
                 and status['completed_comparisons'] == 0 and status['active_worker'] is False
                 and status['uncertain_requests'] == status['uncertain_forward_attempts_upper_bound'] == status['worker_cleanup_failures'] == 0
                 and status['counter_integrity'] is True and status['active_forward_counts_known'] is True,
                 'No image worker or forward may start before an explicit user request.')
        value = {'pid': os.getpid(), 'url': 'http://127.0.0.1:%d%s' % (sock.getsockname()[1], ROUTE),
            'source_identity': identity, 'scope': SCOPE, 'worker_status_at_startup': status,
            'neural_calls_at_startup': 0, 'training_optimizer_VM_calls': 0,
            'model_qualified': False, 'main_app_default_changed': False,
            'process_lifecycle': 'one-disposable-worker-per-request-v2'}
        receipt.parent.mkdir(parents=True, exist_ok=True); L.ordinary(receipt.parent, True)
        with receipt.open('x', encoding='utf8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        print(json.dumps(value), flush=True)
        uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=sock.getsockname()[1], log_level='warning')).run(sockets=[sock])
    finally:
        sock.close()


if __name__ == '__main__':
    main()
