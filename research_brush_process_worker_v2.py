"""Restricted fixed-file worker for the reviewed brush process lifecycle."""
import argparse
import base64
import hashlib
import os
import sys
import time

import research_brush_process_restoration_v2 as L


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    L.B.need(L.TOKEN_RE.fullmatch(args.worker) is not None, 'Generated owned worker token required.')
    directory = L.ordinary(L.REQUEST_ROOT / args.worker, True)
    L.B.need(directory.is_dir(), 'Existing exclusively owned worker directory required.')
    startup = L.read_json(directory / 'startup.json', 16384)
    L.B.need(type(startup) is dict and set(startup) == {'request_id', 'selection', 'deadline_seconds',
            'manifest', 'parent_manifest', 'brush_manifest'} and startup['request_id'] == args.worker
            and startup['selection'] in L.B.P.ARMS and startup['deadline_seconds'] == L.DEADLINE_SECONDS,
            'Exact reviewed worker startup schema required.')
    L.publish(directory, 'identity.json', {'request_id': args.worker, 'worker_pid': os.getpid(),
              'python_prefix': sys.prefix, 'python_executable': sys.executable}, 4096)
    parent, brush, manifest = (startup[key] for key in ('parent_manifest', 'brush_manifest', 'manifest'))
    for row in (parent, brush, manifest):
        L.B.P.bound(row, 1024 ** 2)
    L.configure_sources(L.ROOT / manifest['path'], manifest['sha256'], L.ROOT / parent['path'], parent['sha256'],
                        L.ROOT / brush['path'], brush['sha256'])
    L.B.P.manifest()  # All source/model gates are checked before the unchanged run_one imports Torch.
    deadline = time.monotonic() + L.DEADLINE_SECONDS

    class ObservedRestorer(L.B.SelectedRestorer):
        def count(self, name):
            super().count(name)
            values = self.measured_status()
            sequence = values['neural_forward_attempts'] + values['neural_forward_returns']
            L.B.need(name in ('neural_forward_attempts', 'neural_forward_returns') and 1 <= sequence <= 4,
                     'At most four selected worker counter events required.')
            L.publish(directory, 'event-%d.json' % sequence,
                      {'request_id': args.worker, 'sequence': sequence, 'name': name,
                       'attempts': values['neural_forward_attempts'], 'returns': values['neural_forward_returns']}, 4096)

    restorer = ObservedRestorer()
    for slot in range(2):
        command_path = directory / ('command-%d.json' % slot)
        while not command_path.exists():
            stop = directory / 'stop.json'
            if stop.exists():
                L.B.need(L.read_json(stop, 4096) == {'request_id': args.worker}, 'Exact owned worker stop required.')
                return finish(directory, args.worker, restorer)
            L.B.need(time.monotonic() < deadline, 'Worker request deadline exceeded.')
            time.sleep(.1)
        command = L.read_json(command_path, 4096)
        L.B.need(type(command) is dict and set(command) == {'request_id', 'slot', 'selection', 'input_bytes', 'input_sha256'}
                 and command['request_id'] == args.worker and type(command['slot']) is int and command['slot'] == slot
                 and command['selection'] == startup['selection'] and type(command['input_bytes']) is int
                 and 0 < command['input_bytes'] <= L.B.preparation.MAX_UPLOAD_BYTES
                 and type(command['input_sha256']) is str and len(command['input_sha256']) == 64,
                 'Exact ordered bounded worker command required.')
        path = L.ordinary(directory / ('input-%d.bin' % slot))
        before = path.stat()
        L.B.need(before.st_size == command['input_bytes'], 'Worker input size differs.')
        with path.open('rb') as stream:
            contents = stream.read(L.B.preparation.MAX_UPLOAD_BYTES + 1)
        after = path.stat()
        L.B.need((before.st_size, before.st_mtime_ns, before.st_ino) ==
                 (after.st_size, after.st_mtime_ns, after.st_ino)
                 and hashlib.sha256(contents).hexdigest() == command['input_sha256'], 'Worker input binding differs.')
        ok = True
        try:
            value = restorer.run_one(contents, startup['selection'])
            value['metadata']['worker_lifecycle'] = {'version': 'one-disposable-worker-per-request-v2',
                'request_id': args.worker, 'worker_pid': os.getpid(), 'source_manifest': startup['manifest']}
            value = {key: base64.b64encode(raw).decode('ascii') if key != 'metadata' else raw
                     for key, raw in value.items()}
        except (ValueError, OSError, RuntimeError) as error:
            ok = False
            kind = 'ValueError' if isinstance(error, ValueError) else 'OSError' if isinstance(error, OSError) else 'RuntimeError'
            value = {'kind': kind, 'message': str(error)[:2048]}
        counts = restorer.measured_status()
        L.publish(directory, 'result-%d.json' % slot,
                  {'request_id': args.worker, 'slot': slot, 'selection': startup['selection'],
                   'input_sha256': command['input_sha256'], 'attempts': counts['neural_forward_attempts'],
                   'returns': counts['neural_forward_returns'], 'ok': ok, 'value': value})
        if not ok:
            return finish(directory, args.worker, restorer)
    return finish(directory, args.worker, restorer)


def finish(directory, token, restorer):
    restorer.release_models()
    counts = restorer.measured_status()
    L.publish(directory, 'terminal.json', {'request_id': token, 'clean_exit': True,
              'attempts': counts['neural_forward_attempts'], 'returns': counts['neural_forward_returns']}, 4096)


if __name__ == '__main__':
    main()
