"""Mutation regressions for independent return archive/event verification."""
import gzip
import importlib.util
import io
import tarfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('independent_readout_transport', ROOT / 'scripts/audit_u96_decoder_readout_transport_v1.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_single_gzip_large_fixture_exact(self):
        value = bytes(range(256)) * 4096
        stream = audit.SingleGzip(io.BytesIO(gzip.compress(value)))
        chunks = []
        while True:
            block = stream.read(10240)
            if not block:
                break
            chunks.append(block)
        self.assertEqual(b''.join(chunks), value)
        self.assertTrue(stream.done)

    def test_concatenated_or_trailing_or_truncated_gzip_refused(self):
        for raw in (gzip.compress(b'a') + gzip.compress(b'b'), gzip.compress(b'a') + b'x', gzip.compress(b'a')[:-3]):
            stream = audit.SingleGzip(io.BytesIO(raw))
            with self.assertRaises(ValueError):
                while stream.read(65536):
                    pass

    def test_crc_mutation_refused(self):
        raw = bytearray(gzip.compress(b'face'))
        raw[-8] ^= 1
        with self.assertRaises(Exception):
            audit.SingleGzip(io.BytesIO(raw)).read(65536)

    def tar_tail(self, mutation):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode='w') as tar:
            info = tarfile.TarInfo(audit.STAGE_NAME + '/face.txt')
            info.size = 4
            tar.addfile(info, io.BytesIO(b'face'))
        plain = bytearray(raw.getvalue())
        if mutation:
            plain[1536] = 1  # Inside tarfile's buffered bytes after first EOF block.
        stream = audit.SingleGzip(io.BytesIO(gzip.compress(plain)))
        last = 0
        with tarfile.open(fileobj=stream, mode='r|') as tar:
            for member in tar:
                last = ((member.offset_data + member.size + 511) // 512) * 512
                tar.extractfile(member).read()
        while stream.read(65536):
            pass
        return stream, last

    def test_complete_tar_padding_including_reader_buffer(self):
        stream, last = self.tar_tail(False)
        self.assertEqual(audit.verify_tar_end(stream, last), 9216)

    def test_hidden_payload_inside_tar_reader_buffer_refused(self):
        stream, last = self.tar_tail(True)
        with self.assertRaisesRegex(ValueError, 'hidden tar payload'):
            audit.verify_tar_end(stream, last)

    def test_namespace_paths_refuse_escapes_links_syntax(self):
        for value in ('/x', audit.STAGE_NAME + '/../face', audit.STAGE_NAME + '/a//x',
                      audit.STAGE_NAME + '/a/./x', audit.STAGE_NAME + '/a\\x', audit.STAGE_NAME + '/C:/x'):
            with self.assertRaises(ValueError):
                audit.member_relative(value)
        self.assertEqual(audit.member_relative(audit.STAGE_NAME + '/outputs/face.npy'), 'outputs/face.npy')

    def test_strict_json_rejects_duplicate_and_nonfinite_nested(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":[1e999]}', b'{"a":NaN}'):
            with self.assertRaises(ValueError):
                audit.strict_json(raw)

    def forward_events(self):
        first = {key: 0 for key in audit.EXPECTED_COUNTERS}
        first.update(full_forward_attempts=1, reference_forwards=1, frozen_U96_adapter_site_reservations=4)
        second = dict(first, full_forward_returns=1, frozen_U96_adapter_site_returns=4)
        return [dict(kind='forward_reserved', seconds=0, forward_kind='reference_forwards', counters=first),
                dict(kind='forward_returned', seconds=1, forward_kind='reference_forwards',
                     sites=['32', '64', '128', '256'], counters=second)]

    def test_complete_reserved_return_pair_reconstructed(self):
        row = audit.event_readback(self.forward_events())
        self.assertIsNone(row['unknown_operations']['forward'])
        self.assertEqual(row['independently_reconstructed_counters']['full_forward_returns'], 1)

    def test_unreserved_return_or_duplicate_reservation_refused(self):
        rows = self.forward_events()
        with self.assertRaisesRegex(ValueError, 'Unreserved'):
            audit.event_readback(rows[1:])
        with self.assertRaisesRegex(ValueError, 'prior outcome unknown'):
            audit.event_readback([rows[0], rows[0]])

    def test_attempt_without_return_keeps_unknown(self):
        row = audit.event_readback(self.forward_events()[:1])
        self.assertEqual(row['unknown_operations']['forward'], 'reference_forwards')

    def test_saved_counter_mutation_refused(self):
        rows = self.forward_events()
        rows[1]['counters']['optimizer_updates'] = 1
        with self.assertRaisesRegex(ValueError, 'counters differ'):
            audit.event_readback(rows)

    def test_backward_match_and_time_are_not_inferred(self):
        rows = self.forward_events()
        rows[1]['seconds'] = -1
        with self.assertRaisesRegex(ValueError, 'Monotone'):
            audit.event_readback(rows)
        with self.assertRaisesRegex(ValueError, 'Four backwards'):
            audit.event_readback([dict(kind='optimizer_reserved', update=1, seconds=0,
                                       counters={key: 0 for key in audit.EXPECTED_COUNTERS})])


if __name__ == '__main__':
    unittest.main()
