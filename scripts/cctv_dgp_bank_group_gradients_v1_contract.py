"""Pure scope, affine and aggregation definitions; no neural imports."""
import math

PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
METRICS = ['MSE', 'SSIM_loss', 'ArcFace_loss']


def fixed_reference_matrices(cases, references):
    """Use frozen target->112 affines; five landmarks are not affine coefficients."""
    matrices = []
    for case in cases:
        matrix = references[case['source_person_or_reference']]['matrix112']
        if len(matrix) != 2 or any(len(row) != 3 for row in matrix):
            raise ValueError('Fixed target affine must have shape2x3')
        if not all(math.isfinite(float(value)) for row in matrix for value in row):
            raise ValueError('Nonfinite fixed target affine')
        determinant = float(matrix[0][0])*float(matrix[1][1])-float(matrix[0][1])*float(matrix[1][0])
        if abs(determinant) < 1e-8:
            raise ValueError('Singular fixed target affine')
        matrices.append(matrix)
    return matrices


def scopes(cases, selected_ids):
    wanted = set(selected_ids)
    assert len(wanted) == len(selected_ids) == 100
    blocks = []
    for begin in range(0, len(cases), 5):
        batch = cases[begin:begin+5]
        if batch[0]['id'] not in wanted:
            continue
        assert len(batch) == 5 and all(c['id'] in wanted for c in batch)
        assert [c['profile'] for c in batch] == PROFILES
        assert len({c['source'] for c in batch}) == len({c['source_person_or_reference'] for c in batch}) == 1
        blocks.append(dict(begin=begin, source=batch[0]['source'], reference=batch[0]['source_person_or_reference'],
                           case_ids=[c['id'] for c in batch]))
    assert len(blocks) == 20 and {i for b in blocks for i in b['case_ids']} == wanted
    sources = sorted({b['source'] for b in blocks})
    assert sources == ['dataset/asian_faces', 'dataset/thumbnails128x128']
    assert all(sum(b['source'] == source for b in blocks) == 10 for source in sources)
    groups = [dict(source=source, profile=profile, metric=metric,
                   key=source+'/'+profile+'/'+metric, contributing_references=10)
              for source in sources for profile in PROFILES for metric in METRICS]
    groups += [dict(source=source, profile='degraded', metric='feature_MSE',
                    key=source+'/degraded/feature_MSE', contributing_references=10) for source in sources]
    assert len(groups) == 32 and len({g['key'] for g in groups}) == 32
    return blocks, groups


def batch_values(terms, arc):
    assert all(len(terms[name]) == 5 for name in ['MSE', 'SSIM_loss', 'landmark_structure']) and len(arc) == 5
    return [v for index in range(5) for v in [terms['MSE'][index], terms['SSIM_loss'][index], arc[index]]] + [terms['landmark_structure'][1:].mean()]


def query_key(source, query):
    assert source in ['dataset/asian_faces', 'dataset/thumbnails128x128'] and 0 <= query < 16
    return source+('/'+PROFILES[query//3]+'/'+METRICS[query%3] if query < 15 else '/degraded/feature_MSE')


def accumulate(destination, value, contributors):
    import numpy as np
    assert destination.dtype == np.float64 and value.dtype == np.float32 and destination.shape == value.shape
    assert contributors == 10 and np.isfinite(value).all()
    destination += value.astype(np.float64)/contributors


def projection_seconds(durations):
    assert len(durations) == 2 and all(math.isfinite(t) and t > 0 for t in durations)
    return max(durations)*20*1.5
