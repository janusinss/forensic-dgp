"""Exact measured original-DGP parameter partition; metadata only."""
FUSION_NAMES = (
    'fpn.lateral0.weight', 'fpn.lateral1.weight', 'fpn.lateral2.weight',
    'fpn.lateral3.weight', 'fpn.lateral4.weight',
    'fpn.td1.0.weight', 'fpn.td1.0.bias', 'fpn.td2.0.weight',
    'fpn.td2.0.bias', 'fpn.td3.0.weight', 'fpn.td3.0.bias',
)
DECODER_NAMES = (
    'head1.block0.weight', 'head1.block1.weight', 'head2.block0.weight',
    'head2.block1.weight', 'head3.block0.weight', 'head3.block1.weight',
    'smooth.0.weight', 'smooth.0.bias', 'smooth2.0.weight', 'smooth2.0.bias',
    'final.weight', 'final.bias',
)
SELECTED_NAMES = (
    'fpn.td1.0.weight', 'fpn.td1.0.bias', 'fpn.td2.0.weight', 'fpn.td2.0.bias',
    'fpn.td3.0.weight', 'fpn.td3.0.bias', 'fpn.lateral4.weight',
    'fpn.lateral3.weight', 'fpn.lateral2.weight', 'fpn.lateral1.weight',
    'fpn.lateral0.weight',
) + DECODER_NAMES
FUSION_PARAMETERS = 479616
DECODER_PARAMETERS = 498627
SELECTED_PARAMETERS = 978243


def validate_layout(layout):
    assert len(layout) == 23 and [r['name'] for r in layout] == list(SELECTED_NAMES)
    offset, fusion, decoder = 0, 0, 0
    for row in layout:
        shape = row['shape']
        assert shape and all(type(n) is int and n > 0 for n in shape)
        count = 1
        for n in shape:
            count *= n
        assert type(row['start']) is int and type(row['end']) is int
        assert row['start'] == offset and row['end'] == offset + count
        offset = row['end']
        if row['name'] in FUSION_NAMES:
            fusion += count
        else:
            decoder += count
    assert offset == SELECTED_PARAMETERS and fusion == FUSION_PARAMETERS and decoder == DECODER_PARAMETERS
    return True
