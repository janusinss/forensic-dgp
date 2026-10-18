"""Finite source-derived byte/file projection; runtime guards remain authoritative."""
from . import contract as c


def resource_ledger(static_files=None,static_bytes=None):
    n=c.LIMITS['source_packet_files']if static_files is None else static_files
    b=c.LIMITS['source_packet_bytes']if static_bytes is None else static_bytes
    c.require(type(n)is int and 0<n<=256 and type(b)is int and 0<=b<=32*1024**2,'Invalid static resource projection')
    npy=256*256*3*4+256
    categories={
        'head_RAW':{'files':528,'bytes':528*npy},
        'head_PNG':{'files':528,'bytes':528*c.LIMITS['one256PNG_bytes']},
        'new_parent_RAW':{'files':128,'bytes':128*npy},
        'all_static_source_and64_input_PNG':{'files':n,'bytes':b},
        'all_metadata_and_logs':{'files':528+128+656+32,'bytes':c.LIMITS['metadata_bytes']}}
    files=sum(r['files']for r in categories.values());payload=sum(r['bytes']for r in categories.values())
    c.require(files<=c.LIMITS['regular_files']and payload<=c.LIMITS['result_payload_bytes'],'Prospective payload/file caps insufficient')
    tar=payload+files*1023+1024+10240
    gzip=tar+(tar>>12)+(tar>>14)+(tar>>25)+31
    c.require(gzip<=c.LIMITS['archive_bytes'],'Prospective one-gzip archive cap insufficient')
    temporary=c.LIMITS['one_array_bytes']
    disk=c.LIMITS['result_payload_bytes']+c.LIMITS['archive_bytes']+temporary+c.LIMITS['reserve_free_bytes']
    c.require(disk<c.LIMITS['initial_free_bytes_after_install'],'Post-install admission lacks payload/archive/temp/reserve margin')
    return {'format':'codeformer-fixed-state-response-resource-ledger-v1','complete':True,'prospective_only':True,
        'measured_peak_resources':False,'design_sha256':c.DESIGN_SHA256,'operation_reservations':656,
        'neural_forwards':656,'head_forwards':528,'new_CodeFormer_forwards':128,'gradient_queries':0,'optimizer_steps':0,
        'payload_categories':categories,'projected_files':files,'projected_payload_bytes':payload,
        'USTAR_padding_header_EOA_bound_bytes':tar,'single_default_gzip_bound_bytes':gzip,
        'declared_caps_simultaneous_disk_bytes':disk,
        'admission_margin_bytes':c.LIMITS['initial_free_bytes_after_install']-disk,
        'premises':['All528head responses return one RAW,onePNG,oneJSON; all128newparentRAW plus at most128parentJSON.',
            'All656prospectivecall records and32fixedmetadata/logfiles fit aggregate128MiB metadata envelope.',
            'Static count/bytes includes64shippedinputPNGs andallsource/data/provenance files.',
            'One USTAR regular-only archive with onegzip frame; no duplicate full archive or model/checkpoint output.',
            'At most one1MiBatomic array-write temp; file/disk guards must enforce actual writes and sampled reserve.',
            'No new checkpoints; all initializer/head/base/source states stay read-only parent dependencies.'],
        'runtime_measurements_required':['peak_RSS_bytes','peak_CUDA_reserved_bytes','minimum_sampled_root_free_bytes','phase_seconds'],
        'CPU_index_array_projection_bytes':200*(3*256*256*3*4+256*256)+64*(256*256*3*4+256*256),
        'RAM_CUDA_projection_is_not_a_measured_peak':True}
