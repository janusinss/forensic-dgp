"""Export matched CPU initialization and forward checks; zero training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(args):
    sys.path.insert(0, str(ROOT/'outputs/face_extraction_dependencies'))
    import torch
    from face_occlusion_adapter import initialize_pair, load_adapter, FORMAT, TARGET, SOURCE_SHA, SMP_VERSION
    from detector_training import load_manifest, ReviewedMasks
    from scripts.train_coverage_vm import sha

    torch.set_num_threads(4)
    source = ROOT/'outputs/face_extraction_epoch16.ckpt'
    if sha(source) != SOURCE_SHA:
        raise ValueError('Source checkpoint changed')
    out = ROOT/args.output
    if not out.resolve().is_relative_to(ROOT) or out.exists():
        raise ValueError('Preserve existing outputs; choose a new workspace directory')
    manifest = ROOT/'dataset/detector_training_extension_v2/manifest.json'
    if sha(manifest) != '1f9787ba7bf184f641b298863ae90250b90706e74b6163517a22efe293fcbb2c':
        raise ValueError('Real manifest changed')
    rows = [r for r in load_manifest(manifest) if r['split']=='train']
    image, _ = ReviewedMasks(rows,256)[0]
    pair = initialize_pair(torch.load(source, map_location='cpu', weights_only=True), seed=42)
    phead = pair['pretrained'].network.segmentation_head.state_dict()
    rhead = pair['random'].network.segmentation_head.state_dict()
    if not all(torch.equal(phead[k], rhead[k]) for k in phead):
        raise ValueError('New heads differ across arms')
    if all(torch.equal(v, pair['random'].network.encoder.state_dict()[k])
           for k,v in pair['pretrained'].network.encoder.state_dict().items()):
        raise ValueError('Encoder initializations unexpectedly equal')
    out.mkdir(parents=True)
    report = dict(optimizer_updates=0, seed=42, source_sha256=SOURCE_SHA,
                  source_revision='e75d4a83a696bd7379128319244ef6e5e7885fc8',
                  adapter_sha256=sha(ROOT/'face_occlusion_adapter.py'),
                  exporter_sha256=sha(__file__), manifest_sha256=sha(manifest),
                  torch=str(torch.__version__), heads_equal=True,
                  input_image=rows[0]['image'], input_image_sha256=rows[0]['image_sha256'], arms={})
    for arm,model in pair.items():
        model.eval()
        before = {k:v.clone() for k,v in model.state_dict().items()}
        with torch.inference_mode():
            output = model.detect(image[None])
        if not all(torch.equal(v,before[k]) for k,v in model.state_dict().items()):
            raise ValueError('Forward changed initialization state')
        payload = dict(format=FORMAT, target=TARGET, architecture='resnet18-unet',
                       smp_version=SMP_VERSION, model=model.state_dict(), initialization=arm,
                       optimizer_updates=0, source_sha256=SOURCE_SHA, seed=42,
                       batchnorm_policy='fixed_running_statistics_trainable_affine')
        path = out/f'{arm}.pth';torch.save(payload,path)
        loaded,_ = load_adapter(path,allow_initial=True)
        with torch.inference_mode():
            restored = loaded.detect(image[None])
        if not torch.equal(output,restored):
            raise ValueError('Saved initialization does not reproduce its forward output')
        report['arms'][arm] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                                  parameters=sum(p.numel() for p in model.network.parameters()),
                                  output_shape=list(output.shape), finite=True, exact_reload=True,
                                  state_unchanged=True)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/face_occlusion_initial_v1')
    main(parser.parse_args())
