"""Shared frozen-data utilities for the CCTV DGP VM diagnostic. No optimizer here."""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from skimage.metrics import structural_similarity


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path,data):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(data,indent=2,allow_nan=False)+"\n")


def state_hash(model):
    digest=hashlib.sha256()
    state=model.state_dict() if hasattr(model,"state_dict") else model
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(name.encode()+b"\0"+str(value.dtype).encode()+b"\0")
        digest.update(str(tuple(value.shape)).encode()+b"\0"+value.numpy().tobytes())
    return digest.hexdigest()


def buffer_hash(model):
    return state_hash(dict(model.named_buffers()))


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("VM CUDA required: no CPU training or backward preflight is permitted")


def grid112(matrix):
    """Inverse of saved target->112 affine, sampled with align_corners=False."""
    matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.shape!=(2,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:,:2]))<1e-8:
        raise ValueError("Invalid fixed target affine")
    inverse=cv2.invertAffineTransform(matrix)
    yy,xx=np.meshgrid(np.arange(112),np.arange(112),indexing="ij")
    x=inverse[0,0]*xx+inverse[0,1]*yy+inverse[0,2]
    y=inverse[1,0]*xx+inverse[1,1]*yy+inverse[1,2]
    return np.stack((2*(x+.5)/256-1,2*(y+.5)/256-1),axis=-1).astype(np.float32)


def identity_crop(images,mask,grid):
    crop=F.grid_sample(images.float(),grid.float(),mode="bilinear",padding_mode="zeros",align_corners=False)
    support=F.grid_sample(mask.float(),grid.float(),mode="nearest",padding_mode="zeros",align_corners=False)
    return crop*support+(128/255)*(1-support)


class FixedObservedIdentity(nn.Module):
    def __init__(self,path,device):
        super().__init__()
        from models.identity_loss import ArcFaceIdentityLoss
        # Reuse the validated frozen ONNX converter, not its68-point landmark adapter.
        self.encoder=ArcFaceIdentityLoss(path,device=device).encoder
        self.eval().requires_grad_(False)

    def train(self,mode=True):
        super().train(False)
        return self

    def embedding(self,images,mask,grid):
        crop=identity_crop(images,mask,grid)
        result=F.normalize(self.encoder(crop*2-1).float(),dim=1)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite identity embedding")
        return result


class PilotPerceptual(nn.Module):
    def __init__(self,path,device):
        super().__init__()
        from torchvision.models.vgg import make_layers,cfgs
        self.features=make_layers(cfgs["E"])[:27]
        self.features.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
        self.register_buffer("mean",torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer("std",torch.tensor([.229,.224,.225]).view(1,3,1,1))
        self.to(device).eval().requires_grad_(False)

    def taps(self,x):
        x=(x-self.mean)/self.std
        outputs=[]
        for index,layer in enumerate(self.features):
            x=layer(x)
            if index in (3,8,17,26):
                outputs.append(x)
        return outputs

    def forward(self,x,target):
        actual=self.taps(x)
        with torch.no_grad():
            expected=self.taps(target)
        return sum(weight*F.l1_loss(a,b) for weight,a,b in zip((.1,.2,1.,1.),actual,expected))


def composite(generated,input_rgb,observed):
    return generated*observed+input_rgb*(1-observed)


def image_tensor(rgb):
    return torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).float()/255


class PilotDataset(torch.utils.data.Dataset):
    """Prepared target/input files remove codec/runtime drift between VM arms."""
    def __init__(self,root,protocol,cases):
        self.root=Path(root)
        self.references={r["id"]:r for r in protocol["references"]}
        self.cases=cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self,index):
        case=self.cases[index];ref=self.references[case["reference_id"]]
        low=image_tensor(Image.open(self.root/case["input"]).convert("RGB"))
        target=image_tensor(Image.open(self.root/ref["target"]).convert("RGB"))
        mask=torch.from_numpy((np.asarray(Image.open(self.root/ref["observed"]))>0).copy()).float()[None]
        return {"low":low,"target":target,"mask":mask,"grid":torch.from_numpy(grid112(ref["matrix112"])),
                "index":index,"reference_id":ref["id"],"source":ref["source"],"profile":case["profile"],"case_id":case["id"]}


def masked_charbonnier(generated,target,mask):
    error=torch.sqrt((generated-target).square()+1e-6)*mask
    return (error.sum((1,2,3))/(3*mask.sum((1,2,3)).clamp_min(1))).mean()


def base_loss(generated,target,mask,perceptual):
    from models.losses import SobelGradientLoss
    pixel=masked_charbonnier(generated,target,mask)
    color=F.l1_loss(F.avg_pool2d(generated,11,stride=1,padding=5),F.avg_pool2d(target,11,stride=1,padding=5))
    vgg=perceptual(generated,target)
    # A fixed, low-weight edge term; no FFT or unreliable68-point target fabrication.
    edges=SobelGradientLoss(device=str(generated.device))(generated,target)
    return pixel+.05*color+.1*vgg+.05*edges,{"pixel":pixel.item(),"color":color.item(),"vgg":vgg.item(),"sobel":edges.item()}


def freeze_normalization(model):
    for module in model.modules():
        if isinstance(module,(nn.modules.batchnorm._BatchNorm,nn.modules.instancenorm._InstanceNorm)):
            module.eval()


def exported_pixel_metrics(prediction,target,mask):
    """Evaluate the exact deliverablePNG, not an unexported float stage."""
    actual=prediction.astype(np.float32)/255;ref=target.astype(np.float32)/255
    error=actual-ref;mse=float(np.square(error[mask]).astype(np.float64).mean())
    _,ssmap=structural_similarity(ref,actual,data_range=1,channel_axis=-1,win_size=7,full=True)
    interior=cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    return {"MSE":mse,"PSNR":float(-10*np.log10(mse)) if mse else None,"perfect_match":mse==0,
            "SSIM":float(ssmap[interior].astype(np.float64).mean()),"MAE":float(np.abs(error[mask]).astype(np.float64).mean())}


def aggregate(rows):
    groups={"degraded": [r for r in rows if r["profile"]!="clear"],"clear":[r for r in rows if r["profile"]=="clear"]}
    for source in sorted({r["source"] for r in rows}):
        for profile in sorted({r["profile"] for r in rows}):
            groups[source+"/"+profile]=[r for r in rows if r["source"]==source and r["profile"]==profile]
        groups[source+"/degraded"]=[r for r in rows if r["source"]==source and r["profile"]!="clear"]
    result={}
    for key,items in groups.items():
        if not items:
            raise ValueError("Missing fixed evaluation group")
        mse=float(np.mean([r["MSE"] for r in items]))
        result[key]={"cases":len(items),"identity_pairs":sum(r["ArcFace_observed_fixed"] is not None for r in items),
                     "MSE":mse,"PSNR":float(-10*np.log10(mse)) if mse else None,
                     "perfect_matches":sum(r["perfect_match"] for r in items),
                     "SSIM":float(np.mean([r["SSIM"] for r in items])),"MAE":float(np.mean([r["MAE"] for r in items])),
                     "ArcFace_observed_fixed":float(np.mean([r["ArcFace_observed_fixed"] for r in items]))}
    return result


def qualifies(candidate,baseline,best):
    """Strict baseline source/profile guard; PNG metrics and frozen shared cohort."""
    if set(candidate)!=set(baseline) or set(candidate)!=set(best):
        return False
    for group,reference in baseline.items():
        current=candidate[group]
        if current["cases"]!=reference["cases"] or current["identity_pairs"]!=reference["identity_pairs"] or current["identity_pairs"]!=current["cases"]:
            return False
        for metric in ("MSE","SSIM","ArcFace_observed_fixed"):
            if current.get(metric) is None or reference.get(metric) is None or not math.isfinite(current[metric]) or not math.isfinite(reference[metric]):
                return False
            if metric=="MSE" and (current[metric]<0 or reference[metric]<0 or current[metric]>reference[metric]+1e-12):
                return False
            if metric!="MSE" and current[metric]<reference[metric]-1e-6:
                return False
    best_mse=best["degraded"]["MSE"]
    return math.isfinite(best_mse) and best_mse>0 and candidate["degraded"]["MSE"]<=best_mse*10**(-.1/10)


def verify_bundle(root):
    root=Path(root)
    if (root/"protocol.sha256").read_text(encoding="ascii").strip()!=sha(root/"protocol.json"):
        raise ValueError("Frozen protocol fingerprint differs")
    protocol=read(root/"protocol.json")
    if protocol["arms"]!=[{"id":"camera_no_identity","lambda_identity":0.},{"id":"camera_identity","lambda_identity":.1}]:
        raise ValueError("Matched factor/configuration differs")
    for name,pin in protocol["assets_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name or not (root/name).resolve().is_relative_to(root.resolve()) or sha(root/name)!=pin:
            raise ValueError("Changed/unsafe bundle asset: "+name)
    if protocol["batch_size"]!=8 or protocol["epochs"]!=2 or protocol["runtime_cap_seconds"]!=5400 or protocol["seed"]!=20261003:
        raise ValueError("Frozen runtime/epoch/batch settings differ")
    train=[r for r in protocol["references"] if r["role"]=="train"]
    validation=[r for r in protocol["references"] if r["role"]=="validation"]
    if not train or not validation or len(train)>1024 or len(validation)>128:
        raise ValueError("Finite reference counts differ")
    if len({r["source_sha256"] for r in protocol["references"]})!=len(protocol["references"]):
        raise ValueError("Reference byte duplicates")
    if len({r["id"] for r in protocol["references"]})!=len(protocol["references"]):
        raise ValueError("Duplicate reference ids")
    sources={"dataset/thumbnails128x128","dataset/asian_faces"}
    if {r["source"] for r in train}!=sources or {r["source"] for r in validation}!=sources:
        raise ValueError("Source cohort differs")
    if len({sum(r["source"]==s for r in train) for s in sources})!=1:
        raise ValueError("Training source balance differs")
    for ref in protocol["references"]:
        grid112(ref["matrix112"])
        if ref["role"] not in ("train","validation") or any(ref[k] not in protocol["assets_sha256"] for k in ("target","observed")):
            raise ValueError("Missing reference asset/role")
    if set(protocol["training_epochs"])!={"1","2"}:
        raise ValueError("Frozen epoch keys differ")
    for epoch in protocol["training_epochs"]:
        cases=protocol["training_epochs"][epoch]
        if len(cases)!=len(train) or {c["reference_id"] for c in cases}!={r["id"] for r in train}:
            raise ValueError("Matched epoch membership differs")
    val=protocol["validation_cases"]
    profiles={"clear","blur_lr24","lowlight_lr32","motion_lr48","compound_lr24"}
    if len(val)!=5*len(validation) or {(c["reference_id"],c["profile"]) for c in val}!={(r["id"],p) for r in validation for p in profiles}:
        raise ValueError("Validation coverage differs")
    all_cases=[c for cases in protocol["training_epochs"].values() for c in cases]+val
    if len({c["id"] for c in all_cases})!=len(all_cases) or any(c["input"] not in protocol["assets_sha256"] or c["profile"] not in profiles for c in all_cases):
        raise ValueError("Case asset/profile/identity differs")
    if len(set(protocol["preview_case_ids"]))!=10 or not set(protocol["preview_case_ids"])<=set(c["id"] for c in val):
        raise ValueError("Fixed ten-row preview differs")
    if any(protocol["weights"][k] not in protocol["assets_sha256"] for k in ("dgp","arcface","vgg_trunk")):
        raise ValueError("Missing weight asset")
    updates=2*math.ceil(len(train)/8)
    if protocol["updates_per_arm"]!=updates or protocol["expected_total_updates"]!=2*updates or updates>256:
        raise ValueError("Frozen finite update budget differs")
    return protocol
