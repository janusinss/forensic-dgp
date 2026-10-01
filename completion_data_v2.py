"""Opt-in, training-only source qualification and geometry-preserving fixtures.

This version does not replace CompletionDataset or any frozen replay/benchmark.
Reflection fixtures are procedural hypotheses, not photorealistic evidence.
Mask=1 denotes introduced reflection; valid=1 denotes real source support.
Training consumers must exclude valid=0 from every supervised region/reduction.
"""
import hashlib
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageOps

FORMAT = 'dgp-qualified-completion-data-v2'
REFLECTION_STYLES = ('white_patch','white_streak','scene_reflection','blue_glare')
PADDING_RGB = (96,96,96)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rgb_array(value):
    array=np.asarray(value)
    require(array.ndim==3 and array.shape[2]==3 and min(array.shape[:2])>=2
            and array.dtype==np.uint8, 'Expected uint8 RGB with dimensions >=2')
    return array


def square_preserving_geometry(rgb, size=256):
    """Fit pixel centers using one scale; pad without inventing face content.

    Equal affine scale preserves angles and ratios, including nonsquare input.
    Conservative valid support excludes any interpolated border contribution.
    This is not facial alignment, and no additional resolution is recovered.
    """
    rgb=rgb_array(rgb)
    require(isinstance(size,int) and not isinstance(size,bool) and size>=32,
            'Output size must be an integer >=32')
    height,width=rgb.shape[:2]
    scale=(size-1)/(max(width,height)-1)
    affine=np.array([[scale,0,(size-1-scale*(width-1))/2],
                     [0,scale,(size-1-scale*(height-1))/2]],np.float64)
    result=cv2.warpAffine(rgb,affine,(size,size),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=PADDING_RGB)
    coverage=cv2.warpAffine(np.ones((height,width),np.float32),affine,(size,size),
                            flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    valid=(coverage>=1.-1e-6).astype(np.uint8)
    result[valid==0]=PADDING_RGB
    require(bool(valid.any()), 'No valid source support')
    return {'rgb':result,'valid':valid,'affine':affine,
            'native_size':[width,height],'size':size,'scale':float(scale)}


def map_points(points, affine):
    points=np.asarray(points,np.float64);affine=np.asarray(affine,np.float64)
    require(points.ndim==2 and points.shape[1]==2 and len(points)>0
            and np.isfinite(points).all() and affine.shape==(2,3)
            and np.isfinite(affine).all(), 'Expected finite Nx2 points and 2x3 affine')
    return points@affine[:,:2].T+affine[:,2]


def load_source(root, record, allowed_training_sources, excluded_sha256):
    """Fail closed: reviewed native training sources, exact bytes and membership.

    Eligibility is recorded by an explicit review, never inferred from a folder,
    a procedural zero mask, native dimensions or absence of detector responses.
    The caller supplies an independently verified training membership/exclusion
    registry. Exact hash checks do not establish identity disjointness.
    """
    require(isinstance(record,dict), 'Source record required')
    name=record.get('path','')
    require(isinstance(name,str) and name and '\\' not in name and ':' not in name,
            'Portable root-relative POSIX source path required')
    relative=PurePosixPath(name)
    require(not relative.is_absolute() and all(p not in ('..','.') for p in name.split('/')),
            'Source path cannot escape root or use dot segments')
    root=Path(root).resolve();path=(root/name).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'Source missing or outside workspace')
    digest=record.get('sha256')
    require(isinstance(digest,str) and len(digest)==64
            and all(c in '0123456789abcdef' for c in digest), 'Source SHA256 required')
    require(record.get('split')=='train' and allowed_training_sources.get(name)==digest
            and digest not in excluded_sha256, 'Source is not an allowed disjoint training member')
    require(hashlib.sha256(path.read_bytes()).hexdigest()==digest, 'Reviewed source bytes changed')
    review=record.get('review',{})
    require(record.get('usage')=='paired_unoccluded'
            and isinstance(review,dict) and review.get('occlusion')=='none_observed'
            and review.get('method') in ('assistant_native_screen','human_native_review')
            and review.get('reference_quality')=='native_low_resolution'
            and isinstance(review.get('rationale'),str) and bool(review['rationale'].strip()),
            'Explicit native unoccluded-source review required; unknown/obscured sources are unpaired')
    require(record.get('eyes_reviewed') is True, 'Reviewed native eye positions required')
    with Image.open(path) as image:
        rgb=np.array(ImageOps.exif_transpose(image).convert('RGB'))
    eyes=np.asarray(record.get('eyes'),np.float64)
    require(eyes.shape==(2,2) and np.isfinite(eyes).all()
            and np.all(eyes>=0) and np.all(eyes[:,0]<rgb.shape[1])
            and np.all(eyes[:,1]<rgb.shape[0]), 'Native eye positions outside oriented source')
    return rgb_array(rgb)


def lens_reflection(rgb, eyes, style, seed, valid=None):
    """Introduce partial opaque lens patches over reviewed visible eyes.

    The framed but unoccluded target is shared across all reflection variants.
    Transparent frames are negative controls; their pixels remain in the target.
    All patch pixels are inside lens interiors. Padding/clipped lenses fail.
    """
    rgb=rgb_array(rgb);height,width=rgb.shape[:2]
    require(style in ('clear',*REFLECTION_STYLES), 'Unknown reflection fixture')
    eyes=np.asarray(eyes,np.float64)
    require(eyes.shape==(2,2) and np.isfinite(eyes).all(), 'Two finite reviewed eyes required')
    eyes=eyes[np.argsort(eyes[:,0])]
    delta=eyes[1]-eyes[0];distance=float(np.linalg.norm(delta))
    require(.15*min(height,width)<=distance<=.65*min(height,width)
            and abs(np.degrees(np.arctan2(delta[1],delta[0])))<=25,
            'Reflection fixtures require two separated, near-level eyes')
    if valid is None:valid=np.ones((height,width),np.uint8)
    valid=np.asarray(valid)
    require(valid.shape==(height,width) and np.isin(valid,[0,1]).all(), 'Binary source support required')
    u=delta/distance;v=np.array([-u[1],u[0]])
    rx,ry=distance*.34,distance*.23
    yy,xx=np.mgrid[:height,:width]
    target=rgb.copy();lens_region=np.zeros((height,width),np.uint8)
    frame=np.zeros_like(lens_region);coordinates=[]
    thickness=max(1,round(distance*.028))
    for eye in eyes:
        dx=((xx-eye[0])*u[0]+(yy-eye[1])*u[1])/rx
        dy=((xx-eye[0])*v[0]+(yy-eye[1])*v[1])/ry
        outer=dx*dx+dy*dy<=1.05**2
        require(eye[0]-rx-thickness>=0 and eye[0]+rx+thickness<width
                and eye[1]-rx-thickness>=0 and eye[1]+rx+thickness<height
                and not np.any(outer & (valid==0)), 'Clipped or padded lens; review source/eye positions')
        inner=(dx*dx+dy*dy<=.86**2)
        lens_region[inner]=1;coordinates.append((dx,dy,inner))
        cv2.ellipse(frame,tuple(np.rint(eye).astype(int)),
                    (max(2,round(rx)),max(2,round(ry))),
                    float(np.degrees(np.arctan2(u[1],u[0]))),0,360,1,thickness,lineType=cv2.LINE_8)
    bridge_start=np.rint(eyes[0]+u*rx).astype(int)
    bridge_end=np.rint(eyes[1]-u*rx).astype(int)
    cv2.line(frame,tuple(bridge_start),tuple(bridge_end),1,thickness)
    require(not np.any(frame & (valid==0)), 'Eyeglass frame extends outside source support')
    # Any rasterized frame is protected even at small native scale.
    lens_region[frame==1]=0
    target[frame==1]=(62,66,71)
    output=target.copy();mask=np.zeros_like(lens_region)
    rng=np.random.default_rng(seed)
    if style!='clear':
        for dx,dy,inner in coordinates:
            shift=float(rng.uniform(-.12,.12))
            if style=='white_patch':
                patch=((dx-shift)**2/.58**2+(dy+.05)**2/.72**2<=1)
            elif style=='white_streak':
                patch=(np.abs(dx+dy*.45-shift)<.28) & (np.abs(dy)<.75)
            elif style=='scene_reflection':
                patch=(dx+dy*.35<.35+shift) & (dx>-.7) & (dy>-.72)
            else:
                patch=(dx-shift)**2/.68**2+(dy-.1)**2/.6**2<=1
            patch &= inner & (lens_region==1)
            mask[patch]=1
        require(bool(mask.any()) and mask.sum()<lens_region.sum(), 'Fixture must partially cover lenses')
        noise=rng.integers(-6,7,(height,width,3))
        if style.startswith('white'):
            texture=np.full((height,width,3),(244,245,239),np.int16)+noise
        elif style=='blue_glare':
            texture=np.full((height,width,3),(51,133,223),np.int16)+noise
            texture+=((yy[:,:,None]%11)<2)*18
        else:
            # A stylized sky/window scene; no external or held-out photo texture.
            texture=np.full((height,width,3),(100,117,130),np.int16)+noise
            texture[((xx//9+yy//13)%3)==0]=(45,53,62)
            texture[(yy%13)<2]=(185,194,197)
        output[mask==1]=np.clip(texture,0,255).astype(np.uint8)[mask==1]
    require(np.array_equal(output[mask==0],target[mask==0]), 'Fixture changed visible target pixels')
    return {'input':output,'target':target,'mask':mask,'lens_region':lens_region,'valid':valid.copy()}


def prepare_case(root, record, allowed_training_sources, excluded_sha256, style,
                 seed=42, size=256):
    rgb=load_source(root,record,allowed_training_sources,excluded_sha256)
    base=square_preserving_geometry(rgb,size)
    eyes=map_points(record['eyes'],base['affine'])
    # Byte digest, rather than absolute host path, makes the seed portable.
    stable=int(record['sha256'][:8],16)
    case_seed=(int(seed)+stable+('clear',*REFLECTION_STYLES).index(style)*1000003)%(2**32)
    result=lens_reflection(base['rgb'],eyes,style,case_seed,base['valid'])
    result['base']=base['rgb']
    result['metadata']={'format':FORMAT,'source_id':record['source_id'],'source':record['path'],
                        'source_sha256':record['sha256'],'split':'train','style':style,'seed':case_seed,
                        'native_size':base['native_size'],'size':size,'affine':base['affine'].tolist(),
                        'eyes_native':record['eyes'],'eyes_output':eyes.tolist(),
                        'padding_rgb':list(PADDING_RGB),'padding_pixels':int((base['valid']==0).sum()),
                        'hole_pixels':int(result['mask'].sum()),
                        'restoration_high_resolution_reference':False,
                        'review_method':record['review']['method'],
                        'simulation':'procedural partial opaque reflection; real transfer unproven'}
    return result
