"""Experimental training-only anatomical coverings; inference needs no reference."""
import cv2
import numpy as np

VERSION='anatomical-covering-v1'


def anatomical_covering(rgb,landmarks,kind,seed,*,boundary_policy='reject'):
    """Five points: two eyes, nose, two mouth corners in source-image coordinates.

    These polygons model opaque coverings, not photorealistic accessories or glare.
    Invalid geometry returns no sample, never a silently successful empty mask.
    Geometric checks cannot certify detector landmark accuracy or source cleanliness.
    """
    if kind not in ('eyes','lower'):raise ValueError('Only anatomical eyes/lower supported')
    if boundary_policy not in ('reject','clip'):raise ValueError('Unknown boundary policy')
    version=VERSION if boundary_policy=='reject' else 'anatomical-covering-v2-clip'
    rgb=np.asarray(rgb)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:raise ValueError('Expected uint8 RGB')
    h,w=rgb.shape[:2]
    def reject(reason):return None,None,dict(version=version,status='rejected',reason=reason,kind=kind)
    if landmarks is None:return reject('missing_landmarks')
    points=np.array(landmarks,dtype=float,copy=True)
    if points.shape!=(5,2) or not np.isfinite(points).all():return reject('invalid_landmarks')
    if (points<0).any() or (points[:,0]>=w).any() or (points[:,1]>=h).any():return reject('landmarks_outside_image')
    eyes=points[:2][np.argsort(points[:2,0])]
    delta=eyes[1]-eyes[0];distance=float(np.linalg.norm(delta))
    if not .12*min(h,w)<=distance<=.7*min(h,w):return reject('eye_separation')
    u=delta/distance;v=np.array([-u[1],u[0]])
    if abs(np.degrees(np.arctan2(u[1],u[0])))>40:return reject('extreme_roll')
    center=eyes.mean(0);mouth=points[3:].mean(0);nose=points[2]
    mouth_depth=float((mouth-center)@v);nose_depth=float((nose-center)@v)
    mouth_width=float(abs((points[4]-points[3])@u))
    if not .3*distance<mouth_depth<1.8*distance:return reject('mouth_depth')
    if not 0<nose_depth<mouth_depth:return reject('nose_order')
    if not .15*distance<mouth_width<1.2*distance:return reject('mouth_width')
    if abs(float((mouth-center)@u))>.65*distance:return reject('mouth_offset')
    if kind=='eyes':
        # Rotated band surrounding both detected eye centers.
        half_width=.78*distance;half_height=.24*distance
        polygon=np.array([center-half_width*u-half_height*v,center+half_width*u-half_height*v,
                          center+half_width*u+half_height*v,center-half_width*u+half_height*v])
        anchors=points[:2]
    else:
        half_width=max(.74*distance,.8*mouth_width)
        top=mouth-.35*distance*v;bottom=mouth+.48*distance*v
        polygon=np.array([top-half_width*u,top+half_width*u,
                          bottom+.55*distance*u,bottom-.55*distance*u])
        anchors=points[3:]
    # Explicit V2 simulates a covering extending beyond a cropped photograph.
    # Rasterize the original polygon: clamping its vertices would distort edges.
    outside=bool((polygon<0).any() or (polygon[:,0]>=w).any() or (polygon[:,1]>=h).any())
    if outside and boundary_policy=='reject':return reject('polygon_outside_image')
    mask=np.zeros((h,w),np.uint8)
    vertices=np.rint(polygon).astype(np.int32)
    if outside:
        # OpenCV clips raster edges differently on a smaller canvas. Render the
        # whole polygon first, then crop so pixel labels retain crop equivalence.
        left=max(0,2-int(vertices[:,0].min()));top_pad=max(0,2-int(vertices[:,1].min()))
        right=max(w,int(vertices[:,0].max())+3);bottom=max(h,int(vertices[:,1].max())+3)
        canvas=np.zeros((bottom+top_pad,right+left),np.uint8)
        cv2.fillPoly(canvas,[vertices+np.array([left,top_pad])],1)
        mask=canvas[top_pad:top_pad+h,left:left+w].copy()
    else:
        cv2.fillPoly(mask,[vertices],1)
    if not .005<=float(mask.mean())<=.55:return reject('mask_area')
    if any(mask[min(h-1,int(round(y))),min(w-1,int(round(x)))]==0 for x,y in anchors):return reject('anchor_not_covered')
    rng=np.random.default_rng(seed)
    texture=np.clip(rng.uniform(25,230,(1,1,3))+rng.normal(0,8,(h,w,3)),0,255).astype('uint8')
    covered=np.where(mask[...,None].astype(bool),texture,rgb)
    event=dict(version=version,status='generated',kind=kind,seed=int(seed),
                             polygon=polygon.tolist(),mask_fraction=float(mask.mean()),
                             limitation='Five-point anchors do not define full eye/mouth extent or certify realism')
    if boundary_policy=='clip':event['border_clipped']=outside
    return covered,mask,event
