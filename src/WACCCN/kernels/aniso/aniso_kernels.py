#!/usr/bin/env python3
import os, re, glob, numpy as np
from scipy.spatial import cKDTree

eps = 1e-12

def to_xy(coords):
    arr = np.asarray(coords)
    return arr[:, -2:].astype(float)

def median_nn_distance(coords):
    xy = to_xy(coords); d,_ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:,1]))

def build_radius_graph(coords, R_rel=2.2, deg_min=10, deg_max=18, h=None, Rmax_rel=None):
    xy = to_xy(coords); 
    if h is None: h = median_nn_distance(xy)
    R = R_rel*h; tree = cKDTree(xy)
    neighbors = tree.query_ball_point(xy, r=R)
    pairs=[]
    for i, idxs in enumerate(neighbors):
        for j in idxs:
            if j!=i: pairs.append((i,j))
    return np.asarray(pairs), h, R

def p99_median_normalize_map(X, mask=None):
    X = np.asarray(X,float); 
    if X.ndim==2: X=X[...,None]
    H,W,C=X.shape
    if mask is None: mask=np.ones((H,W),bool)
    out=np.empty_like(X)
    for c in range(C):
        vals=X[...,c]; msel=mask & np.isfinite(vals)
        hi=np.percentile(vals[msel],99) if np.any(msel) else 0
        vals=np.minimum(vals,hi)
        med=np.median(vals[msel]) if np.any(msel) else 1.0
        out[...,c]=vals/(med+eps)
    return out if C>1 else out[...,0]

def corridor_pixel_weights(ci,cj,H,W,tissue_mask=None,
                           sigma_perp=1.0,sigma_para=1.0,lateral_mult=3.0):
    ci=np.asarray(ci,float); cj=np.asarray(cj,float)
    v=cj-ci; d=float(np.linalg.norm(v))
    if d<1e-9: return np.array([],int),np.array([],int),np.array([],float)
    u=v/d; e2=np.array([-u[1],u[0]])
    lat=lateral_mult*sigma_perp+1
    x_min=max(0,int(min(ci[0],cj[0])-lat)); x_max=min(W-1,int(max(ci[0],cj[0])+lat))
    y_min=max(0,int(min(ci[1],cj[1])-lat)); y_max=min(H-1,int(max(ci[1],cj[1])+lat))
    XX,YY=np.meshgrid(np.arange(x_min,x_max+1),np.arange(y_min,y_max+1))
    dx,dy=XX-ci[0],YY-ci[1]; s=dx*u[0]+dy*u[1]; ell=dx*e2[0]+dy*e2[1]
    sel=(s>=0)&(s<=d)&(np.abs(ell)<=lateral_mult*sigma_perp)
    if tissue_mask is not None: sel&=tissue_mask[YY,XX]
    w=np.exp(-(ell**2)/(2*sigma_perp**2))*np.exp(-((s-0.5*d)**2)/(2*sigma_para**2))*sel
    if w.sum()<=0: return np.array([],int),np.array([],int),np.array([],float)
    w=w/(w.sum()+eps)
    return YY[sel].ravel().astype(int),XX[sel].ravel().astype(int),w[sel].ravel()

