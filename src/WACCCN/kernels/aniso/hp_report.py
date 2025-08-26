#!/usr/bin/env python3
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

def load_xy(path):
    if path.endswith(".npy"): return np.load(path)[:,-2:].astype(float)
    df=pd.read_csv(path)
    x_candidates=["x","x_um","X","coord_x","imagecol","x_pixel","xpix"]
    y_candidates=["y","y_um","Y","coord_y","imagerow","y_pixel","ypix"]
    xcol=next((c for c in x_candidates if c in df.columns),None)
    ycol=next((c for c in y_candidates if c in df.columns),None)
    if xcol and ycol: return df[[xcol,ycol]].to_numpy(float)
    num=df.select_dtypes(include=[np.number])
    return num.iloc[:,-2:].to_numpy(float)

def weak_cc(n,edges):
    adj=[[] for _ in range(n)]
    for u,v in edges: adj[u].append(v); adj[v].append(u)
    seen=np.zeros(n,bool); comps=0; largest=0
    for i in range(n):
        if not seen[i]:
            comps+=1; s=[i]; seen[i]=True; size=0
            while s:
                u=s.pop(); size+=1
                for w in adj[u]:
                    if not seen[w]: seen[w]=True; s.append(w)
            largest=max(largest,size)
    return comps,largest

def report(coords_path,npz_path):
    xy=load_xy(coords_path); z=np.load(npz_path)
    edges=z["edges"]; K=z["kernel"]; a=z["a"]; r=z["r"]
    N=xy.shape[0]; E=edges.shape[0]
    print("N:",N,"E:",E)
    cc_count,cc_big=weak_cc(N,edges)
    print("CCs:",cc_count,"largest:",cc_big)
    print("K range:",K.min(),K.max())
    print("a range:",a.min(),a.max())
    print("r range:",r.min(),r.max())
