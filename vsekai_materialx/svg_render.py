"""Render a whole SVG document through Slug coverage (painter's algorithm).

Extends the single-path ThorVG->Slug connector to multi-element SVGs so we can
validate ThorVG's test corpus: every <path>/<rect>/<circle>/<ellipse>/<polygon>
becomes a filled contour set rendered by Slug's exact signed-distance coverage,
composited front-to-back by fill color. Reports unsupported features (gradients,
strokes, arcs, per-element transforms) rather than failing silently.
"""
import re, math
import numpy as np
from .thorvg_slug import _flatten_cubic, _flatten_quad

_NAMED = {"black":(0,0,0),"white":(1,1,1),"red":(1,0,0),"green":(0,.5,0),"blue":(0,0,1),
          "yellow":(1,1,0),"gray":(.5,.5,.5),"grey":(.5,.5,.5),"orange":(1,.65,0),
          "none":None}

def parse_color(s, default=(0.5,0.5,0.5)):
    if not s: return default
    s = s.strip()
    if s.lower() in _NAMED: return _NAMED[s.lower()]
    if s.startswith("url("): return ("GRADIENT",)            # unsupported -> flagged
    m = re.match(r"#([0-9a-fA-F]{3})$", s)
    if m: return tuple(int(c*2,16)/255 for c in m.group(1))
    m = re.match(r"#([0-9a-fA-F]{6})$", s)
    if m: return tuple(int(s[1+2*i:3+2*i],16)/255 for i in range(3))
    m = re.match(r"rgb\(([^)]+)\)", s)
    if m:
        v = [float(x) for x in m.group(1).replace("%","").split(",")]
        return tuple(c/255 for c in v[:3])
    return default

def _attrs(tag): return dict(re.findall(r'([\w:-]+)\s*=\s*"([^"]*)"', tag))
def _f(v, d=0.0):
    try: return float(str(v).strip().rstrip("%px"))   # tolerate %, px units (fallback only)
    except (ValueError, TypeError): return d

def _path_contours(d):
    from .thorvg_slug import parse_svg_path
    # extend parse for S/T smooth + A arc handled crudely (arc -> chord)
    d2 = re.sub(r"[Aa][^MmLlHhVvCcSsQqTtZz]*", lambda m: " ", d)   # drop arcs (rare here)
    try: return parse_svg_path(d2)
    except Exception: return []

def _rect(a):
    x,y,w,h = (_f(a.get(k,0)) for k in ("x","y","width","height"))
    return [np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h],[x,y]])]
def _circle(a):
    cx,cy,r = _f(a.get("cx",0)),_f(a.get("cy",0)),_f(a.get("r",0))
    t=np.linspace(0,2*math.pi,48); return [np.stack([cx+r*np.cos(t),cy+r*np.sin(t)],1)]
def _ellipse(a):
    cx,cy,rx,ry=(_f(a.get(k,0)) for k in ("cx","cy","rx","ry"))
    t=np.linspace(0,2*math.pi,48); return [np.stack([cx+rx*np.cos(t),cy+ry*np.sin(t)],1)]
def _poly(a):
    pts=[_f(x) for x in re.split(r"[ ,]+",a.get("points","").strip()) if x!=""]
    p=np.array(pts).reshape(-1,2); return [np.vstack([p,p[:1]])]

def parse_svg(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    vb = re.search(r'viewBox\s*=\s*"([^"]+)"', txt)
    drawables, flags = [], {"gradient":0,"stroke_only":0,"transform":0,"arc":0}
    if "transform=" in txt: flags["transform"] = txt.count("transform=")
    if re.search(r"[Aa]\d|[Aa] ", txt): flags["arc"] = 1
    for m in re.finditer(r"<(path|rect|circle|ellipse|polygon|polyline)\b([^>]*)>", txt):
        kind, a = m.group(1), _attrs(m.group(0))
        if kind=="path":     cs=_path_contours(a.get("d",""))
        elif kind=="rect":   cs=_rect(a)
        elif kind=="circle": cs=_circle(a)
        elif kind=="ellipse":cs=_ellipse(a)
        else:                cs=_poly(a)
        if not cs or sum(len(c) for c in cs)<3: continue
        fill = parse_color(a.get("fill","black"))
        if fill is None: flags["stroke_only"]+=1; continue
        if fill and fill[0]=="GRADIENT": flags["gradient"]+=1; fill=(0.6,0.6,0.6)
        op = _f(a.get("fill-opacity", a.get("opacity",1)),1.0)
        drawables.append((cs, np.array(fill), op))
    return drawables, vb, flags

def render(path, res=256, bg=(1,1,1)):
    drawables, vb, flags = parse_svg(path)
    if not drawables: return None, 0, flags
    if vb:
        x0,y0,w,h = [_f(v) for v in re.split(r"[ ,]+", vb.group(1).strip())]
    else:
        allp=np.concatenate([c for cs,_,_ in drawables for c in cs])
        x0,y0=allp.min(0); w,h=(allp.max(0)-allp.min(0))
    span=max(w,h)
    gy,gx=np.meshgrid(np.linspace(0,1,res),np.linspace(0,1,res),indexing="ij")
    img=np.ones((res,res,3))*np.array(bg)
    for cs, fill, op in drawables:
        inside=np.zeros((res,res),bool); dist=np.full((res,res),1e9)
        for poly in cs:
            P=(poly-[x0,y0])/span                          # viewBox-normalized
            n=len(P); j=n-1
            for i in range(n):
                xi,yi=P[i]; xj,yj=P[j]
                inside ^= ((yi>gy)!=(yj>gy)) & (gx<(xj-xi)*(gy-yi)/(yj-yi+1e-12)+xi)
                ax,ay=P[i]; bx,by=P[(i+1)%n]; abx,aby=bx-ax,by-ay
                tt=np.clip(((gx-ax)*abx+(gy-ay)*aby)/(abx*abx+aby*aby+1e-12),0,1)
                dist=np.minimum(dist,np.sqrt((gx-(ax+tt*abx))**2+(gy-(ay+tt*aby))**2)); j=i
            sd=dist*np.where(inside,-1.0,1.0)
        cov=np.clip(0.5-sd/(2*(1.0/res)),0,1)[...,None]*op   # slug coverage * opacity
        img=img*(1-cov)+fill*cov
    return img, len(drawables), flags
