"""D. Empirical oracle: our frames vs (i) real experimental LPTEM, (ii) the Jamali
lab's OWN synthetic frames from the same simulator. Both are external references."""
import glob, zipfile, collections
import numpy as np, cv2
from jepa_lptem.sim.render_bridge import draw_spheres, render_frame


def load(zpath, pred, k=12):
    zf = zipfile.ZipFile(zpath)
    out = []
    for n in [x for x in zf.namelist() if pred(x)][:k]:
        a = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_GRAYSCALE)
        if a is not None:
            out.append(a.astype(float))
    return out


def stats(img):
    """Detector-free descriptors: contrast, and the noise/particle scale separation."""
    bg = np.percentile(img, 90)          # background is the bright majority
    fg = np.percentile(img, 2)           # particles are the dark tail
    contrast = (bg - fg) / (bg + fg + 1e-9)
    # local per-pixel noise: std of the Laplacian, robust to slow illumination drift
    noise = float(np.std(cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)))
    return contrast, noise / (bg + 1e-9)


real = load("data/sam-em/experimentalAuNanorods.zip", lambda x: x.endswith(".jpeg"))
theirsyn = load("data/sam-em/ablationValidationSet.zip",
                lambda x: "/jpeg_images/" in x and x.endswith(".jpg"))
ours = [np.load(f)["frames"] for f in sorted(glob.glob("data/synthetic/smoke/*.npz"))]
ours = [v[len(v) // 2].astype(float) for v in ours]

print(f"{'source':32s} {'n':>3s} {'contrast':>10s} {'rel.noise':>10s}")
for name, imgs in (("REAL experimental Au nanorods", real),
                   ("THEIR synthetic (LPTEMsimulator)", theirsyn),
                   ("OURS (smoke clips)", ours)):
    if not imgs:
        print(f"{name:32s}  -- none loaded --"); continue
    c = np.array([stats(i) for i in imgs])
    print(f"{name:32s} {len(imgs):3d} {c[:,0].mean():10.3f} {c[:,1].mean():10.3f}")

print()
print("PHYSICAL optical depth (pre-normalisation, no noise, our renderer):")
for label, gold in (("upstream flat 9 nm nanorod", 9.0),
                    ("our 20 px sphere, centre", None),
                    ("our 20 px sphere, mean over disk", None)):
    pass
_, g = draw_spheres([[256, 256]], [20.0], (512, 512), pixel_size_nm=2.0)
p = render_frame(g, mtf=None, add_noise=False)
p9 = render_frame(np.where(g > 0, 9.0, 0.0), mtf=None, add_noise=False)
print(f"  background                 {p.max():8.2f}")
print(f"  our sphere centre (80 nm)  {p.min():8.2f}   depth {100*(1-p.min()/p.max()):5.1f}%")
print(f"  flat 9 nm rod (upstream)   {p9.min():8.2f}   depth {100*(1-p9.min()/p9.max()):5.1f}%")
