"""Patch geoopt arcosh to prevent NaN at Lorentz manifold boundary.

When embeddings approach the boundary of the hyperboloid, the inner product
in the distance function can produce values <= 1.0 due to floating point
imprecision. This causes arcosh to return NaN, which cascades through all
200M parameters within a single step.

This patch clamps the arcosh input to >= 1.0 + 1e-7, ensuring the function
always returns a valid positive distance.

Usage:
    python patch_geoopt_arcosh.py

Run this ONCE after installing geoopt, before training.
"""
import importlib
import os
import sys

def find_geoopt_math():
    """Find geoopt's lorentz/math.py in the installed packages."""
    try:
        import geoopt
        geoopt_dir = os.path.dirname(geoopt.__file__)
        path = os.path.join(geoopt_dir, "manifolds", "lorentz", "math.py")
        if os.path.exists(path):
            return path
    except ImportError:
        pass

    # Fallback: search common paths
    for base in sys.path:
        path = os.path.join(base, "geoopt", "manifolds", "lorentz", "math.py")
        if os.path.exists(path):
            return path

    return None


def main():
    path = find_geoopt_math()
    if not path:
        print("ERROR: Could not find geoopt/manifolds/lorentz/math.py")
        sys.exit(1)

    print(f"Found: {path}")
    code = open(path).read()

    # Check if already patched
    if "Prevent NaN from manifold boundary" in code:
        print("Already patched — nothing to do.")
        return

    old = "    z = torch.sqrt(torch.clamp_min(x.double().pow(2) - 1.0, 1e-15))"
    new = "    x = torch.clamp_min(x, 1.0 + 1e-7)  # Prevent NaN from manifold boundary\n    z = torch.sqrt(torch.clamp_min(x.double().pow(2) - 1.0, 1e-15))"

    if old in code:
        code = code.replace(old, new, 1)
        open(path, "w").write(code)
        print("SUCCESS: arcosh input clamped at 1.0 + 1e-7")
    else:
        print("ERROR: Could not find target line in arcosh function")
        sys.exit(1)


if __name__ == "__main__":
    main()
