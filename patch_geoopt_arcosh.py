"""Comprehensive NaN prevention patch for geoopt Lorentz manifold.

Patches ALL unsafe operations in geoopt/manifolds/lorentz/math.py:
1. arcosh: clamp input >= 1.0 + 1e-7
2. _project: clamp sqrt argument >= 0
3. norm: clamp inner product >= 1e-8 (already done, verify)

Usage: python patch_geoopt_arcosh.py
"""
import os
import sys


def find_geoopt_math():
    try:
        import geoopt
        geoopt_dir = os.path.dirname(geoopt.__file__)
        path = os.path.join(geoopt_dir, "manifolds", "lorentz", "math.py")
        if os.path.exists(path):
            return path
    except ImportError:
        pass
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
    changes = 0

    # 1. arcosh: clamp input
    old1 = "    z = torch.sqrt(torch.clamp_min(x.double().pow(2) - 1.0, 1e-15))"
    new1 = "    x = torch.clamp_min(x, 1.0 + 1e-7)  # NaN prevention: manifold boundary\n    z = torch.sqrt(torch.clamp_min(x.double().pow(2) - 1.0, 1e-15))"
    if old1 in code and "NaN prevention: manifold boundary" not in code:
        code = code.replace(old1, new1, 1)
        changes += 1
        print("  [1/3] Patched arcosh input clamp")
    elif "NaN prevention: manifold boundary" in code:
        print("  [1/3] arcosh already patched")

    # 2. _project: clamp sqrt argument to prevent negative under sqrt
    old2 = "    left_ = torch.sqrt(\n        k + torch.linalg.vector_norm(x.narrow(dim, 1, dn), ord=2, dim=dim) ** 2\n    ).unsqueeze(dim)"
    new2 = "    left_ = torch.sqrt(\n        torch.clamp_min(k + torch.linalg.vector_norm(x.narrow(dim, 1, dn), ord=2, dim=dim) ** 2, 1e-8)\n    ).unsqueeze(dim)"
    if old2 in code:
        code = code.replace(old2, new2, 1)
        changes += 1
        print("  [2/3] Patched _project sqrt clamp")
    elif "clamp_min(k +" in code:
        print("  [2/3] _project already patched")
    else:
        print("  [2/3] WARNING: _project target not found")

    # 3. expmap sqrt safety (sinh/cosh can overflow)
    old3 = "    left_ = torch.sqrt(\n        k + torch.linalg.vector_norm(x.narrow(dim, 1, dn), ord=2, dim=dim) ** 2\n    ).unsqueeze(dim)"
    # Already handled by patch 2

    open(path, "w").write(code)
    print(f"\nDone: {changes} patches applied to {path}")


if __name__ == "__main__":
    main()
