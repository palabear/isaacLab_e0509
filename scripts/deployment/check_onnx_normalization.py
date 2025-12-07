#!/usr/bin/env python3
"""
Check normalization parameters in ONNX model
"""

import onnx
import numpy as np

onnx_path = '/home/jiwoo/IsaacLab/logs/rsl_rl/e0509_pick_place/2025-12-05_09-11-46/exported/policy.onnx'

print(f"Loading ONNX model: {onnx_path}")
model = onnx.load(onnx_path)

print("\n" + "="*70)
print("ONNX Model Information")
print("="*70)

print("\nInputs:")
for inp in model.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {shape}")

print("\nOutputs:")
for out in model.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {shape}")

print("\nInitializers (looking for normalization parameters):")
obs_mean = None
obs_std = None

for init in model.graph.initializer:
    if 'mean' in init.name.lower() or 'std' in init.name.lower() or 'div' in init.name.lower() or 'normalizer' in init.name.lower():
        data = onnx.numpy_helper.to_array(init).flatten()
        print(f"\n  {init.name}:")
        print(f"    Shape: {init.dims}")
        print(f"    Length: {len(data)}")
        print(f"    First 10 values: {data[:10]}")
        print(f"    Min: {data.min():.6f}, Max: {data.max():.6f}, Mean: {data.mean():.6f}")
        
        if "mean" in init.name.lower():
            obs_mean = data
        elif "div" in init.name.lower() or "std" in init.name.lower():
            obs_std = data

print("\n" + "="*70)
if obs_mean is not None:
    print(f"✅ Found observation mean (shape: {obs_mean.shape})")
    print(f"   Sample: {obs_mean[:5]}")
else:
    print("❌ Observation mean NOT FOUND")

if obs_std is not None:
    print(f"✅ Found observation std (shape: {obs_std.shape})")
    print(f"   Sample: {obs_std[:5]}")
else:
    print("❌ Observation std NOT FOUND")

print("\nAll initializer names:")
for init in model.graph.initializer:
    print(f"  - {init.name} (shape: {init.dims})")
