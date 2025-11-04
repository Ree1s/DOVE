# Token Merge Configuration Guide

## Overview

Token merge is an optimization technique that reduces the number of tokens processed in the transformer by intelligently merging similar tokens during the forward pass. This can significantly reduce memory usage and computation cost while maintaining quality.

## Configuration Parameters

### `--enable_token_merge` (bool)
- **Default**: `false`
- **Description**: Master switch to enable/disable token merging
- **Example**: `--enable_token_merge true`

### `--token_merge_routes` (string)
- **Default**: `None`
- **Description**: Semicolon-separated routing specifications defining which layers to apply token merging
- **Format**: `"start-end@ratio;start-end@ratio;..."`
  - `start`: First layer index to start merging (0-indexed)
  - `end`: Last layer index to end merging and restore tokens
  - `ratio`: Fraction of tokens to remove (0.0 = no reduction, 1.0 = remove all)
- **Examples**:
  - Single route: `"10-15@0.3"` - Merge 30% of tokens from layer 10 to 15
  - Multiple routes: `"5-10@0.2;15-20@0.4"` - Two sequential routing windows
  - Conservative: `"10-20@0.15"` - Remove only 15% of tokens
  - Aggressive: `"5-25@0.5"` - Remove 50% of tokens (not recommended initially)

### `--token_merge_default_ratio` (float)
- **Default**: `0.0`
- **Description**: Default merge ratio applied to all layers when `--token_merge_routes` is not specified
- **Range**: `0.0` (disabled) to `1.0` (remove all tokens)
- **Example**: `--token_merge_default_ratio 0.25` - Apply 25% reduction across all 30 layers

### `--token_merge_seed` (int)
- **Default**: `42`
- **Description**: Random seed for stochastic token selection. Use the same seed for reproducibility
- **Example**: `--token_merge_seed 12345`

### `--token_merge_restore_adapter_expansion` (int)
- **Default**: `2`
- **Description**: Expansion factor for the RestoreAdapter MLP that refines restored tokens
- **Example**: `--token_merge_restore_adapter_expansion 4` - Larger adapter (more capacity)

## Usage Examples

### Example 1: Baseline (No Token Merge)
```bash
# Run original Stage 2 training without any modifications
accelerate launch train.py \
    --model_name "dove-s2" \
    --enable_token_merge false \
    # ... other args ...
```

### Example 2: Conservative Token Merge
```bash
# Good starting point: 20% reduction in middle layers
accelerate launch train.py \
    --model_name "dove-s2" \
    --enable_token_merge true \
    --token_merge_routes "10-20@0.2" \
    --token_merge_seed 42 \
    # ... other args ...
```

### Example 3: Multiple Routing Windows
```bash
# Advanced: Different ratios for different layer ranges
accelerate launch train.py \
    --model_name "dove-s2" \
    --enable_token_merge true \
    --token_merge_routes "5-10@0.15;15-25@0.3" \
    --token_merge_seed 42 \
    # ... other args ...
```

### Example 4: Global Default Ratio
```bash
# Apply same ratio to all layers (simple but less flexible)
accelerate launch train.py \
    --model_name "dove-s2" \
    --enable_token_merge true \
    --token_merge_default_ratio 0.25 \
    --token_merge_seed 42 \
    # ... other args ...
```

### Example 5: Aggressive Reduction (Experimental)
```bash
# ⚠️ Warning: High ratios may degrade quality
accelerate launch train.py \
    --model_name "dove-s2" \
    --enable_token_merge true \
    --token_merge_routes "8-22@0.5" \
    --token_merge_seed 42 \
    # ... other args ...
```

## Best Practices

### 1. Start Conservative
- Begin with **ratios ≤ 0.3** (30% reduction)
- Target middle layers (e.g., 10-20 for a 30-layer model)
- Avoid merging in the first few layers or last few layers

### 2. Gradual Experimentation
```bash
# Step 1: Baseline
--enable_token_merge false

# Step 2: No-op test (verify no regression)
--enable_token_merge true --token_merge_default_ratio 0.0

# Step 3: Small ratio
--enable_token_merge true --token_merge_routes "10-15@0.15"

# Step 4: Moderate ratio
--enable_token_merge true --token_merge_routes "10-20@0.25"

# Step 5: Multiple windows
--enable_token_merge true --token_merge_routes "5-10@0.2;15-25@0.3"
```

### 3. Monitor These Metrics
- **Training loss**: Should remain stable (slight increase is acceptable)
- **Validation metrics**: PSNR, SSIM should not degrade significantly
- **Memory usage**: Should decrease proportionally to merge ratio
- **Training speed**: May improve due to fewer tokens

### 4. Layer Selection Guidelines
For CogVideoX-5B (30 layers):
- **Early layers (0-8)**: Avoid merging; these extract low-level features
- **Middle layers (9-21)**: Safe to merge; good ratio/quality tradeoff
- **Late layers (22-29)**: Avoid merging; critical for final refinement

### 5. Ratio Guidelines
| Ratio | Description | Use Case |
|-------|-------------|----------|
| 0.0 - 0.15 | Conservative | Production, minimal risk |
| 0.15 - 0.3 | Moderate | Good balance, recommended |
| 0.3 - 0.5 | Aggressive | Experimentation, memory-constrained |
| 0.5+ | Very aggressive | Not recommended, quality degradation likely |

## Validation Workflow

### Step 1: Baseline Test
```bash
# Run 100-500 steps without token merge to establish baseline loss
bash finetune/train_ddp_one_s2_debug.sh
# Note the final training loss and validation metrics
```

### Step 2: No-Op Test
```bash
# Verify routing infrastructure doesn't affect baseline
# Add to your training script:
TOKEN_MERGE_ARGS=(
    --enable_token_merge true
    --token_merge_default_ratio 0.0
)
# Loss should match baseline
```

### Step 3: Small Ratio Test
```bash
# Test actual token merging with conservative settings
TOKEN_MERGE_ARGS=(
    --enable_token_merge true
    --token_merge_routes "12-18@0.2"
)
# Loss should increase by <5%, memory should decrease
```

### Step 4: Compare and Iterate
- Compare loss curves between baseline and merged runs
- Check validation metrics (PSNR, SSIM, LPIPS)
- Adjust ratio or layer ranges based on results

## Expected Behavior

### Memory Savings
- **20% merge ratio**: ~15-18% memory reduction
- **30% merge ratio**: ~22-27% memory reduction
- **50% merge ratio**: ~35-45% memory reduction

### Quality Impact
- **<20% ratio**: Minimal to no quality loss
- **20-30% ratio**: Slight quality loss (<1 dB PSNR typically acceptable)
- **>40% ratio**: Noticeable quality degradation

### Training Speed
- Depends on memory bottleneck and batch size
- May enable larger batch sizes → faster convergence
- Per-step time may improve slightly due to fewer tokens

## Troubleshooting

### Warning: "Routing will be a no-op"
**Problem**: Token merge enabled but no routes configured
```bash
# ❌ Bad
--enable_token_merge true
# (no routes specified, default_ratio=0.0)

# ✅ Good
--enable_token_merge true --token_merge_routes "10-15@0.3"
```

### Warning: "Very aggressive ratio"
**Problem**: Ratio >70% may degrade quality
```bash
# ⚠️ Risky
--token_merge_routes "10-20@0.8"

# ✅ Safer
--token_merge_routes "10-20@0.3"
```

### Loss increases significantly
**Solution**: Reduce merge ratio or narrow layer range
```bash
# If loss increases >10%, try:
--token_merge_routes "12-18@0.15"  # Smaller ratio
# or
--token_merge_routes "14-16@0.3"   # Fewer layers
```

### NaN or Inf in outputs
**Solution**: Check RestoreAdapter initialization or disable routing
```bash
# Should not happen with proper initialization
# If it does, report as a bug and disable routing temporarily
--enable_token_merge false
```

## Integration with Existing Scripts

### Minimal Changes to `train_ddp_one_s2_debug.sh`

Add this section after line 95:

```bash
# Token Merge Configuration (optional)
TOKEN_MERGE_ARGS=(
    --enable_token_merge true
    --token_merge_routes "10-20@0.25"
    --token_merge_seed 42
)

# Then add to launch command (line 98):
accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${TOKEN_MERGE_ARGS[@]}" \  # ← Add this line
    "${DATA_ARGS[@]}" \
    # ... rest of args ...
```

### To disable routing, just comment out or set:
```bash
TOKEN_MERGE_ARGS=(
    --enable_token_merge false
)
```

## Advanced Topics

### Custom RestoreAdapter Capacity
Increase adapter capacity for better restoration quality:
```bash
--token_merge_restore_adapter_expansion 4  # Default is 2
```
This increases RestoreAdapter parameters from ~14.7M to ~29.5M but may improve quality.

### Reproducibility
Always use the same seed for reproducible token selection:
```bash
--token_merge_seed 42  # Use consistent seed across runs
```

### Layer-Specific Strategies
```bash
# Strategy 1: Focus on middle layers
--token_merge_routes "10-20@0.3"

# Strategy 2: Multiple small windows
--token_merge_routes "8-12@0.2;16-20@0.2;24-28@0.2"

# Strategy 3: Gradual reduction
--token_merge_routes "5-8@0.1;10-15@0.2;18-24@0.3"
```

## FAQ

**Q: Does token merge affect the checkpoint size?**
A: Yes, minimally. Adds ~14.7M parameters (RestoreAdapter) = ~59 MB with fp16.

**Q: Can I change routing config when resuming from checkpoint?**
A: Yes, routing is controlled by CLI args, not saved in the checkpoint.

**Q: Does this work with LoRA training?**
A: Yes, token merge is compatible with LoRA adapters.

**Q: Should I use token merge for inference?**
A: It can reduce inference memory, but quality impact should be validated first.

**Q: What if I want different ratios per layer?**
A: Use multiple routing windows: `"10-12@0.2;13-15@0.3;16-18@0.4"`

## Performance Benchmarks

*Coming soon: Add benchmarks for memory, speed, and quality on standard datasets*

## References

- ToMe (Token Merging): [Bolya et al., 2023](https://arxiv.org/abs/2210.09461)
- LIEM (Learned Importance-based Token Merging): Similar concept adapted for video diffusion

---

For issues or questions, please refer to the main DOVE documentation or open a GitHub issue.
