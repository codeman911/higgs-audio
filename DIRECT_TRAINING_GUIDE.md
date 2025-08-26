# Direct Training Usage Guide

## Problem Solved ✅

The over-engineering issue has been fixed. You now have **two clean options**:

## Option 1: Direct Training (Recommended) 🎯

Use your data exactly as-is without any conversion or dummy file creation:

```bash
cd /vs/higgs-audio

python3 launch_training_direct.py \
    --train_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \
    --batch_size 4 \
    --learning_rate 5e-4
```

**What it does**:
- ✅ Uses your input data directly
- ✅ No format conversion
- ✅ No dummy file creation
- ✅ Minimal validation
- ✅ Direct path to training

## Option 2: Enhanced Launcher with Bypass 🔄

Use the enhanced launcher but bypass all conversion:

```bash
cd /vs/higgs-audio

python3 launch_training_with_data_fix.py \
    --input_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \
    --use_input_directly \
    --skip_audio_check \
    --batch_size 4 \
    --learning_rate 5e-4
```

**Flags explained**:
- `--use_input_directly`: Skip all conversion, use your data as-is
- `--skip_audio_check`: Don't check for audio files or create dummy files
- `--skip_validation`: Skip format validation (optional)

## Option 3: Original Scripts 📜

The original training scripts also work if your data is in correct format:

```bash
cd /vs/higgs-audio

bash scripts/launch_8xh200_training.sh \
    --train_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \
    --batch_size 4 \
    --learning_rate 5e-4
```

## Key Changes Made 🔧

1. **Removed Over-Engineering**:
   - Data conversion is now optional and only happens if needed
   - Dummy audio file creation is disabled by default
   - Input data is used directly when possible

2. **Added Bypass Options**:
   - `--use_input_directly`: Skip all conversion
   - `--skip_audio_check`: Skip audio file handling  
   - `--skip_validation`: Skip format validation

3. **Smart Detection**:
   - Automatically detects if your data is already in correct format
   - Only converts if absolutely necessary
   - Preserves original file paths and structure

## Why Was It Creating Dummy Files? 🤔

The original issue was that the pipeline was being overly cautious and:
1. **Always converting data** even when it was already correct
2. **Always checking audio files** and creating dummy ones for missing references
3. **Over-validating** instead of trusting your input data

This has been fixed - the pipeline now respects your data as-is.

## Recommended Approach 💡

For your use case with 653,160 samples that are already in ChatML format, use:

```bash
python3 launch_training_direct.py \
    --train_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \
    --batch_size 4 \
    --learning_rate 5e-4
```

This will:
- ✅ Start training immediately with your data
- ✅ No conversion or dummy file creation
- ✅ Clean, direct pipeline
- ✅ Optimal for large datasets like yours

The training will now work directly with your 653K samples without any unnecessary processing. 🚀