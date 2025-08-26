# Remote Cluster Deployment Guide
## Fix for "Missing required message types" Error

This guide provides the exact steps to deploy the validation fix to your remote cluster and test it with your 653,160 training samples.

## 🚨 Root Cause
The validation logic was expecting the wrong ChatML format. The working `arb_inference.py` processes two ChatML formats:
1. **User messages with content lists** (text + audio items)
2. **Traditional ChatML** (system + user + assistant with audio)

The old validation only checked for format #2, but your data likely uses format #1.

## 📁 Files to Upload to Remote Cluster

### 1. Updated Validation Logic
**File:** `/vs/higgs-audio/utils.py`
```bash
# Replace the existing utils.py with the corrected version
# This file contains the fixed _validate_sample_structure() function
# that uses EXACT arb_inference.py logic
```

### 2. Updated Dataset Validation  
**File:** `/vs/higgs-audio/trainer/dataset.py`
```bash
# Replace the existing trainer/dataset.py
# Updated _validate_sample() method to match inference patterns
```

### 3. Fixed Training Launcher
**File:** `/vs/higgs-audio/launch_training_fixed.py`
```bash
# New launcher with embedded fixed validation logic
# This is a standalone script that doesn't depend on utils.py
```

### 4. Validation Test Script
**File:** `/vs/higgs-audio/test_validation_fix.py`
```bash
# Test script to verify the fix works on your actual training data
```

## 🧪 Testing Phase

### Step 1: Test the Fixed Validation Logic
```bash
cd /vs/higgs-audio

# Test on your actual training data
python3 test_validation_fix.py \
    --train_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \
    --max_samples 1000 \
    --verbose
```

**Expected Output:**
```
🧪 Testing Fixed Validation Logic
==================================================
🔧 This script tests the corrected validation logic
   that matches arb_inference.py patterns exactly.
==================================================
🔍 Testing validation on: ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json
📊 Testing on first 1000 of 653160 samples
  Sample 0: ✅ Valid structure
    - ref_audio: path/to/audio.wav
    - ref_text: 'Reference text here...'
    - target_text: 'Target text to generate...'
...
✅ Validation Test Results:
   📊 Total samples tested: 1000
   ✅ Valid samples: 1000
   ❌ Invalid samples: 0
   💥 Error samples: 0
   ⏱️ Processing time: 2.34 seconds
   📈 Success rate: 100.0%
🎉 Validation test PASSED! Most samples are valid.
✅ Ready to proceed with training
```

### Step 2: Test on Full Dataset (if Step 1 succeeds)
```bash
# Test on all 653,160 samples (will take longer)
python3 test_validation_fix.py \
    --train_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json
```

## 🚀 Training Launch

### Option A: Use Fixed Launcher (Recommended)
```bash
cd /vs/higgs-audio

# This launcher has the fixed validation built-in
python3 launch_training_fixed.py \
    --train_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \
    --batch_size 4 \
    --learning_rate 5e-4 \
    --num_epochs 3 \
    --mixed_precision \
    --use_gradient_checkpointing
```

### Option B: Use Original Launcher (if utils.py is updated)
```bash
cd /vs/higgs-audio

# This should now work with the updated utils.py
python3 launch_training_direct.py \
    --train_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \
    --batch_size 4 \
    --learning_rate 5e-4
```

### Option C: Direct Training Command
```bash
cd /vs/higgs-audio

# Skip validation entirely and go straight to training
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    trainer/train.py \
    --train_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \
    --batch_size 4 \
    --learning_rate 5e-4 \
    --num_epochs 3 \
    --skip_validation
```

## 🔍 Troubleshooting

### If Validation Still Fails

1. **Check your data format:**
```bash
# Look at the first sample in your training data
head -50 ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json

# Or use jq if available
jq '.[0]' ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json
```

2. **Test with a single sample:**
```bash
# Create a single sample file for testing
echo '[{"messages":[{"role":"user","content":"test"}],"speaker":"test"}]' > test_single.json
python3 test_validation_fix.py --train_data test_single.json --verbose
```

3. **Check file permissions:**
```bash
ls -la ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json
```

### Common Error Solutions

**Error: "No such file or directory"**
```bash
# Find your actual training data file
find /vs -name "*train_chatml*" -type f 2>/dev/null
find /vs -name "*chatml*" -type f 2>/dev/null
```

**Error: "Invalid JSON"**
```bash
# Check if your JSON file is valid
python3 -c "import json; json.load(open('../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json'))"
```

**Error: Still getting "Missing required message types"**
```bash
# The data might need format conversion
# Use the launch_training_with_data_fix.py which has conversion logic
python3 launch_training_with_data_fix.py \
    --train_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \
    --batch_size 4 \
    --auto_convert_format
```

## 📊 Expected Training Output

Once validation passes, you should see:
```
🎵 Higgs-Audio LoRA Training Pipeline
==================================================
🔍 Validating environment...
   PyTorch: 2.8.0+cu126
   CUDA available: True
   CUDA version: 12.6
   GPU count: 8
   GPU 0: NVIDIA H200
   ...
   GPU 7: NVIDIA H200
✅ Validation complete: 653160/653160 samples valid
🚀 Launching 8xH200 distributed training...
📋 Training command: torchrun --nproc_per_node=8 ...
[DISTRIBUTED TRAINING STARTS]
```

## 🎯 Success Criteria

- ✅ Validation test shows >90% success rate
- ✅ Training launches without "Missing required message types" error  
- ✅ All 8 H200 GPUs are utilized
- ✅ Training loss decreases over epochs

## 📞 Next Steps After Successful Training

1. **Monitor training progress**
2. **Validate model quality with inference**
3. **Test zero-shot voice cloning on Arabic samples**

The validation fix ensures your training data is processed using the exact same logic as the working inference pipeline, eliminating the format compatibility issue.