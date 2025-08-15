# 🚨 HIGGS-AUDIO DEBUG REPORT - CRITICAL REFERENCE
**DO NOT REPEAT THESE MISTAKES**

**Timeline**: Aug 13-15, 2025 | **Issues**: 12 total | **Resolved**: 8 | **Recurring**: 4

---

## ✅ **PERMANENTLY FIXED ISSUES**

### 1. **Tensor Alignment Mismatch** 
- **Problem**: Audio logits `[T,8,V]` vs labels `[8,T]` flattening mismatch
- **Symptom**: Audio loss plateau at 6.933 (random chance)
- **Solution**: `audio_logits.permute(1, 0, 2)` before loss computation
- **Result**: Audio learning restored, loss drops properly

### 2. **Speaker ID Type Error**
- **Problem**: String speaker IDs in zero-shot data cause tensor creation failure
- **Solution**: `numeric_speaker_id = 0 if isinstance(speaker_id, str) else int(speaker_id)`

### 3. **ChatML Field Names**  
- **Problem**: `audio_wv` instead of `audio_waveforms_concat`
- **Solution**: Use correct ChatMLDatasetSample field names

### 4. **Model Parameter Names**
- **Problem**: Generic `labels` parameter not recognized  
- **Solution**: Use `label_ids` and `label_audio_ids` parameters

### 5. **Codebook Alignment**
- **Problem**: Model (12) vs Tokenizer (8) codebook mismatch
- **Solution**: Set `config.audio_num_codebooks = 8`

---

## 🔥 **CRITICAL RECURRING FAILURES**

### 1. **AUDIO TOKEN CONTAMINATION** ⚠️ **BLOCKER**
**Root Cause**: `prepare_chatml_sample` puts audio tokens (`<|audio_eos|>`) in BOTH input AND labels

**Symptoms**:
```
Text predictions: [128012, 128012, 128012, ...]  # All <|audio_eos|>
Text labels:      [22071, 100385, 124492, ...]   # Arabic text  
Text accuracy: 0-2% (should be 15-30%)
```

**Failed Attempts**:
- ❌ Monkey-patch `prepare_chatml_sample` → Breaks audio pipeline (NaN loss)  
- ❌ Custom role separation → Double-processes audio, breaks collator
- ❌ Post-processing masking → Still breaks audio processing

**Status**: UNRESOLVED - Core architectural issue

### 2. **AUDIO PIPELINE FRAGILITY** ⚠️ **CRITICAL**
**Problem**: ANY modification to `prepare_chatml_sample` or audio flow → NaN audio loss

**Fragile Components**:
- `prepare_chatml_sample` function (NEVER modify)
- `audio_contents` format (Expected by collator)  
- `HiggsAudioSampleCollator` parameters (Exact config required)

**Pattern**: Every text contamination fix breaks audio processing

---

## 💥 **RECURRING MISTAKE PATTERNS**

### Pattern #1: **Breaking Audio While Fixing Text** (4+ times)
- **Mistake**: Modifying `prepare_chatml_sample` to fix text contamination
- **Result**: Audio loss becomes NaN, audio learning stops
- **Rule**: **NEVER touch `prepare_chatml_sample` or audio pipeline**

### Pattern #2: **Over-Engineering Solutions** (3+ times)  
- **Mistake**: Custom complex solutions vs using `train_higgs_lora.py`
- **Result**: New failure points, compatibility issues
- **Rule**: **Use proven working code first**

### Pattern #3: **No Rollback Plan** (5+ times)
- **Mistake**: Making changes without maintaining working baseline  
- **Result**: Time wasted re-implementing working solutions
- **Rule**: **Always maintain rollback capability**

---

## 🎯 **CRITICAL RULES FOR AI AGENTS**

### 🚫 **ABSOLUTE DON'Ts**
1. **NEVER modify `prepare_chatml_sample`** - Breaks audio 100% of time
2. **NEVER create custom audio processing** - Use `train_higgs_lora.py` exactly
3. **NEVER assume fixes won't break other systems** - Test text AND audio  
4. **NEVER over-engineer** - Minimal changes only

### ✅ **PROVEN WORKING APPROACH**
1. Use `train_higgs_lora.py` as gold standard
2. Make surgical, minimal changes  
3. Test both pathways simultaneously
4. Roll back immediately on NaN audio loss

---

## 📊 **CURRENT STATUS**

**Working**: Audio pipeline (when unmodified), LoRA setup, data loading  
**Broken**: Text predictions (all `<|audio_eos|>` tokens)  
**Blocker**: Cannot fix text without breaking audio

**Next Steps**: 
1. Establish stable baseline (audio loss numerical)
2. Research alternative approaches that don't break audio pipeline
3. Consider architectural changes to DualFFN vs token filtering

---

## 🔬 **ROOT CAUSE ANALYSIS**

**Core Issue**: ChatML format fundamentally incompatible with DualFFN architecture
- ChatML mixes text/audio tokens by design  
- DualFFN expects clean pathway separation
- No clean solution exists within current architecture

**Possible Solutions**:
- Modify DualFFN to handle mixed tokens (complex)
- Create ChatML-to-DualFFN adapter layer  
- Use different data format aligned with DualFFN
- Implement token-type-aware loss computation

---

**⚠️ FINAL WARNING**: The pattern of breaking audio pipeline while fixing text contamination has repeated 4+ times. Future agents must NOT repeat this cycle. Focus on working baseline first, then explore architectural solutions.
