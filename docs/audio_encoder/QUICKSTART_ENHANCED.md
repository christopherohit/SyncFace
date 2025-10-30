# Quick Start: Enhanced Audio Encoder

Get up and running with the enhanced audio encoder in 5 minutes!

## 🚀 One-Command Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Test with Whisper encoder (recommended)
python main.py \
  --workspace output/quickstart \
  --use_enhanced_encoder \
  --enhanced_encoder_type whisper \
  --use_prosody \
  --O --test
```

That's it! The foundation models will be automatically downloaded on first run.

---

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU with 8GB+ VRAM
- PyTorch 1.13+
- Transformers 4.36+

---

## 🎯 Quick Examples

### Example 1: Basic Whisper Encoder

Most common use case - good quality, fast:

```bash
python main.py \
  --workspace output/whisper_demo \
  --use_enhanced_encoder \
  --enhanced_encoder_type whisper \
  --use_prosody \
  --freeze_audio_backbone \
  --O --test
```

### Example 2: Hybrid Mode (Recommended)

Best balance - maintains lip sync + adds prosody:

```bash
python main.py \
  --workspace output/hybrid_demo \
  --use_enhanced_encoder \
  --enhanced_encoder_type hybrid \
  --foundation_model_type whisper \
  --use_prosody \
  --O --test
```

### Example 3: Low Memory (6GB GPU)

For systems with limited GPU memory:

```bash
python main.py \
  --workspace output/lowmem_demo \
  --use_enhanced_encoder \
  --enhanced_encoder_type whisper \
  --freeze_audio_backbone \
  --batch_size 1 \
  --O --test
```

---

## 🔧 Configuration Files

Use pre-made config files for common scenarios:

```bash
# Basic Whisper
python main.py --config configs/enhanced_audio_config.yaml --config_name basic

# Hybrid mode
python main.py --config configs/enhanced_audio_config.yaml --config_name hybrid

# Maximum quality
python main.py --config configs/enhanced_audio_config.yaml --config_name ensemble
```

---

## 📊 What You Get

The enhanced encoder provides:

| Feature | Before (LRS2) | After (Enhanced) |
|---------|--------------|------------------|
| **Lip Sync** | ✅ Excellent | ✅ Excellent |
| **Prosody** | ❌ None | ✅ Rich (pitch, energy, rhythm) |
| **Emotion** | ❌ None | ✅ Captured |
| **Speaking Style** | ❌ Generic | ✅ Personalized |
| **Languages** | English | 99+ languages |

---

## 🎬 Training Your Own Model

### Step 1: Prepare Data

Place your video in `data/your_character/`:
```
data/
  your_character/
    your_character.mp4
```

### Step 2: Process Data

```bash
python data_utils/process.py --path data/your_character --task 0
python data_utils/process.py --path data/your_character --task 1
python data_utils/process.py --path data/your_character --task 2
```

### Step 3: Train with Enhanced Encoder

```bash
python main.py \
  --workspace output/your_character \
  --use_enhanced_encoder \
  --enhanced_encoder_type whisper \
  --use_prosody \
  --freeze_audio_backbone \
  --iters 100000
```

### Step 4: Test

```bash
python main.py \
  --workspace output/your_character \
  --use_enhanced_encoder \
  --enhanced_encoder_type whisper \
  --aud demo/test.wav \
  --O --test
```

---

## 🐛 Troubleshooting

### Problem: Out of memory

**Solution:**
```bash
# Reduce batch size
python main.py --use_enhanced_encoder --batch_size 1

# Or use smaller model
python main.py --use_enhanced_encoder --enhanced_encoder_type whisper \
  --whisper_model openai/whisper-base
```

### Problem: Models downloading slow

**Solution:**
```bash
# Pre-download models
python -c "from transformers import WhisperModel; WhisperModel.from_pretrained('openai/whisper-small')"

# Or set custom cache directory
export HF_HOME=/fast_disk/cache
```

### Problem: Can't import enhanced encoder

**Solution:**
```bash
# Reinstall dependencies
pip install transformers>=4.36.0 accelerate>=0.20.0
```

---

## 📈 Performance Tips

### Faster Training
- Use `--freeze_audio_backbone` (foundation model stays frozen)
- Use smaller model: `openai/whisper-small` or `openai/whisper-base`
- Increase batch size if you have GPU memory

### Better Quality
- Use `--enhanced_encoder_type ensemble` (slow but best)
- Enable contrastive learning: `--use_contrastive`
- Fine-tune backbone: remove `--freeze_audio_backbone` (slow)

### Lower Memory
- Set `--batch_size 1`
- Use `openai/whisper-tiny` or `openai/whisper-base`
- Disable prosody: remove `--use_prosody`

---

## 🔍 Comparing Models

Quick test to compare original vs enhanced:

```bash
# Original LRS2 encoder
python main.py --workspace output/original --asr_model ave --O --test

# Enhanced Whisper encoder
python main.py --workspace output/enhanced \
  --use_enhanced_encoder \
  --enhanced_encoder_type whisper \
  --O --test

# Compare results visually
```

---

## 📚 Next Steps

- Read full documentation: [ENHANCED_AUDIO_ENCODER.md](ENHANCED_AUDIO_ENCODER.md)
- Try different foundation models
- Experiment with contrastive learning
- Fine-tune on your specific data

---

## 💡 Tips & Best Practices

1. **Start with Whisper**: It's the most robust and well-tested
2. **Use Hybrid for Production**: Best balance of quality and compatibility
3. **Freeze Backbone Initially**: Faster training, good results
4. **Fine-tune if Needed**: Only if you have large dataset and compute
5. **Monitor GPU Memory**: Start with small batches, increase gradually

---

## 🆘 Getting Help

- Check [ENHANCED_AUDIO_ENCODER.md](ENHANCED_AUDIO_ENCODER.md) for detailed docs
- Open an issue on GitHub
- Review example configs in `configs/enhanced_audio_config.yaml`

---

## ✅ Checklist

- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] GPU with 8GB+ VRAM available
- [ ] Data prepared in correct format
- [ ] Config file created or flags set
- [ ] Ran training command
- [ ] Tested with inference

---

Happy talking head generation! 🎭

