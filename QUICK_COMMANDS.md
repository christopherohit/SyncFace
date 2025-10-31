# 🔗 Quick Commands Reference

## 🚀 **Essential Commands** (Copy & Run)

### **1. Data Processing**
```bash
# Process Macron video (audio, images, faces, flow)
python data_utils/process.py data/Macron/Macron.mp4 --task -1 --asr ave
```

### **2. Training Commands**

#### **Basic Training** (Original LRS2)
```bash
python main.py data/Macron --workspace output/basic --asr_model ave --iters 100000 -O
```

#### **Enhanced Training** ⭐ **(RECOMMENDED)**
```bash
python main.py data/Macron --workspace output/enhanced --use_enhanced_encoder --enhanced_encoder_type hybrid --use_prosody --freeze_audio_backbone --asr_model ave --iters 100000 -O
```

#### **Whisper Only**
```bash
python main.py data/Macron --workspace output/whisper --use_enhanced_encoder --enhanced_encoder_type whisper --use_prosody --freeze_audio_backbone --asr_model ave --iters 100000 -O
```

#### **Ensemble (Max Quality)**
```bash
python main.py data/Macron --workspace output/ensemble --use_enhanced_encoder --enhanced_encoder_type ensemble --use_prosody --freeze_audio_backbone --asr_model ave --iters 100000 --batch_size 2 -O
```

### **3. Emotion Recognition**
```bash
# Train emotion model (requires RAVDESS dataset)
python scripts/train_emotion_recognition.py --dataset ravdess --data_path /path/to/RAVDESS --emotion_model wav2vec2 --use_pretrained --epochs 50 --output_dir output/emotion

# Full emotion-aware training
python main.py data/Macron --workspace output/emotion --use_enhanced_encoder --enhanced_encoder_type hybrid --use_emotion --emotion_checkpoint output/emotion/best_emotion_model.pth --asr_model ave --iters 100000 -O
```

### **4. Testing Commands**
```bash
# Test trained model
python main.py data/Macron --workspace output/enhanced --test -O

# GUI mode (interactive)
python main.py data/Macron --workspace output/enhanced --test --gui -O

# Test on training data
python main.py data/Macron --workspace output/enhanced --test --test_train -O
```

### **5. Utility Commands**
```bash
# Interactive training menu
python train_macron.py

# Test network refactoring
python test_network_refactor.py

# Compare encoders
python scripts/test_enhanced_encoder.py --compare all --audio demo/test.wav

# Check GPU memory
nvidia-smi
```

---

## 📋 **Command Categories**

### 🔧 **Data Processing**
- Extract audio, images, face landmarks, optical flow
- Multiple ASR model support (AVE, HuBERT, DeepSpeech)

### 🎯 **Training Modes**
- **Basic**: Original LRS2 encoder
- **Enhanced**: Foundation models (Whisper, SpeechT5, EnCodec)
- **Hybrid**: LRS2 + Foundation models (best balance)
- **Ensemble**: Multiple models (maximum quality)

### 😊 **Emotion Integration**
- Train emotion recognition on RAVDESS dataset
- Add emotion-sensitive facial expressions
- Multiple emotion models (Wav2Vec2, CNN, Prosody)

### 🧪 **Testing & Validation**
- Model inference and video generation
- GUI for real-time interaction
- Evaluation metrics (PSNR, LPIPS, LMD)
- Memory and performance monitoring

### 🎮 **Interactive Modes**
- GUI for real-time control
- ASR real-time mode
- Torso training support

---

## 🎨 **Recommended Workflows**

### **For Beginners** (Quick Start)
```bash
# 1. Process data
python data_utils/process.py data/Macron/Macron.mp4 --task -1 --asr ave

# 2. Train (recommended setup)
python main.py data/Macron --workspace output/macron_final --use_enhanced_encoder --enhanced_encoder_type hybrid --use_prosody --asr_model ave --iters 100000 -O

# 3. Test
python main.py data/Macron --workspace output/macron_final --test -O
```

### **For Researchers** (Maximum Quality)
```bash
# 1. Train emotion model
python scripts/train_emotion_recognition.py --dataset ravdess --data_path /path/to/RAVDESS --emotion_model wav2vec2 --output_dir output/emotion

# 2. Full training with emotion
python main.py data/Macron --workspace output/max_quality --use_enhanced_encoder --enhanced_encoder_type ensemble --use_prosody --use_contrastive --use_emotion --emotion_checkpoint output/emotion/best_emotion_model.pth --asr_model ave --iters 200000 --batch_size 2 -O
```

### **For Production** (Balanced)
```bash
# Fast, high-quality, deployable
python main.py data/Macron --workspace output/production --use_enhanced_encoder --enhanced_encoder_type hybrid --use_prosody --freeze_audio_backbone --asr_model ave --iters 100000 -O
```

---

## ⚙️ **Configuration Options**

### **Memory Settings**
```bash
# Low memory (6GB GPU)
--batch_size 1 --enhanced_encoder_type whisper

# High memory (24GB GPU)
--batch_size 4 --enhanced_encoder_type ensemble
```

### **Quality Settings**
```bash
# Speed priority
--iters 50000 --enhanced_encoder_type whisper

# Quality priority
--iters 200000 --enhanced_encoder_type ensemble --use_contrastive
```

### **Emotion Settings**
```bash
# Subtle emotions
--emotion_strength 0.3 --emotion_smoothing ema

# Strong emotions
--emotion_strength 1.0 --emotion_smoothing conv
```

---

## 🚨 **Common Issues & Fixes**

### **Out of Memory**
```bash
# Reduce batch size
--batch_size 1

# Use smaller model
--enhanced_encoder_type whisper

# Use hybrid
--enhanced_encoder_type hybrid
```

### **Missing Dependencies**
```bash
# Install requirements
pip install transformers accelerate torchaudio librosa

# Check installation
python -c "import transformers; print('OK')"
```

### **Model Not Found**
```bash
# Pre-download models
python -c "from transformers import WhisperModel; WhisperModel.from_pretrained('openai/whisper-small')"
```

---

## 📊 **Expected Results**

### **Training Metrics** (After 100k iterations)
- **PSNR**: 35-40 dB (higher = better)
- **LPIPS**: 0.02-0.08 (lower = better)
- **LMD**: 2.0-3.0 pixels (lower = better)

### **Performance**
- **Basic**: ~20M params, 6GB VRAM
- **Enhanced**: ~250M params, 8GB VRAM
- **Ensemble**: ~500M params, 12GB VRAM

---

## 🎯 **Quick Copy-Paste Commands**

**Just copy and run:**

```bash
# Best overall (hybrid)
python main.py data/Macron --workspace output/macron_best --use_enhanced_encoder --enhanced_encoder_type hybrid --use_prosody --freeze_audio_backbone --asr_model ave --iters 100000 -O

# Test it
python main.py data/Macron --workspace output/macron_best --test -O

# GUI mode
python main.py data/Macron --workspace output/macron_best --test --gui -O
```

**That's it!** 🎉

---

*Generated: 2025 - SyncFace Enhanced Commands Reference*
