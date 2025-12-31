# Contributing to DRL Wildfire Detection

Thank you for your interest in contributing to this project! 🔥

## 🚀 How to Contribute

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/25-26_HKI_DATN_21021441_ThangLT.git
cd 25-26_HKI_DATN_21021441_ThangLT

# Add upstream remote
git remote add upstream https://github.com/Geospatial-Technology-Lab/25-26_HKI_DATN_21021441_ThangLT.git
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib scipy rasterio tqdm gym scikit-learn
```

### 4. Make Your Changes

- Follow the existing code style
- Add comments for complex logic
- Update documentation if needed

### 5. Test Your Changes

```bash
# Run a quick test with synthetic data
python train_integrated_main.py --algorithm a3c --episodes 10 --use_synthetic

# Or test original version
python a3c/a3c_main.py --episodes 10
```

### 6. Commit and Push

```bash
git add .
git commit -m "feat: Add your feature description"
git push origin feature/your-feature-name
```

### 7. Create Pull Request

- Go to your fork on GitHub
- Click "New Pull Request"
- Describe your changes clearly

---

## 📝 Commit Message Format

Use conventional commits:

| Type | Description |
|------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `refactor:` | Code refactoring |
| `test:` | Adding tests |
| `chore:` | Maintenance |

**Examples:**
```
feat: Add new reward shaping for fire detection
fix: Resolve memory leak in CNN environment
docs: Update README with new CLI options
```

---

## 🐛 Reporting Bugs

Open an issue with:
1. **Description**: What happened?
2. **Expected behavior**: What should happen?
3. **Steps to reproduce**: How to trigger the bug?
4. **Environment**: OS, Python version, GPU

---

## 💡 Feature Requests

Open an issue with:
1. **Description**: What feature do you want?
2. **Use case**: Why is it useful?
3. **Proposed solution**: How would you implement it?

---

## 📁 Project Structure

When adding new algorithms:

```
new_algorithm/
├── new_algorithm.py       # Core algorithm implementation
├── new_algorithm_main.py  # Training script with argparse
└── integrated_new.py      # CNN+ICM integrated version (optional)
```

---

## 🔧 Code Style

- **Python**: Follow PEP 8
- **Docstrings**: Use Google style
- **Type hints**: Recommended for function signatures

```python
def compute_reward(state: np.ndarray, action: int) -> float:
    """
    Compute reward for given state-action pair.
    
    Args:
        state: Current environment state
        action: Action taken by agent
        
    Returns:
        Reward value
    """
    pass
```

---

## 📧 Contact

For questions, contact the maintainer:
- **Le Toan Thang** - toanthangvietduc@gmail.com

---

Thank you for contributing! 🙏
