# Neural Nets from Scratch (PyTorch)

This repository contains **from-scratch implementations** of **Linear Regression** and a **Feedforward Neural Network** trained on the **Fashion-MNIST dataset**, using PyTorch **autograd only**.

> The implementations are **course-guided**, but written and explored step-by-step to deeply understand forward pass, backpropagation, and gradient descent without relying on high-level abstractions.

---

## 📌 What’s Implemented

- ✅ Linear Regression from scratch  
- ✅ Feedforward Neural Network (manual layers)
- ✅ Manual forward pass
- ✅ Automatic backpropagation using `loss.backward()`
- ✅ Manual weight and bias updates
- ✅ Gradient zeroing and computation graph control using `torch.no_grad()`
- ✅ Model saving and loading

---

## 🧠 Concepts Learned & Applied

- PyTorch dynamic computation graphs
- Autograd and gradient accumulation
- Mean Squared Error (MSE) loss
- Gradient Descent from first principles
- Why weight updates must not be tracked by autograd
- Model persistence using `.pth` files

---

## 📂 Project Files

├── Linear_Regression_From_Scratch_Using_PyTorch.ipynb
├── Fashion_MNIST_NN_using_Pytorch.ipynb
├── Fashion_Model.pth
└── README.md




---

## 🧪 Dataset

- **Fashion-MNIST**
- 10 classes of clothing items
- 28×28 grayscale images

---

## 💾 Saved Model

- `Fashion_Model.pth` contains the trained neural network parameters.
- Saved using `torch.save()` after training.
- Can be loaded for inference or continued training.

### Load the saved model

```python
model.load_state_dict(torch.load("Fashion_Model.pth"))
model.eval()
```



🚀 How to Run
pip install torch torchvision
jupyter notebook
