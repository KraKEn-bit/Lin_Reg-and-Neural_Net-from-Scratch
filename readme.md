# Neural Nets from Scratch (PyTorch)

This repository contains **from-scratch implementations** of **Linear Regression** and a **Feedforward Neural Network** trained on the **Fashion-MNIST dataset**, using PyTorch **autograd only**.

> The implementations are **course-guided**, but written and explored step-by-step to deeply understand forward pass, backpropagation, and gradient descent without relying on high-level abstractions.

---

## 📌 What’s Implemented

-  Linear Regression from scratch  
-  Feedforward Neural Network (manual layers)
-  Manual forward pass
-  Automatic backpropagation using `loss.backward()`
-  Manual weight and bias updates
-  Gradient zeroing and computation graph control using `torch.no_grad()`
-  Model saving and loading

---


## 📂 Project Files<br>
<br>
├── Linear_Regression_From_Scratch_Using_PyTorch.ipynb<br>
├── Fashion_MNIST_NN_using_Pytorch.ipynb<br>
├── Fashion_Model.pth<br>
└── README.md<br>




---

## Dataset:

- **Fashion-MNIST**
- 10 classes of clothing items
- 28×28 grayscale images

---

## Model:

- `Fashion_Model.pth` contains the trained neural network parameters.
- Saved using `torch.save()` after training.
- Can be loaded for inference or continued training.


### Load the saved model

```python
model.load_state_dict(torch.load("Fashion_Model.pth"))
model.eval()
```

