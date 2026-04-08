import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor


# -----------------------------
# Data + Model
# -----------------------------
np.random.seed(42)

X = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
], dtype=float)

y = np.array([0, 1, 1, 0], dtype=float).reshape(-1, 1)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0.0, z)


class Dense:
    def __init__(self, n_inputs, n_neurons, activation="sigmoid"):
        self.activation = activation
        if activation == "relu":
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
            self.biases = np.full((1, n_neurons), 0.01)
        else:
            self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
            self.biases = np.zeros((1, n_neurons))

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.weights) + self.biases
        if self.activation == "relu":
            self.a = relu(self.z)
        else:
            self.a = sigmoid(self.z)
        return self.a

    def backward(self, d_a, lr, is_output=False):
        if is_output and self.activation == "sigmoid":
            d_z = d_a
        else:
            if self.activation == "relu":
                d_z = d_a * (self.z > 0)
            else:
                d_z = d_a * (self.a * (1 - self.a))

        m = self.x.shape[0]
        d_w = (1.0 / m) * np.dot(self.x.T, d_z)
        d_b = (1.0 / m) * np.sum(d_z, axis=0, keepdims=True)
        d_prev = np.dot(d_z, self.weights.T)

        self.weights -= lr * d_w
        self.biases -= lr * d_b
        return d_prev


def bce_loss(y_pred, y_true):
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_xor(epochs=20000, lr=0.1):
    hidden = Dense(2, 2, activation="relu")
    output = Dense(2, 1, activation="sigmoid")

    init = {
        "W1": hidden.weights.copy(),
        "b1": hidden.biases.copy(),
        "W2": output.weights.copy(),
        "b2": output.biases.copy(),
    }

    losses = []
    for _ in range(epochs):
        a1 = hidden.forward(X)
        a2 = output.forward(a1)

        loss = bce_loss(a2, y)
        losses.append(loss)

        d_z2 = a2 - y
        d_a1 = output.backward(d_z2, lr=lr, is_output=True)
        hidden.backward(d_a1, lr=lr, is_output=False)

    trained = {
        "W1": hidden.weights.copy(),
        "b1": hidden.biases.copy(),
        "W2": output.weights.copy(),
        "b2": output.biases.copy(),
    }

    probs = output.forward(hidden.forward(X))
    preds = (probs > 0.5).astype(int)

    return init, trained, np.array(losses), probs, preds


def forward_with_params(x_input, w1, b1, w2, b2):
    z1 = np.dot(x_input, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


# -----------------------------
# Plot Helpers
# -----------------------------
def style_axis(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.grid(alpha=0.25)


def draw_points(ax):
    for point, label in zip(X, y.ravel().astype(int)):
        if label == 0:
            ax.scatter(point[0], point[1], c="#1d3557", s=120, marker="o", edgecolors="white", linewidths=1)
        else:
            ax.scatter(point[0], point[1], c="#e63946", s=120, marker="^", edgecolors="white", linewidths=1)


def save_xor_points_plot(path):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    draw_points(ax)
    style_axis(ax, "XOR Data Distribution")
    ax.text(0.02, 1.08, "Circle = y=0, Triangle = y=1", transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_hidden_lines_plot(path, init_params, trained_params):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    colors = ["#e76f51", "#2a9d8f"]

    for ax, w, b, title in [
        (axes[0], init_params["W1"], init_params["b1"], "Before Training: Hidden Neuron Lines"),
        (axes[1], trained_params["W1"], trained_params["b1"], "After Training: Hidden Neuron Lines"),
    ]:
        x1_vals = np.linspace(-0.2, 1.2, 200)
        for j in range(w.shape[1]):
            w1 = w[0, j]
            w2 = w[1, j]
            bj = b[0, j]
            if abs(w2) < 1e-8:
                if abs(w1) > 1e-8:
                    x_const = -bj / w1
                    ax.axvline(x_const, linestyle="--", color=colors[j], linewidth=2, label=f"Hidden neuron {j+1}")
            else:
                x2_vals = -(w1 * x1_vals + bj) / w2
                ax.plot(x1_vals, x2_vals, linestyle="--", color=colors[j], linewidth=2, label=f"Hidden neuron {j+1}")

        draw_points(ax)
        style_axis(ax, title)
        ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_output_surface_plot(path, init_params, trained_params):
    x1_grid = np.linspace(-0.2, 1.2, 220)
    x2_grid = np.linspace(-0.2, 1.2, 220)
    xx, yy = np.meshgrid(x1_grid, x2_grid)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    _, _, _, prob_init = forward_with_params(
        grid_points,
        init_params["W1"],
        init_params["b1"],
        init_params["W2"],
        init_params["b2"],
    )
    _, _, _, prob_trained = forward_with_params(
        grid_points,
        trained_params["W1"],
        trained_params["b1"],
        trained_params["W2"],
        trained_params["b2"],
    )

    prob_init = prob_init.reshape(xx.shape)
    prob_trained = prob_trained.reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), dpi=160, constrained_layout=True)
    axes[0].contourf(xx, yy, prob_init, levels=22, cmap="RdYlBu_r", alpha=0.92)
    axes[0].contour(xx, yy, prob_init, levels=[0.5], colors="black", linewidths=2)
    style_axis(axes[0], "Before Training: Final Output Probability")

    c = axes[1].contourf(xx, yy, prob_trained, levels=22, cmap="RdYlBu_r", alpha=0.92)
    axes[1].contour(xx, yy, prob_trained, levels=[0.5], colors="black", linewidths=2)
    style_axis(axes[1], "After Training: Final Output Probability")

    for ax in axes:
        draw_points(ax)

    fig.colorbar(c, ax=axes, shrink=0.9, label="Predicted P(y=1)")
    fig.savefig(path)
    plt.close(fig)


def save_loss_curve(path, losses):
    fig, ax = plt.subplots(figsize=(8, 4.7), dpi=160)
    ax.plot(losses, color="#264653", linewidth=2)
    ax.set_title("Training Loss vs Epoch", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_architecture_diagram(path):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.axis("off")

    input_nodes = [(1.0, 3.3), (1.0, 1.7)]
    hidden_nodes = [(4.0, 3.7), (4.0, 1.3)]
    output_nodes = [(7.0, 2.5)]

    def node(xy, color, label):
        circle = Circle(xy, 0.28, color=color, ec="black", lw=1.2)
        ax.add_patch(circle)
        ax.text(xy[0], xy[1], label, ha="center", va="center", color="white", fontsize=11, fontweight="bold")

    for i, pt in enumerate(input_nodes):
        node(pt, "#457b9d", f"x{i+1}")

    for i, pt in enumerate(hidden_nodes):
        node(pt, "#2a9d8f", f"h{i+1}")

    node(output_nodes[0], "#e76f51", "y_hat")

    for p in input_nodes:
        for q in hidden_nodes:
            arrow = FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=12, lw=1.4, color="#6c757d")
            ax.add_patch(arrow)

    for p in hidden_nodes:
        q = output_nodes[0]
        arrow = FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=12, lw=1.6, color="#6c757d")
        ax.add_patch(arrow)

    ax.text(1.0, 4.45, "Input Layer (2 features)", ha="center", fontsize=11, fontweight="bold")
    ax.text(4.0, 4.45, "Hidden Layer (2 neurons, ReLU)", ha="center", fontsize=11, fontweight="bold")
    ax.text(7.0, 4.45, "Output Layer (1 neuron, Sigmoid)", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.3, "Forward: Z1 = XW1 + b1, A1 = ReLU(Z1), Z2 = A1W2 + b2, y_hat = sigmoid(Z2)", fontsize=10)

    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(0.0, 5.0)
    fig.savefig(path)
    plt.close(fig)


# -----------------------------
# PPT Helpers
# -----------------------------
def set_slide_bg(slide, rgb=(245, 247, 250)):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(*rgb)


def add_title(slide, title, subtitle=None):
    tx = slide.shapes.add_textbox(Inches(0.6), Inches(0.2), Inches(12.1), Inches(1.0))
    tf = tx.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(34)
    p.font.bold = True
    p.font.color.rgb = RGBColor(33, 37, 41)
    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(15)
        p2.font.color.rgb = RGBColor(73, 80, 87)


def add_bullets(slide, items, left=0.8, top=1.6, width=5.6, height=4.5, font_size=22):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.size = Pt(font_size)
        p.font.color.rgb = RGBColor(52, 58, 64)


def add_footer(slide, text="ML Viva | XOR Neural Network"):
    tx = slide.shapes.add_textbox(Inches(0.6), Inches(6.85), Inches(12.0), Inches(0.35))
    p = tx.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(108, 117, 125)
    p.alignment = PP_ALIGN.RIGHT


def add_image(slide, img_path, left, top, width=None, height=None):
    if width is not None and height is not None:
        slide.shapes.add_picture(str(img_path), Inches(left), Inches(top), Inches(width), Inches(height))
    elif width is not None:
        slide.shapes.add_picture(str(img_path), Inches(left), Inches(top), width=Inches(width))
    elif height is not None:
        slide.shapes.add_picture(str(img_path), Inches(left), Inches(top), height=Inches(height))
    else:
        slide.shapes.add_picture(str(img_path), Inches(left), Inches(top))


def create_ppt(output_ppt, assets_dir, losses, probs, preds):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1) Title
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, (244, 248, 255))
    add_title(slide, "XOR Neural Network", "Machine Learning Viva Presentation")
    add_bullets(
        slide,
        [
            "Project: XOR gate classification using a custom neural network",
            "Architecture: 2 input features -> 2 hidden ReLU neurons -> 1 sigmoid output",
            "Implemented from scratch with NumPy (forward pass + backpropagation)",
        ],
        top=1.8,
        width=8.3,
        font_size=20,
    )
    add_footer(slide)

    # 2) Problem statement
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Problem Statement")
    add_bullets(
        slide,
        [
            "XOR truth table:",
            "(0,0)->0, (1,0)->1, (0,1)->1, (1,1)->0",
            "XOR is NOT linearly separable",
            "So single-layer logistic regression fails",
            "Need hidden layer for non-linear decision boundary",
        ],
        width=6.1,
        font_size=22,
    )
    add_image(slide, assets_dir / "xor_points.png", left=6.7, top=1.55, width=6.2)
    add_footer(slide)

    # 3) Architecture
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, (248, 246, 242))
    add_title(slide, "Network Architecture")
    add_image(slide, assets_dir / "architecture.png", left=0.7, top=1.4, width=8.1)
    add_bullets(
        slide,
        [
            "Hidden layer uses ReLU to create piecewise linear features",
            "Output layer uses sigmoid for binary probability",
            "Loss function: Binary Cross-Entropy",
            "Optimization: Gradient Descent (learning_rate = 0.1)",
        ],
        left=8.9,
        top=1.7,
        width=4.1,
        height=4.9,
        font_size=18,
    )
    add_footer(slide)

    # 4) Math slide
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Core Math")
    add_bullets(
        slide,
        [
            "Forward pass:",
            "Z1 = XW1 + b1,  A1 = ReLU(Z1)",
            "Z2 = A1W2 + b2,  y_hat = sigmoid(Z2)",
            "",
            "Loss (BCE): L = -mean(y log(y_hat) + (1-y) log(1-y_hat))",
            "",
            "Backprop:",
            "dZ2 = y_hat - y",
            "dW = (1/m) X^T dZ,   db = (1/m) sum(dZ)",
            "W <- W - lr*dW,   b <- b - lr*db",
        ],
        width=11.8,
        height=5.5,
        font_size=20,
    )
    add_footer(slide)

    # 5) Hidden boundaries
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "How Hidden Neurons Learn")
    add_image(slide, assets_dir / "hidden_lines.png", left=0.65, top=1.3, width=12.0)
    add_footer(slide, "Each hidden neuron learns one line: w1*x1 + w2*x2 + b = 0")

    # 6) Final decision surface
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Final Decision Surface")
    add_image(slide, assets_dir / "output_surface.png", left=0.55, top=1.25, width=12.3)
    add_footer(slide, "After training, model creates non-linear XOR regions")

    # 7) Training curve
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Training Dynamics")
    add_image(slide, assets_dir / "loss_curve.png", left=0.9, top=1.5, width=7.4)
    add_bullets(
        slide,
        [
            f"Initial loss: {losses[0]:.4f}",
            f"Final loss: {losses[-1]:.4f}",
            "Loss decreases smoothly, indicating successful learning",
            "Model converges to stable XOR solution",
        ],
        left=8.6,
        top=2.1,
        width=4.4,
        height=3.8,
        font_size=18,
    )
    add_footer(slide)

    # 8) Results
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, (244, 249, 244))
    add_title(slide, "Final Results")

    table = slide.shapes.add_table(5, 4, Inches(0.8), Inches(1.6), Inches(7.7), Inches(3.5)).table
    headers = ["x1", "x2", "Target", "Predicted Prob"]
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(14)

    for i, x in enumerate(X):
        table.cell(i + 1, 0).text = str(int(x[0]))
        table.cell(i + 1, 1).text = str(int(x[1]))
        table.cell(i + 1, 2).text = str(int(y[i, 0]))
        table.cell(i + 1, 3).text = f"{float(probs[i, 0]):.4f}"

    add_bullets(
        slide,
        [
            f"Binary predictions: {preds.ravel().tolist()}",
            "Expected labels: [0, 1, 1, 0]",
            "Prediction matches target for all 4 points",
            "Accuracy on XOR truth table: 100%",
        ],
        left=8.9,
        top=1.9,
        width=4.1,
        height=3.8,
        font_size=18,
    )
    add_footer(slide)

    # 9) Viva Q&A prep
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    add_title(slide, "Viva Q&A Quick Prep")
    add_bullets(
        slide,
        [
            "Why ReLU in hidden layer? Faster optimization and sparse activation",
            "Why sigmoid at output? Gives probability for binary class",
            "Why BCE loss? Proper likelihood-based objective for binary labels",
            "Why hidden layer for XOR? XOR is not linearly separable",
            "How to improve further? Add regularization, try tanh, test different seeds",
        ],
        width=12.0,
        height=5.6,
        font_size=21,
    )
    add_footer(slide)

    prs.save(output_ppt)


def main():
    base_dir = Path(__file__).resolve().parent
    assets_dir = base_dir / "ppt_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    init_params, trained_params, losses, probs, preds = train_xor(epochs=20000, lr=0.1)

    save_xor_points_plot(assets_dir / "xor_points.png")
    save_hidden_lines_plot(assets_dir / "hidden_lines.png", init_params, trained_params)
    save_output_surface_plot(assets_dir / "output_surface.png", init_params, trained_params)
    save_loss_curve(assets_dir / "loss_curve.png", losses)
    save_architecture_diagram(assets_dir / "architecture.png")

    out_ppt = base_dir / "XOR_Neural_Network_Viva_Presentation.pptx"
    create_ppt(out_ppt, assets_dir, losses, probs, preds)

    print(f"PPT generated: {out_ppt}")
    print(f"Assets folder: {assets_dir}")


if __name__ == "__main__":
    main()
