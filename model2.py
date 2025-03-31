from transformers import pipeline

# Load question-answering pipeline
qa_pipeline = pipeline("question-answering", model="deepset/minilm-uncased-squad2")

# Load context (PDF converted to text)
pdf_text = """
# References for Algorithms and Models

In the development of your Advanced Driver Assistance System (ADAS) using Python and GTA V as a simulator, several key algorithms and models play pivotal roles. Below is an in-depth overview of these components, including their mathematical foundations and practical applications.

## Hough Line Transform

### Overview

The **Hough Line Transform** is a fundamental technique in image processing used to detect straight lines within an image. By transforming points from the Cartesian coordinate system to a parameter space, it identifies lines even in noisy environments.

### Mathematical Foundation

In the Cartesian coordinate system, a straight line is represented as:

```
y = mx + c
```

Where:
- `m` is the slope,
- `c` is the y-intercept.

The Hough Transform, however, utilizes the polar coordinate representation:

```
ρ = x * cos(θ) + y * sin(θ)
```

Here:
- `ρ` is the perpendicular distance from the origin to the line,
- `θ` is the angle between the x-axis and the line connecting the origin to the closest point on the line.

By mapping all edge points `(x, y)` in the image to the `(ρ, θ)` space, the algorithm identifies the accumulation points corresponding to potential lines.

### Application

1. **Edge Detection**: Apply an edge detection algorithm (e.g., Canny Edge Detector) to identify potential edge points.
2. **Parameter Space Mapping**: For each edge point, compute possible `(ρ, θ)` values and map them in an accumulator matrix.
3. **Peak Detection**: Identify peaks in the accumulator matrix, which correspond to the most likely lines in the image.

*For a comprehensive guide on the Hough Transform, refer to this [resource](https://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm).*

## Canny Edge Detection

### Overview

Developed by **John F. Canny** in 1986, the Canny Edge Detection algorithm is renowned for its ability to detect a wide range of edges in images with optimal accuracy.

### Process

The algorithm involves several steps:

1. **Noise Reduction**: Apply a Gaussian filter to smooth the image and reduce noise.
2. **Gradient Calculation**: Compute the intensity gradients of the image.
3. **Non-Maximum Suppression**: Thin the edges by suppressing pixels that are not at the maximum gradient.
4. **Double Thresholding**: Classify edges as strong, weak, or non-relevant based on gradient magnitude thresholds.
5. **Edge Tracking by Hysteresis**: Finalize edge detection by connecting weak edges to strong edges if they are in close proximity.

### Mathematical Foundation

1. **Gaussian Smoothing**:

   ```
   H(i, j) = (1 / (2πσ²)) * exp(-((i - (k+1))² + (j - (k+1))²) / (2σ²))
   ```

   Where `σ` is the standard deviation of the Gaussian distribution, and `(2k+1) x (2k+1)` is the kernel size.

2. **Gradient Calculation**:

   ```
   G = √(Gx² + Gy²)
   θ = atan2(Gy, Gx)
   ```

   Where `Gx` and `Gy` are the gradients in the x and y directions, respectively, and `θ` is the gradient direction.

*For a detailed explanation of the Canny Edge Detector, visit this [page](https://en.wikipedia.org/wiki/Canny_edge_detector).*

## YOLOv8n for Object Detection

### Overview

**YOLO (You Only Look Once)** is a state-of-the-art, real-time object detection system. The **YOLOv8n** variant is a lightweight model designed for high-speed applications, making it suitable for real-time scenarios like driving simulations.

### Architecture

YOLOv8n employs a single neural network that divides the image into a grid and predicts bounding boxes and class probabilities for each grid cell simultaneously. This unified approach allows for rapid and accurate object detection.

### Mathematical Foundation

The model optimizes the following multi-part loss function:

```
Loss = λ_coord * Σ (x - x̂)² + (y - ŷ)² + λ_coord * Σ (√w - √ŵ)² + (√h - √ĥ)²
       + Σ (C - Ĉ)² + λ_noobj * Σ (C - Ĉ)² + Σ_i Σ_c (p_i(c) - p̂_i(c))²
```

Where:
- `(x, y, w, h)` represent the bounding box coordinates and dimensions,
- `C` is the confidence score,
- `p_i(c)` is the class probability,
- `λ_coord` and `λ_noobj` are hyperparameters to balance the loss components.

*For an in-depth guide on YOLO object detection, refer to this [article](https://encord.com/blog/yolo-object-detection-guide/).*

## Support Vector Machines (SVM)

### Overview

**Support Vector Machines** are supervised learning models used for classification and regression tasks. They are particularly effective in high-dimensional spaces and are widely applied in pattern recognition.

### Mathematical Foundation

Given a training dataset with labels, SVM aims to find the optimal hyperplane that separates the data points of different classes with the maximum margin. The decision function is defined as:

```
f(x) = sign(w · x + b)
```

Where:
- `w` is the weight vector,
- `x` is the input feature vector,
- `b` is the bias term.

The optimization problem involves minimizing `||w||²` subject to the constraint that all data points are correctly classified with a margin of at least 1.

### Kernel Trick

For non-linearly separable data, SVM employs kernel functions to map the input features into higher-dimensional spaces where a linear separator may exist. Common kernels include:
- **Linear Kernel**: `K(x, y) = x · y`
- **Polynomial Kernel**: `K(x, y) = (x · y + c)^d`
- **Radial Basis Function (RBF) Kernel**: `K(x, y) = exp(-γ ||x - y||²)`

*For a comprehensive understanding of SVM, consider exploring machine learning textbooks or academic papers dedicated to this topic.*

---

*Note: For enhanced comprehension, consider incorporating visual representations of these algorithms and models in your report.* 
"""

while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == "exit":
        break

    # Get the answer
    result = qa_pipeline(question=question, context=pdf_text)
    print(f"Answer: {result['answer']} (Confidence: {result['score']:.2f})")
