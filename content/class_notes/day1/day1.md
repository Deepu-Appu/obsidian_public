### ` How does the Machine Algorithm `

- If there is $x1$ and $x2$  are features in a problem statement we have to find and $y$ is the output we have to find.
- We find this through using Machine learning algorithm.
- The Machine learning gives equation using the, (equation to find the similar values or we can say prediction) this is the training phase.
- At last we predict output.
### ` Deep Learning `

- Deep learning comes into play when the data become more complicated, I mean when the data has more feature it would become hard for single machine learning model to predict the output.
- So, that's why we use neural network to calculate the output.
- Just like bagging , Where we use multiple model, in neural network we use multiple model, just like human neural network.
- This is the *Deep Learning*.
	![[Pasted image 20250825120258.png]]
- Taking an example of making chai, chai is the output and ingredient are the inputs.
	![[ChatGPT Image Aug 25, 2025, 12_07_08 PM.png]]
	
### ` Explanation (Teachers = Hidden Layers, Students = Inputs/Outputs) `

- Imagine **3 students** who each know some part of a subject (like math, science, and history). They provide information (inputs).
- This information goes to **3 teachers**. Each teacher processes the students' answers in a different way (like giving weight to each student's input).
- The teachers then discuss among themselves and combine their processed knowledge.
- Finally, the teachers give a final decision/answer (the network’s output).

👉 Here:
- **Students = Input neurons (x1, x2, x3)**
- **Teachers = Hidden neurons (h1, h2, h3)**
- **Final Result = Output neuron (y)**
	![[ChatGPT Image Aug 25, 2025, 12_23_03 PM.png]]
- We use `MSE`(Mean square error), so that it would be easy to calculate the loss use it find the best co-efficient or weight using the gradient descend.
### ` Concept `:

- **Loss Function (MSE – Mean Squared Error):**
    - Measures how far the predictions are from the actual outputs.
    - We want this error to be as **small as possible**.
    
- **Gradient Descent:**
    - Think of it like rolling a ball downhill on a curve (loss surface).
    - The goal is to reach the **lowest point** (best fit).
    
- **Global Minima:**
    - The **lowest point** on the curve → best possible fit, lowest error.
    
- **Local Minima:**
    - A small dip (valley), not the absolute lowest. The algorithm may get stuck here.
    
- **Global Maxima:**
    - The **highest point** on the curve (but in MSE we don’t want maxima, since error is high). 
    ![[Pasted image 20250825130745.png]]
> [!info] In the context of various computational methods, particularly in machine learning for tasks like calculating loss, absolute value (magnitude) serves a specific purpose: to nullify the effect of the sign of a prediction or error. This means that whether a prediction is too high (positive error) or too low (negative error), the system should only consider the size or magnitude of that error, not its direction. By taking the absolute value, you obtain non-negative values, ensuring that errors are treated equally regardless of their positive or negative direction.

However, despite its conceptual utility, the **absolute function is generally not used** in method like gradient descend due to a critical limitation: **It is not differentiable**. Specifically, the absolute value function is not differentiable at the zero value, and while it is continuous on its left and right halves, it exhibits a change in behavior when moving from negative to positive values.

For a loss function to be effectively utilized in gradient **gradient descent**, which relies on **differentiation** to find the minimum of the function, **it must be differentiable**. A non-differentiable function cannot be directly incorporated into a continuous equation that automates the learning process.

That's why we use **MSE** because it is differentiable.

1. ### ` The Purpose of Squaring the Error ` 
	- **Nullifying Signs**: One of the primary reasons for squaring the loss is to **cancel the effect of the sign** of a prediction error. Whether a prediction is too high or too low, the loss function should ideally only consider the **magnitude** or the absolute value of the error, not its direction. Squaring each error inherently **removes the sign** (as both positive and negative numbers becomes positive when squared), thus focusing purely on the magnitude
2. ### ` Why Squaring is Preferred over Absolute Value (Magnitude) `
	- **Differentiability**: The critical reason for choosing squaring over taking the absolute value is **differentiability**.
	- The **absolute function is not differentiable**, specifically **at the zero value**
	- Although it is continuous on its left and right halves, it exhibits a "change in behavior when it comes from negative values to positive values," making it non-differentiable at zero.
	- For a loss function to be effectively utilized in **gradient descent**, which relies on **differentiation to find the minimum** of the function, it **must be differentiable**. A reason for choosing squaring over taking the absolute value is **differentiability**.
	- **Enabling Gradient Descent**: Squaring the error result in a **differentiable function**, which is crucial for optimization methods like gradient descent. This differentiability allows the calculation of gradient (derivatives) necessary to update model parameters and minimize the loss function. The "squared sum of errors" is a common component of loss function considered by gradient descent, similar to its use in the R-squared formula.
3. ### ` Practical Consideration for Squared Loss `
	- **The Half Factor $(1/2)$** : Often, the actual loss function includes a factor of **one-half (1/2) multiplied by the squared error** . This is done for simple practical reason: to **simplify the derivatives**. The derivative of $x^2$ is $2x$, and multiplying by $1/2$ **effectively results** in x, streamlining the mathematical operations. while not strictly necessary, it makes calculations "more simple".
	- **Boarder Importance of Differentiability**: The need for differentiability extends beyond just loss function. For example, in logistic regression, a **sigmoid function** is chosen over a step function (which would eliminate the "narrow zone" of prediction) precisely because the sigmoid function is **continuous and differentiable**. This allows it to be used in models that require differentiation. Functions that "*look very simple on paper are not viable*" if they are not differentiable when creating computational models. 
### ` Derivative with respect to one variable `

**Derivative with respect to one variable** primarily within the context of functions that may or may not involve variable. This concept is fundamental to understanding how computational models, particularly neural networks, function and learn through differentiation.

1. ### ` Definition and purpose `
	- A derivative measures **how a functions is changing** with respect to a specific variable.
	- The notation $dy/dx$ typically indicates an **absolute derivative** when $y$ is composed of only one variable, $x$. In this simple case, "*this works*".
2. ### ` Handling Multiple Variables (Partial Derivatives) ` 
	- When a function $y$ is made up of multiple variable, such as $x1, x2, x3$ one **cannot simply** say $dy/dx$.
	- Instead, the concept shift to **partial derivatives**, denoted by `do` $(∂)$. For example, `do y do x1` $(∂y/∂x1)$ means that `x2` and `x3` are **not considered variables**; rather, they are **treated as constants**.
	- The core idea remains: "*there are only two things. One is variable, one is constant. Your function changes with the variable* " 
	- When calculating a partial derivatives (e.g., `do y do x1`), you consider all other variables (e.g.,`x2` and `x3` ) to be constant. This process is repeated for each variable if you need to find out how the function changes with respect to each of them individually (e.g., `do y do x2`, `do y do x3`, where the other two are held constant).
### ` Application in Linear Regression `

1. ### ` The Prediction Equation with Multiple Features `
	- In a linear regression model, a prediction (`y_prediction`) is formed by combining multiple features (e.g., `x1`, `x2`, `x3`) with their corresponding **weights** (e.g., `w1`,`w2`,`w3`).
	- The general form of such an equation is represented as a sum of products of weights and features. exact full equation `W1*x1 + W2*x2 + W3*x3`.
	- It's clarified that "*W1, W2, W3 and W4" are four variables in this equation*" , implying `w` could be an intercept term in addition to the feature weights.
2. ### ` Role in Model Learning and Optimization `
	- The **weights** (`w1`, `w2`, `w3`) within this prediction equation are the parameters that the model optimizes through **gradient descent** . Partial derivatives are calculated with respect to each of these weights to determine how they should be adjusted to minimize the loss. For instance, `w1_new` is calculated using `w1_current` and the partial derivative of the loss with respect to `w1`(e.g., `DL by DW1`)
3. ### ` The Weight Update Rule Explained `
	- `W1:W1_new = W1_old + LR * [(y_actual - y_pred) * X1]`
	- `W2:W2_new = W2_old + LR * [(y_actual - y_pred) * X2]`
	- Here, `w_new` refers to the updated weights, `w_old` is the current weight, `LR` (effectively the Learning Rate) is a factor that controls the step size of the update, and `DL/DW` represents the partial derivative of the loss function with respect to the specific weight.
4. ### ` Components of the Formula `
	- `w_old`: This is the current value of the weight (e.g., `w1`) before the update. The model starts with initial weights (often random) and iteratively adjusts them.
	- `LR (Learning Rate / EA)`: This parameter dictates how large a step the model takes in the direction indicated by the gradient. A suitable learning rate is crucial for efficient convergence (`convergence is the state reached by a model during training when its predictions stabilize and the underlying loss function stops decreasing significantly, indicating that further training would yield diminishing returns`). The source refer to it as `EA`.
	- `DL/DW` (Gradient of the Loss Function): This is the **partial derivative of the loss function** (L) **with respect to a specific weight** (W).
5. ### ` Underlying Mathematical Principles `
	- The inclusion of a `1/2` factor in the loss function (e.g., `1/2 * (y_actual - y_pred)^2`) is also mentioned to simplify the derivative, as it cancels out the `2` from the derivative of `x^2`
	- The understanding of basic differential concepts, including partial derivatives, is crucial for grasping how these weight updates work.
6. ### ` Context of Linear Regression `
	- The prediction for a linear model with multiple features is given as `y_prediction = W1.x1 + W2.X2 + W3.x3`. This formula `DL/DW1 = (y_actual - y_pred) * X1` is explicitly identified as the derivative of the loss function with respect to weight `W1` *for this type of linear equation only*.
	-  ◦ **(y_actual - y_pred)** **(Residual/Delta)**: This part of the formula is referred to as "the delta or residual". It represents the **error** in the model's prediction, which is the difference between the actual observed value (`y_actual`) and the value predicted by the model (`y_pred`)
### ` Chain Rule `

**Chain Rule Differentiation** as an indispensable technique for handling the intricate, layered structures of modern machine learning models, especially when moving beyond simpler linear relationships.
1. ### ` Simplifying Complex Calculations `
	- While it is theoretically possible to substitute intermediate functions to express the final variable directly in terms of the initial one, this approach "*will actually get complex if the intermediate equations are complex*". The Chain Rule offers a "*much simpler to solve*" and systematic way to handle these nested dependencies "*without breaking your head*" . It avoid the need to "*find out the final function every time*" on a "mathematical computational level also".
2. ### ` Application in Non-Linear Models `
	- The Chain Rule becomes crucial when machine learning models incorporate **non-linear activation functions**, such as the **sigmoid function**, especially in models like logistic regression or neural network.
	- When sigmoid function is introduced, the differential calculation becomes more complex, requiring "*the differential of it multiply with partial derivatives of the loss function with respect to w1*". This indicates a multi-layered application of differentiation that the Chain Rule facilitates.
### ` Foundation for Gradient Descent and Differentiability `

- Differentiation, including the Chain Rule, is fundamental to **gradient descent**, the optimization algorithm used to update mode parameters.
- The ability to apply the Chain Rule, like other differentiation methods, relies on the **differentiability of the loss function**. This is why squaring errors (e.g., in Mean Squared Error) is preferred over absolute values, as an "*Absolute function is not differentiable*" at zero and "*cannot be directly differentiable*" or "*put it in a continuous equation*" for automation.
1. ### ` Extension of Partial Derivatives `
	- The Chain Rule effectively extends the concepts of partial derivatives. While partial derivatives address function of multiple independent variables (e.g., `do y do x1` when `y` depends on `x1`,`x2`,`x3`), the Chain Rule then links theses partial derivatives across multiple layers of dependency in composite
2.## ` Computational Efficiency and Programmability `
	- The Chain Rule is highlighted as something that is  "*programmable also*". This underscores its practical importance in computational frameworks for machine learning, enabling efficient and automated gradient calculations, which  are central to algorithm like backpropagation in neural networks.

Notes, Audio, Video: <a href="https://mega.nz/folder/WBcnSYSK#5rC2hpmOcFOJlyjLHMwBsA" download>Download Markdown File</a>
