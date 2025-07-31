# Linear Algebra
## Notation
- `∀` : "for any"
- `∃` : "there exists"

---

## Vector Operations
### Basic Operations
- **Addition/Subtraction**:  
  $(u_1,u_2) + (v_1,v_2) = (u_1+v_1,u_2+v_2)$  
- **Scalar Multiplication**:  
  $k \cdot (v_1,v_2) = (k\cdot v_1, k\cdot v_2)$

### Dot Product (Inner Product)
$$\vec{u} \cdot \vec{v} = \sum_{i=1}^{n} u_i v_i$$
- **Geometric Interpretation**:  
  $\vec{u} \cdot \vec{v} = \|\vec{u}\|\|\vec{v}\| \cos{\theta}$  
  ($\theta$ = angle between vectors)
- **Orthogonality Condition**:  
  $\vec{u} \cdot \vec{v} = 0 \iff \vec{u} \perp \vec{v}$

### Cross Product (3D Only)
$$\vec{u} \times \vec{v} = (u_2v_3 - u_3v_2,\ u_3v_1 - u_1v_3,\ u_1v_2 - u_2v_1)$$
- **Properties**:
  - Result is perpendicular to input vectors (right-hand rule determines direction)
  - $\|\vec{u} \times \vec{v}\| = \|\vec{u}\|\|\vec{v}\| \sin{\theta}$ = area of parallelogram spanned by $\vec{u},\vec{v}$

### Norms
$$ \|\vec{x}\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p} \quad (p \geq 1) $$
- **Special Cases**:
  - Euclidean ($L^2$): $\|\vec{x}\|_2 = \sqrt{\sum x_i^2}$
  - Manhattan ($L^1$): $\|\vec{x}\|_1 = \sum |x_i|$
  - Infinity ($L^\infty$): $\|\vec{x}\|_\infty = \max |x_i|$
- **Convention**: $|\vec{v}|$ denotes Euclidean norm

### Linear Combinations & Span
- **Linear Combination**:  
  $\vec{w} = \sum_{i=1}^{n} c_i \vec{v}_i$  
  (Scalars $c_i$, vectors $\vec{v}_i$)
- **Span**: Set of all linear combinations of $\{\vec{v}_1,\dots,\vec{v}_n\}$  
  Example: $\text{span}\left(\begin{bmatrix}1\\0\\0\end{bmatrix},\begin{bmatrix}0\\1\\0\end{bmatrix}\right)$ = $xy$-plane in $\mathbb{R}^3$

### Orthonormal Vectors
Set $\{\vec{v}_1,\dots,\vec{v}_n\}$ satisfies:
1. **Orthogonal**: $\vec{v}_i \cdot \vec{v}_j = 0 \quad (i \neq j)$
2. **Unit Length**: $ \forall i \quad \|\vec{v}_i\| = 1$

---

## Matrix Operations

### Notations
- $A \in \mathbb{R}^{n\times m}$ Represents a matrix $A$ is a matrix of dimensions $n \times n$ whose entries are real numbers.
- $I$ is the identity matrix. for $I_3$, or the three dimension one, it would be $\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$.
- $A_{i,j}$ means the $i$ th row and $j$ th column of matrix $A$.

### Basic Operations
- **Addition and Subtraction**:
 Only matrices of the same dimensions can be added or subtracted. $(A+B)_{i,j} = A_{i,j} + B_{i,j}$. $(A-B)_{i,j} = A_{i,j} - B_{i,j}$.
- **Scalar Multiplication**:
Let $k$ be a scalar. $(kA)_{i,j} = k \times A_{i,j}$.

### Transpose
This is essentially flipping a matrix over its main diagonal(the upper-left to down-right diagonal). let $A = \begin{pmatrix} a & b \\ c& d \end{pmatrix}$ then its transpose $A^T = \begin{pmatrix} a & c \\ b& d \end{pmatrix}$

### Determinant
The determinant of a matrix, denoted as $\text{det}(A)$, provides important information about a matrix.
- A matrix $A$ is invertible if and only if $\text{det}(A) \neq 0$. We call a matrix without an inverse singular.
- The determinant can represent the scaling factor of the volume change when the matrix is applied as a linear transformation. In $\mathbb{R}^2$, the determinant of a $R^{2\times2}$ matrix represents the area scaling factor of the parallelogram formed by the column vectors of the matrix. 

**Properties**:
- Cramer's rule: The solution components of $A\vec{x}=\vec{b}$ are given by $x_i = \frac{\text{det}(A_i)}{\text{det}A}$, where $A_i$ is the matrix formed by replacing the $i$th column of $A$ with the vector $\vec{b}$
- Row operations: 
  - Swapping two rows changes the sign of the determinant
  - Multiplying a row by a scalar $k$ multiplies the determinant by $k$
  - Adding a multiple of one row to another rorw doesn't change the determinant
- $\text{det}(AB) = \text{det}(A)\cdot \text{det}(B)$
- $\text{det}(A^T)=\text{det}(A)$
- $\text{det}(0) = 0 \quad \text{det}(I) = 1$ (this means at any valid dimension where the identity matrix is defined, its determinant if 1)
- diagonal matrix: a matrix that only has entries on its main diagonal. for a diagonal matrix, its determinant is the product of all of its entries.

**Calculation**
- For a $1 \times 1$ matrix $A=\begin{pmatrix} a \end{pmatrix}$, $\text{det}(A) = a$.
- For a $2 \times 2$ matrix $A=\begin{pmatrix} a & b \\ c & d \end{pmatrix}$, $\text{det}(A) = ad - bc$.
- For $n \times n$ matrix, the determinant can be found by the property of diagonal matrices. This means we can use Gaussian elimination (covered below) to find a diagonal matrix and then find the determinant of that diagonal matrix. Remember to multiply the final determinant with the scale factors of each row due to the second property written in row operations. 

### Inverse
**only square matrices that are non-singular (non-zero determinant) have inverses.** The inverse of $A$, denoted by $A^{-1}$, is the matrix such that $A \times A^{-1} = A^{-1} \times A = I$. To find the inverse, you can use gaussian elimination to solve the equation where the augmented matrix would be: $A|I$. For a $2\times2$ matrix, $\begin{pmatrix} a & b \\ c & d \end{pmatrix}, its inverse is $\frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$.

### Matrix Multiplication
For $A \in \mathbb{R}^{m \times p}$, $B \in \mathbb{R}^{p \times n}$:  
$$C = AB \quad \text{where} \quad c_{ij} = \sum_{k=1}^{p} a_{ik}b_{kj}$$
- $c_{ij}$ = dot product of $i$-th row of $A$ and $j$-th column of $B$

![image.png](attachment:image.png)

---
## Linear Transformations and Eigen Theory
### Linear Maps
A linear map (linear transformation) is a function $T:V \to W$ between vector spaces $V$ and $W$ over field $\mathbb{F}$ satisfying $\forall \vec{u},\vec{v} \in V, k\in \mathbb{F}$:
1. $T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v})$
2. $T(k\vec{v}) = kT(\vec{v})$

### Matrix Representation
Matrix multiplication represents linear maps. For $A \in \mathbb{F}^{m \times n}$, define $T_A : \mathbb{F}^n \to \mathbb{F}^m$ by:
$$T_A(\vec{x}) = A\vec{x}$$
This satisfies linearity:
- $T_A(\vec{x} + \vec{y}) = A(\vec{x} + \vec{y}) = A\vec{x} + A\vec{y} = T_A(\vec{x}) + T_A(\vec{y})$
- $T_A(k\vec{x}) = A(k\vec{x}) = k(A\vec{x}) = kT_A(\vec{x})$

### Example: Rotation in $\mathbb{R}^2$
The rotation matrix for counter-clockwise rotation by $\theta$:
$$R_\theta = \begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}$$
defines linear map $T_\theta(\vec{x}) = R_\theta \vec{x}$.

For $\theta = 60^\circ = \frac{\pi}{3}$:
$$R_{60^\circ} = \begin{bmatrix}
\frac{1}{2} & -\frac{\sqrt{3}}{2} \\
\frac{\sqrt{3}}{2} & \frac{1}{2}
\end{bmatrix}$$

#### Transforming Standard Basis:
$$\mathbf{e}_1 = \begin{bmatrix}1\\0\end{bmatrix},\quad \mathbf{e}_2 = \begin{bmatrix}0\\1\end{bmatrix}$$

$$T_{60^\circ}(\mathbf{e}_1) = \begin{bmatrix}\frac{1}{2} \\ \frac{\sqrt{3}}{2}\end{bmatrix},\quad T_{60^\circ}(\mathbf{e}_2) = \begin{bmatrix}-\frac{\sqrt{3}}{2} \\ \frac{1}{2}\end{bmatrix}$$

Geometrically: All vectors rotated counter-clockwise by $60^\circ$ about origin, preserving lengths and angles.

### Eigenvectors and Eigenvalues
An eigenvector $\vec{v} \neq \vec{0}$ of linear transformation $T$ is a direction invariant under $T$, scaled by eigenvalue $\lambda$:
$$T(\vec{v}) = \lambda \vec{v}$$

- $\lambda > 0$: Direction preserved
- $\lambda < 0$: Direction reversed

#### Finding Eigenvalues/Eigenvectors
For matrix $A$ representing $T$:
1. Solve characteristic equation: $\det(A - \lambda I) = 0$
2. For each $\lambda_i$, solve $(A - \lambda_i I)\vec{v} = \vec{0}$

##### Example
For $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$:  
Characteristic equation:  
$$\det\begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix} = (3-\lambda)(2-\lambda) = 0$$  
Eigenvalues: $\lambda_1 = 3, \lambda_2 = 2$

- For $\lambda_1=3$:  
$$\begin{pmatrix}0 & 1 \\ 0 & -1\end{pmatrix}\begin{pmatrix}v_1\\v_2\end{pmatrix} = \begin{pmatrix}0\\0\end{pmatrix} \implies v_2=0$$  
Eigenvectors: $\begin{pmatrix}v_1\\0\end{pmatrix} = v_1\begin{pmatrix}1\\0\end{pmatrix} \ (v_1 \neq 0)$

- For $\lambda_2=2$:  
$$\begin{pmatrix}1 & 1 \\ 0 & 0\end{pmatrix}\begin{pmatrix}v_1\\v_2\end{pmatrix} = \begin{pmatrix}0\\0\end{pmatrix} \implies v_1 = -v_2$$  
Eigenvectors: $\begin{pmatrix}v_1\\-v_1\end{pmatrix} = v_1\begin{pmatrix}1\\-1\end{pmatrix} \ (v_1 \neq 0)$

*Note that the determinant of a matrix is equal to the product of its eigenvalues.

---

## Matrix Decomposition
### Gaussian Elimination

This is an algorithm for solving a linear equation system using linear algebra methods. The idea is to rewrite the matrix into an upper triangular matrix that represents the equation system. For example, for the linear equation:
$$\begin{cases} x + 2y - 4z = 5 \\ 2x + y - 6z = 8 \\ 4x - y - 12z = 13 \end{cases}$$
We have a coefficient matrix and an augmented matrix
$$ A_{\text{Coefficient}} = \begin{bmatrix} 1 & 2 & {-4} \\ 2 & 1 & {-6} \\ 4 & {-1} & {-12} \end{bmatrix}$$
$$
A_{\text{Augmented}} = \left[
\begin{array}{rrr|r}
1 & 2 & -4 & 5 \\
2 & 1 & -6 & 8 \\
4 & -1 & -12 & 13
\end{array}
\right]
$$

The corresponding upper triangular matrix would be
$$
A_{\text{Augmented}} = \left[
\begin{array}{rrr|r}
1 & 2 & -4 & 5 \\
0 & -3 & 2 & -2 \\
0 & 0 & -2 & -1
\end{array}
\right]
$$
The equivalent linear system is
$$\begin{cases} x + 2y - 4z = 5 \\ -3y + 2z = -2 \\ -2z = -1 \end{cases}$$

How is this done? This is an example to show the process. We name the $i$th row as $R_i$, the $i$th column as $C_i$, and $M_{i,j}$ the value in the $i$th row and the $j$th column. we start with $i=1$ and end at $i=n$, which is the number of rows the matrix has. Define constant $k\in \mathbb{R}$.
$$
\left[
\begin{array}{rrr|r}
2 & 2 & 6 & 4 \\
2 & 1 & 7 & 6 \\
-2 & -6 & -7 & -1
\end{array}
\right]
$$
We first scale $R_i$ such that $M_{i,i}=1$. 
$$
\left[
\begin{array}{rrr|r}
1 & 1 & 3 & 2 \\
2 & 1 & 7 & 6 \\
-2 & -6 & -7 & -1
\end{array}
\right]
$$
We add $k \cdot R_i$ to all $R_j$ with $j>i$ such that $M_{j,i} = 0$.
$$
\left[
\begin{array}{rrr|r}
1 & 1 & 3 & 2 \\
0 & -1 & 1 & 2 \\
0 & -4 & -1 & 3
\end{array}
\right]
$$
We repeat the process on the next $i$. Here $i=2$, so we scale the second row such that $M_{i,i}=1$.
$$
\left[
\begin{array}{rrr|r}
1 & 1 & 3 & 2 \\
0 & 1 & -1 & -2 \\
0 & -4 & -1 & 3
\end{array}
\right]
$$
We then add a multiple of $R_2$ to all rows under it such that the second element of each such row becomes $0$. Here, we add $4\cdot R_2$ to $R_3$.
$$
\left[
\begin{array}{rrr|r}
1 & 1 & 3 & 2 \\
0 & 1 & -1 & -2 \\
0 & 0 & -5 & -5
\end{array}
\right]
$$
We then repeat the process for the last row.
$$
\left[
\begin{array}{rrr|r}
1 & 1 & 3 & 2 \\
0 & 1 & -1 & -2 \\
0 & 0 & 1 & 1
\end{array}
\right]
$$
We now have a upper triangular matrix that can be easily used to find the solutions of the initial linear equation system. To solve the system, we essentially need to find $M_{ij}$ where $i=j$. Starting with $i=n$, we reverse the process. We take $k\cdot R_i$ and add it to all $R_j$ with $j<i$ such that $M_{j,i} = 0$.
$$
\left[
\begin{array}{rrr|r}
1 & 1 & 0 & -1 \\
0 & 1 & 0 & -1 \\
0 & 0 & 1 & 1
\end{array}
\right]
$$
We repeat for $i=n-1$.
$$
\left[
\begin{array}{rrr|r}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & -1 \\
0 & 0 & 1 & 1
\end{array}
\right]
$$

A generalization in natural language:

start with $i=1$

1. Start with $i=1$, Scale $R_i$ such that $M_{i,i}=1$.
2. Add $k \cdot R_i$ to all $R_j$ with $j>i$ such that $M_{j,i} = 0$.
3. Move to the next $i$ by adding $1$

After this has been done to all rows:
1. Add $k \cdot R_i$ to all $R_j$ with $j<i$ such that $M_{j,i}=0$
2. move to the next $i$ by subtracting $1$

A psuedo code:
$$
\begin{aligned}
&\textbf{Pseudocode for Gaussian Elimination:} \\
&\textbf{Input:} \text{Matrix } M \text{ of size } n \times n \\
&\textbf{Output:} \text{Row-echelon form of } M \\
& \\
&\text{1. Forward Elimination:} \\
&\text{for } i = 1 \text{ to } n-1 \text{ do} \hspace{1em} \\
&\hspace{2em} \text{(a) Scale } R_i \text{ such that } M_{i,i} = 1: \\
&\hspace{3em} \text{if } M_{i,i} \neq 0 \text{ then} \\
&\hspace{4em} \text{Scale factor } s = \frac{1}{M_{i,i}} \\
&\hspace{4em} R_i = s \cdot R_i \\
&\hspace{3em} \text{end if} \\
&\hspace{2em} \text{(b) Add } k \cdot R_i \text{ to all } R_j \text{ with } j > i \text{ such that } M_{j,i} = 0: \\
&\hspace{3em} \text{for } j = i+1 \text{ to } n \text{ do} \hspace{1em} \\
&\hspace{4em} \text{if } M_{j,i} \neq 0 \text{ then} \\
&\hspace{5em} k = -M_{j,i} \\
&\hspace{5em} R_j = R_j + k \cdot R_i \\
&\hspace{4em} \text{end if} \\
&\hspace{3em} \text{end for} \\
&\text{end for} \\
& \\
&\text{2. Backward Elimination:} \\
&\text{for } i = n-1 \text{ down to } 1 \text{ do} \hspace{1em} \\
&\hspace{2em} \text{(a) Add } k \cdot R_i \text{ to all } R_j \text{ with } j < i \text{ such that } M_{j,i} = 0: \\
&\hspace{3em} \text{for } j = i-1 \text{ down to } 1 \text{ do} \hspace{1em} \\
&\hspace{4em} \text{if } M_{j,i} \neq 0 \text{ then} \\
&\hspace{5em} k = -M_{j,i} \\
&\hspace{5em} R_j = R_j + k \cdot R_i \\
&\hspace{4em} \text{end if} \\
&\hspace{3em} \text{end for} \\
&\text{end for} \\
& \\
&\textbf{End}
\end{aligned}
$$

---

## Algebraic Structures
### Fields ($\mathbb{F}$)
Set of scalars with two operations (`+`, `·`) satisfying:
1. **Closure** under addition and multiplication
2. **Commutativity**: $a+b=b+a$, $a·b=b·a$
3. **Associativity**: $(a+b)+c=a+(b+c)$, $(a·b)·c=a·(b·c)$
4. **Identities**:  
   - Additive: $a+0=a$  
   - Multiplicative: $a·1=a$ $(1 \neq 0)$
5. **Inverses**:  
   - Additive: $a + (-a) = 0$  
   - Multiplicative: $a·a^{-1}=1$ $(a \neq 0)$
6. **Distributivity**: $a·(b+c) = a·b + a·c$

*Example*: $\mathbb{R}$ (real numbers)

### Vector Spaces ($V$ over $\mathbb{F}$)
Set of vectors with two operations (`+`, scalar multiplication) satisfying:
1. **Closure** under vector addition and scalar multiplication
2. **Commutativity/Associativity** of addition
3. **Zero vector**: $\vec{v} + \vec{0} = \vec{v}$
4. **Additive inverse**: $\vec{v} + (-\vec{v}) = \vec{0}$
5. **Distributivity**:  
   - $k(\vec{u} + \vec{v}) = k\vec{u} + k\vec{v}$  
   - $(α+β)\vec{v} = α\vec{v} + β\vec{v}$
6. **Associativity of scalars**: $α(β\vec{v}) = (αβ)\vec{v}$
7. **Multiplicative identity**: $1·\vec{v} = \vec{v}$

*Examples*: $\mathbb{R}^n$, polynomials of degree $\leq 4$, $2\times 3$ matrices

### Subspaces ($W \subseteq V$)
Subset satisfying:
1. $k \in \mathbb{F}, \vec{w} \in W \implies k\vec{w} \in W$
2. $\vec{u},\vec{v} \in W \implies \vec{u} + \vec{v} \in W$

*Equivalent to*: $W$ is itself a vector space  
*Example*: Plane $x+y+z=0$ in $\mathbb{R}^3$

### Basis
Set $B = \{\vec{b}_1,\dots,\vec{b}_n\}$ that is:
1. **Linearly Independent**:  
   $\sum c_i\vec{b}_i = \vec{0} \implies c_i = 0 \ \forall i$
2. **Spanning**: $\forall \vec{v} \in V,\ \exists c_i : \vec{v} = \sum c_i\vec{b}_i$
- **Dimension** ($\dim V$): Number of basis vectors ($n$)
- *Example*: Standard basis $\{(1,0,0),(0,1,0),(0,0,1)\}$ for $\mathbb{R}^3$