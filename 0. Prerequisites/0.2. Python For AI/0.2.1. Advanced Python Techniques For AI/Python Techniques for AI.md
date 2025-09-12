<div align="center">

# All About **Linear & Logistic Regression**
###### Will Chen | SHSID Data Science Group

<div align="left">

### Why advanced Python techniques for AI? 

- In machine learning and AI, Python language features are often overlooked. It's very easy for someone to know all about things like NumPy and PyTorch but miss some of the essential Python syntax that makes the subsequent parts much easier. 
- In this lesson, we'll go over several of the most important Python techniques that you will definitely be using in the future, whether if it's training a model on your own, or playing with a model locally. 
- The goal of this lesson is to get you started on some Python syntax for AI. It's fine if you don't totally understand a feature down to its C basis, that's not what we're teaching anyway. We want to familiarize you with ways that you can use to make your workflow more efficient. 

### 1. Metaclasses

#### Concept: 
 Metaclasses allow you to control the creation of classes. ML's application to this include: 
    *   **Automatic Model Registration:** Registering models into a central registry simply by defining them.
    *   **Enforcing APIs:** Ensuring all models implement specific methods (e.g., `fit`, `predict`).
    *   **Injecting Common Configuration:** Automatically adding common attributes or methods to all models.

#### Example: 

```python
class ModelRegistry(type):
    _models = {}

    def __new__(mcs, name, bases, attrs):
        # Create the class normally
        new_class = super().__new__(mcs, name, bases, attrs)

        # Register the class if it's not the base Model class itself
        if name != 'BaseModel':
            key = attrs.get('model_name', name.lower())
            ModelRegistry._models[key] = new_class
            print(f"Registered model: {key}")

        return new_class

    @classmethod
    def get_model(mcs, name):
        return mcs._models.get(name)

class BaseModel(metaclass=ModelRegistry):
    def __init__(self, config=None):
        self.config = config or {}

    def train(self, data):
        raise NotImplementedError("Subclasses must implement 'train'")

    def predict(self, input_data):
        raise NotImplementedError("Subclasses must implement 'predict'")

    @classmethod
    def from_config(cls, config):
        return cls(config)

class MyLogisticRegression(BaseModel):
    model_name = "logistic_regression"

    def train(self, data):
        print(f"Training Logistic Regression with config: {self.config}")
        # ... actual training logic ...

    def predict(self, input_data):
        print(f"Predicting with Logistic Regression for input: {input_data}")
        return [0.5] * len(input_data)

class MyDecisionTree(BaseModel):
    model_name = "decision_tree"

    def train(self, data):
        print(f"Training Decision Tree with config: {self.config}")
        # ... actual training logic ...

    def predict(self, input_data):
        print(f"Predicting with Decision Tree for input: {input_data}")
        return [1] * len(input_data)

# Usage
log_reg_model_cls = ModelRegistry.get_model("logistic_regression")
if log_reg_model_cls:
    log_reg_instance = log_reg_model_cls.from_config({"learning_rate": 0.01})
    log_reg_instance.train(["data1", "data2"])
    log_reg_instance.predict(["input1"])

# You can even iterate registered models
print("\nAll registered models:", list(ModelRegistry._models.keys()))
```

### 2. Descriptors

#### Concept: 
Descriptors allow you to customize how attribute access (get, set, delete) works for an object. They are powerful for:
    *   **Automatic Feature Transformation:** Applying a transformation (e.g., scaling, one-hot encoding) when a feature is set.
    *   **Data Validation:** Ensuring features meet certain criteria (e.g., non-negative, within a range) upon assignment.
    *   **Memoization/Caching of Derived Features:** Computing an expensive derived feature only once.

#### Example: 

```python
class FeatureValidator:
    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self.public_name} must be a number.")
        if value < 0:
            raise ValueError(f"{self.public_name} cannot be negative.")
        if self.public_name == 'age' and not (1 <= value <= 120):
            raise ValueError("Age must be between 1 and 120.")
        setattr(obj, self.private_name, value)

class ScaledFeature:
    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise ValueError(f"{self.public_name} must be a number to be scaled.")
        setattr(obj, self.private_name, value * self.scale_factor) # Apply scaling

class PatientRecord:
    age = FeatureValidator()
    bmi = FeatureValidator()
    scaled_glucose = ScaledFeature(scale_factor=0.1) # Example: raw glucose 100 becomes 10.0

    def __init__(self, age, bmi, glucose):
        self.age = age
        self.bmi = bmi
        self.scaled_glucose = glucose # This will be scaled by the descriptor

# Usage
p1 = PatientRecord(age=30, bmi=25.5, glucose=120)
print(f"Patient 1: Age={p1.age}, BMI={p1.bmi}, Scaled Glucose={p1.scaled_glucose}")

try:
    p2 = PatientRecord(age=-5, bmi=20, glucose=80) # This will raise an error
except ValueError as e:
    print(f"Error creating patient 2: {e}")

try:
    p3 = PatientRecord(age=45, bmi="twenty", glucose=90) # This will raise an error
except ValueError as e:
    print(f"Error creating patient 3: {e}")

p4 = PatientRecord(age=60, bmi=30.1, glucose=150)
print(f"Patient 4: Age={p4.age}, BMI={p4.bmi}, Scaled Glucose={p4.scaled_glucose}")
```

### 3. Abstract base classes (ABCs) 

#### Concept: 
 The `abc` module allows you to define abstract classes and methods, enforcing that subclasses provide specific implementations. Crucial for:
    *   **Standardizing Model Interfaces:** Ensuring all models (e.g., a `Classifier` or `Regressor`) adhere to a common `fit`, `predict` API.
    *   **Building Extensible Frameworks:** Guiding developers on how to extend your AI/ML framework.
    *   **Type Hinting with Confidence:** Providing clearer contracts for type checkers.

#### Example: 

```python
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    """Abstract base class for all estimators."""

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, X, y):
        """Fit the estimator to the training data."""
        self._is_fitted = True

    @abstractmethod
    def predict(self, X):
        """Make predictions on new data."""
        if not self._is_fitted:
            raise RuntimeError("Estimator not fitted. Call 'fit' first.")
        pass # Subclasses will implement this

    def get_params(self):
        """Return parameters of the estimator."""
        return self.__dict__

    def set_params(self, **params):
        """Set parameters of the estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Estimator has no parameter '{key}'")
        return self

# This class will raise an error if not all abstract methods are implemented
# class MyIncompleteModel(BaseEstimator):
#     pass
# TypeError: Can't instantiate abstract class MyIncompleteModel with abstract methods fit, predict

class MyLinearRegressor(BaseEstimator):
    def __init__(self, learning_rate=0.01, max_iter=100):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None # Placeholder for learned weights

    def fit(self, X, y):
        print(f"Fitting Linear Regressor with LR={self.learning_rate}, Iter={self.max_iter}")
        # In a real scenario, X and y would be processed here
        # For demonstration, just set a dummy weight
        self.weights = [0.1, 0.2] * (len(X[0]) // 2) if X and X[0] else []
        super().fit(X, y) # Call base class fit to set _is_fitted

    def predict(self, X):
        super().predict(X) # Check if fitted
        print(f"Predicting with Linear Regressor for {len(X)} samples using weights {self.weights}")
        # Dummy prediction
        return [sum(x_val * w_val for x_val, w_val in zip(x, self.weights))
                if self.weights else 0 for x in X]


# Usage
model = MyLinearRegressor(learning_rate=0.05)
print(f"Initial params: {model.get_params()}")

model.set_params(max_iter=200)
print(f"Updated params: {model.get_params()}")

# Example data
X_train = [[1, 2], [3, 4], [5, 6]]
y_train = [3, 7, 11]
X_test = [[7, 8], [9, 10]]

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")

# Try to predict before fitting
unfitted_model = MyLinearRegressor()
try:
    unfitted_model.predict(X_test)
except RuntimeError as e:
    print(f"Error: {e}")
```

### 4. Generators and Iterators

#### Concept: 

Generators (functions with `yield`) and custom iterators provide a memory-efficient way to process large datasets without loading everything into memory at once. Critical for:
    *   **Large Datasets:** Training models on data that doesn't fit in RAM.
    *   **On-the-Fly Data Augmentation:** Generating augmented samples during training without pre-storing them.
    *   **Streaming Data:** Processing data from real-time feeds.

#### Example: 

```python
import time
import random

def data_loader_generator(file_path, batch_size=32):
    """
    Simulates loading and processing data in batches from a file.
    Yields batches of (features, labels).
    """
    print(f"Starting data loading from {file_path}...")
    with open(file_path, 'r') as f:
        batch = []
        for line_num, line in enumerate(f):
            # Simulate parsing a line into features and a label
            # In a real scenario, this would involve more complex parsing
            features = [float(x) for x in line.strip().split(',')[0:-1]]
            label = int(line.strip().split(',')[-1])

            batch.append((features, label))

            if len(batch) == batch_size:
                print(f"Yielding batch of {len(batch)} samples (up to line {line_num+1})")
                yield batch
                batch = []
        # Yield any remaining samples in the last batch
        if batch:
            print(f"Yielding final batch of {len(batch)} samples")
            yield batch
    print(f"Finished data loading from {file_path}.")

# Create a dummy data file
with open("large_data.csv", "w") as f:
    for i in range(1, 105): # 104 samples
        features = [random.random() * 10 for _ in range(5)]
        label = random.randint(0, 1)
        f.write(','.join(map(str, features + [label])) + '\n')

# Simulate training loop
print("--- Training Loop Start ---")
data_gen = data_loader_generator("large_data.csv", batch_size=20)
epoch_num = 1
for batch_idx, batch_data in enumerate(data_gen):
    # Simulate processing a batch
    print(f"Epoch {epoch_num}, Batch {batch_idx+1}: Processing {len(batch_data)} samples.")
    # Here you would typically feed batch_data into your model's training step
    # For demonstration, we just sleep
    time.sleep(0.1)

print("--- Training Loop End ---")


# Another example: On-the-fly data augmentation
def augment_image(image_data):
    # Simulate an image augmentation (e.g., rotation, flip)
    # In reality, image_data would be a NumPy array or similar
    augmented = image_data + "_augmented"
    print(f"  Augmenting: {image_data} -> {augmented}")
    return augmented

def image_augmenter(image_paths):
    for path in image_paths:
        # Simulate loading an image
        print(f"Loading image from {path}...")
        original_image = f"image_content_from_{path}"
        yield augment_image(original_image) # Yield augmented image

print("\n--- Image Augmentation ---")
image_files = ["img_001.jpg", "img_002.jpg", "img_003.jpg"]
for augmented_img in image_augmenter(image_files):
    print(f"Received augmented image: {augmented_img}")
    # Here, you'd feed the augmented_img to your model
```

### 5. Context Managers

#### Concept: 

Context managers (`with` statement) ensure resources are properly acquired and released, even if errors occur. Vital for:
    *   **Managing Training Sessions:** Ensuring models are saved, logs are flushed, or temporary files are cleaned up after a training run.
    *   **GPU Memory Allocation:** (While actual allocation is library-dependent, a context manager could wrap the library calls for clean release).
    *   **Managing Model Checkpoints:** Loading and saving models, ensuring file handles are closed.

#### Example: 

```python
import os
import tempfile
import json

class TrainingSession:
    def __init__(self, model_name, log_dir="logs", checkpoint_dir="checkpoints"):
        self.model_name = model_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.temp_log_file = None
        self.training_start_time = None
        print(f"Preparing training session for {model_name}...")

    def __enter__(self):
        # Create directories if they don't exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Open a log file (or create a temporary one)
        self.temp_log_file = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                         prefix=f"{self.model_name}_session_",
                                                         suffix=".log", dir=self.log_dir)
        self.training_start_time = time.time()
        self.temp_log_file.write(f"[{time.ctime()}] Training session started for {self.model_name}\n")
        print(f"  Log file created: {self.temp_log_file.name}")
        return self # Return self to be assigned to 'session' in 'with' statement

    def log(self, message):
        if self.temp_log_file:
            self.temp_log_file.write(f"[{time.ctime()}] {message}\n")
            self.temp_log_file.flush() # Ensure message is written immediately
            print(f"  [LOG] {message}")

    def save_checkpoint(self, model_state, step):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_step_{step}.ckpt")
        with open(checkpoint_path, 'w') as f:
            json.dump(model_state, f, indent=4)
        self.log(f"Model checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.log(f"Training session exited with an error: {exc_type.__name__}: {exc_val}")
            print(f"  --- ERROR during training: {exc_val} ---")
        else:
            self.log("Training session completed successfully.")

        self.log(f"Training duration: {time.time() - self.training_start_time:.2f} seconds.")
        if self.temp_log_file:
            self.temp_log_file.close()
            print(f"  Log file closed: {self.temp_log_file.name}")
        print(f"Finished training session for {self.model_name}.")

# Usage
print("--- Scenario 1: Successful Training ---")
with TrainingSession("MyCNNModel") as session:
    session.log("Initializing model parameters...")
    # Simulate training steps
    for i in range(3):
        session.log(f"Training step {i+1} of 10...")
        time.sleep(0.5)
        if (i+1) % 1 == 0:
            # Simulate saving a checkpoint
            session.save_checkpoint({"weights": [random.random() for _ in range(5)], "step": i+1}, i+1)
    session.log("Training finished.")

print("\n--- Scenario 2: Training with an Error ---")
with TrainingSession("MyRNNModel") as session:
    session.log("Initializing model parameters...")
    try:
        for i in range(2):
            session.log(f"Training step {i+1}...")
            time.sleep(0.5)
            if i == 1:
                raise ValueError("Simulating an error during training!")
    except ValueError as e:
        # The __exit__ will catch this exception and log it
        pass

# Clean up created directories (optional)
import shutil
if os.path.exists("logs"):
    shutil.rmtree("logs")
if os.path.exists("checkpoints"):
    shutil.rmtree("checkpoints")
```
### 6. Decorators

#### Concept: 

Decorators allow you to wrap functions or methods, adding functionality before or after their execution without modifying their core logic. Excellent for:
    *   **Logging:** Automatically logging function calls, arguments, and return values.
    *   **Caching/Memoization:** Storing results of expensive computations. (`functools.lru_cache` is a built-in decorator for this!)
    *   **Performance Profiling:** Measuring execution time of functions.
    *   **Pre/Post-processing:** Adding common data pre- or post-processing steps.
    *   **Retry Logic:** Automatically retrying flaky operations (e.g., API calls for data).

#### Example: 

```python
import time
from functools import wraps, lru_cache

# 1. Logging Decorator
def log_method_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__ if args else "Function"
        method_name = func.__name__
        print(f"[{time.ctime()}] {class_name}.{method_name} called with args: {args[1:]}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"[{time.ctime()}] {class_name}.{method_name} finished. Result: {result}")
        return result
    return wrapper

# 2. Performance Profiling Decorator
def profile_method(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__ if args else "Function"
        method_name = func.__name__
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"[{time.ctime()}] {class_name}.{method_name} executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

class ModelTrainer:
    def __init__(self, name="DefaultModel"):
        self.name = name
        self.history = []

    @log_method_call
    @profile_method
    def train_epoch(self, data_batch, epoch_num):
        print(f"  Training {self.name} for epoch {epoch_num} on batch of {len(data_batch)} samples.")
        # Simulate some heavy computation
        time.sleep(random.uniform(0.1, 0.5))
        loss = random.random() * 0.1 + (1 / (epoch_num + 1)) # Simulate decreasing loss
        accuracy = 0.8 + random.random() * 0.1
        self.history.append({'epoch': epoch_num, 'loss': loss, 'accuracy': accuracy})
        return {'loss': loss, 'accuracy': accuracy}

    @log_method_call
    @lru_cache(maxsize=128) # Built-in caching decorator
    def predict_features(self, features_tuple): # Must take hashable args for lru_cache
        print(f"  Predicting for features: {features_tuple} (potentially expensive operation)")
        time.sleep(0.3) # Simulate prediction time
        return sum(features_tuple) / len(features_tuple) # Dummy prediction

# Usage
trainer = ModelTrainer("AwesomeNet")
for epoch in range(1, 4):
    dummy_data = [f"sample_{i}" for i in range(100)]
    metrics = trainer.train_epoch(dummy_data, epoch)
    print(f"  Epoch {epoch} Results: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}")

print("\n--- Cached Prediction ---")
features1 = (1.0, 2.0, 3.0)
features2 = (4.0, 5.0, 6.0)

print(f"Prediction for {features1}: {trainer.predict_features(features1)}")
print(f"Prediction for {features2}: {trainer.predict_features(features2)}")
print(f"Prediction for {features1}: {trainer.predict_features(features1)}") # This will be cached!
print(f"Prediction for {features2}: {trainer.predict_features(features2)}") # This will be cached!

features3 = (7.0, 8.0, 9.0)
print(f"Prediction for {features3}: {trainer.predict_features(features3)}") # New computation
```

### 7. Type Hinting and Reflection (`mypy`)

#### Concept: 

Type hints (`list[int]`, `dict[str, Any]`, `Callable`, `Union`, `Optional`) allow you to declare the expected types of variables, function arguments, and return values.
    *   **Improved Readability and Maintainability:** Makes code easier to understand and refactor.
    *   **Early Bug Detection:** Static analysis tools like `mypy` can catch type-related errors *before* runtime.
    *   **Better IDE Support:** Enhanced auto-completion and error checking in editors.
    *   **Refinement of ML Model Interfaces:** Clearly define what kind of data your `fit` and `predict` methods expect.

#### Example: 

```python
from typing import List, Dict, Any, Union, Optional, Tuple, Callable

# Type aliases for clarity in ML contexts
Features = List[float]
Labels = List[int]
Prediction = Union[int, float] # Can be a single value
Metrics = Dict[str, float]

class Dataset:
    def __init__(self, data: List[Tuple[Features, int]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def get_sample(self, index: int) -> Tuple[Features, int]:
        return self.data[index]

class MLModel:
    def __init__(self, name: str, hyperparameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self._is_fitted: bool = False

    def fit(self, X: List[Features], y: Labels) -> None:
        """
        Trains the model on the provided features and labels.
        """
        if not X or not y:
            raise ValueError("X and y cannot be empty.")
        if len(X) != len(y):
            raise ValueError("Number of features and labels must match.")

        print(f"Fitting {self.name} model with {len(X)} samples.")
        # Simulate training logic
        time.sleep(0.2)
        self._is_fitted = True

    def predict(self, X_test: List[Features]) -> List[Prediction]:
        """
        Makes predictions for the given test features.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        if not X_test:
            return []

        print(f"Predicting with {self.name} for {len(X_test)} samples.")
        # Simulate prediction logic
        return [random.uniform(0, 1) for _ in X_test] # Example: regression output

    def evaluate(self, y_true: Labels, y_pred: List[Prediction]) -> Metrics:
        """
        Calculates evaluation metrics.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("True and predicted labels must have same length.")
        # Simulate metric calculation (e.g., accuracy, MSE)
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if int(p > 0.5) == t) / len(y_true) if y_true else 0
        return {"accuracy": accuracy, "f1_score": 0.0} # Dummy F1

# Usage with type hints
model = MLModel(name="DecisionTree", hyperparameters={"max_depth": 5})

train_features: List[Features] = [[1.0, 2.1], [3.5, 4.0], [5.2, 6.7]]
train_labels: Labels = [0, 1, 0]

model.fit(train_features, train_labels)

test_features: List[Features] = [[7.0, 8.0], [9.1, 10.2]]
predictions: List[Prediction] = model.predict(test_features)

true_test_labels: Labels = [1, 1]
metrics: Metrics = model.evaluate(true_test_labels, predictions)

print(f"Model Name: {model.name}")
print(f"Predictions: {predictions}")
print(f"Evaluation Metrics: {metrics}")

# --- Example of what mypy would catch ---
# model.fit("bad_data", [1, 2]) # Mypy: Argument "X" to "fit" of "MLModel" has incompatible type "str"; expected "List[List[float]]"
# model.fit(train_features, [1.0, 2.0]) # Mypy: Argument "y" to "fit" of "MLModel" has incompatible type "List[float]"; expected "List[int]"
```

### 8. `collections` Module

#### Concept: 

Python's `collections` module provides highly optimized, specialized container datatypes. Useful for:
    *   **`defaultdict`:** Grouping data (e.g., samples by class, features by type) without explicit checks.
    *   **`Counter`:** Frequency counting (e.g., label distribution, word counts in NLP).
    *   **`deque`:** Efficiently adding/removing from both ends (e.g., maintaining a fixed-size buffer of recent predictions or gradients).
    *   **`namedtuple` / `typing.NamedTuple`:** Creating lightweight, immutable data structures for cleaner record-keeping (e.g., experiment results, dataset entries).

#### Example: 

```python
from collections import defaultdict, Counter, deque, namedtuple
import random

# 1. defaultdict for grouping data
print("--- defaultdict for Grouping ---")
data_points = [
    {'class': 'A', 'value': 10},
    {'class': 'B', 'value': 20},
    {'class': 'A', 'value': 15},
    {'class': 'C', 'value': 25},
    {'class': 'B', 'value': 30},
]

grouped_by_class = defaultdict(list)
for dp in data_points:
    grouped_by_class[dp['class']].append(dp['value'])

print("Grouped by class:", dict(grouped_by_class))

# 2. Counter for frequency analysis
print("\n--- Counter for Frequencies ---")
labels = [0, 1, 0, 0, 1, 2, 1, 0, 2, 1, 0, 1]
label_counts = Counter(labels)
print("Label counts:", label_counts)
print("Most common labels:", label_counts.most_common(2))

# 3. deque for rolling averages/buffers
print("\n--- deque for Rolling Average ---")
# Simulating a stream of metric values
metric_stream = [random.uniform(0.5, 1.5) for _ in range(20)]
window_size = 5
recent_metrics = deque(maxlen=window_size)

for i, metric in enumerate(metric_stream):
    recent_metrics.append(metric)
    if len(recent_metrics) == window_size:
        avg = sum(recent_metrics) / window_size
        print(f"Step {i+1}: Current metric={metric:.2f}, Rolling average ({window_size} steps)={avg:.2f}")
    else:
        print(f"Step {i+1}: Current metric={metric:.2f}, Buffer not full yet.")

# 4. namedtuple for structured experiment results
print("\n--- namedtuple for Structured Results ---")
ExperimentResult = namedtuple("ExperimentResult", ["model_name", "dataset", "accuracy", "loss", "timestamp"])

results = []
results.append(ExperimentResult("CNN_v1", "ImageNet", 0.92, 0.08, time.time()))
results.append(ExperimentResult("RNN_v2", "TextCorpus", 0.88, 0.15, time.time()))

for res in results:
    print(f"Model: {res.model_name}, Dataset: {res.dataset}, Accuracy: {res.accuracy:.2f}, Loss: {res.loss:.2f}")
    # Access by attribute name, not magic index
    # print(res[0]) # Also works, but less readable
```

### 9. `@property` 

#### Concept: 

The `@property` decorator allows you to define methods that can be accessed like attributes, providing a clean way to add logic (validation, computation, caching) when getting or setting an attribute.
    *   **Data Validation:** Ensure internal state is consistent upon modification.
    *   **Derived Attributes:** Compute attributes on-the-fly or lazily, rather than storing them directly.
    *   **Encapsulation:** Hide internal implementation details.

#### Example: 

```python
class FeatureSet:
    def __init__(self, raw_data: List[float], normalize: bool = True):
        self._raw_data = raw_data
        self._normalized_data = None
        self.normalize_on_access = normalize

    @property
    def raw_data(self) -> List[float]:
        return self._raw_data

    @raw_data.setter
    def raw_data(self, value: List[float]):
        if not all(isinstance(x, (int, float)) for x in value):
            raise ValueError("All elements in raw_data must be numeric.")
        if any(x < 0 for x in value):
            print("Warning: Raw data contains negative values.")
        self._raw_data = value
        self._normalized_data = None # Invalidate cached normalized data

    @property
    def normalized_data(self) -> List[float]:
        """
        Returns normalized data (min-max scaling). Computes only once if not explicitly invalidated.
        """
        if self._normalized_data is None or not self.normalize_on_access:
            print("  (Re)calculating normalized data...")
            if not self._raw_data:
                self._normalized_data = []
            else:
                min_val = min(self._raw_data)
                max_val = max(self._raw_data)
                if max_val == min_val:
                    self._normalized_data = [0.0] * len(self._raw_data)
                else:
                    self._normalized_data = [(x - min_val) / (max_val - min_val) for x in self._raw_data]
        return self._normalized_data

    @property
    def mean_feature_value(self) -> float:
        """
        Calculates the mean of the raw features on demand.
        """
        if not self._raw_data:
            return 0.0
        return sum(self._raw_data) / len(self._raw_data)

# Usage
fs = FeatureSet([10, 20, 30, 40, 50])
print(f"Initial raw data: {fs.raw_data}")
print(f"Normalized data (first access): {fs.normalized_data}")
print(f"Normalized data (second access - cached): {fs.normalized_data}") # No re-calculation
print(f"Mean feature value: {fs.mean_feature_value:.2f}")

# Modify raw data, which invalidates normalized data
fs.raw_data = [5, 15, 25, 35, 45]
print(f"\nNew raw data: {fs.raw_data}")
print(f"Normalized data (after change): {fs.normalized_data}") # Re-calculated
print(f"Mean feature value: {fs.mean_feature_value:.2f}")

try:
    fs.raw_data = [1, 2, "not a number"]
except ValueError as e:
    print(f"\nError setting raw data: {e}")

# Example with negative numbers
fs_neg = FeatureSet([-10, 0, 10, 20])
print(f"\nRaw data with negatives: {fs_neg.raw_data}")
print(f"Normalized data for negatives: {fs_neg.normalized_data}")
```

### 10. `inspect` Module

#### Concept:

The `inspect` module provides functions to examine live objects, including modules, classes, methods, and functions. Useful for:
    *   **Dynamic Model Loading:** Discovering and loading models from a directory based on their class structure.
    *   **Hyperparameter Tuning:** Inspecting a function's signature to automatically generate hyperparameter grids (e.g., identifying default values or required arguments).
    *   **Plugin Architectures:** Allowing users to register custom components (e.g., custom loss functions) and validating their interfaces.
    *   **Automated Documentation Generation:** Extracting information programmatically.

#### Example: 

```python
import inspect
import sys
import os

class SomeBaseClass:
    pass

class MyModel(SomeBaseClass):
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100, optimizer: str = "Adam"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.weights = None

    def fit(self, X_train, y_train, batch_size: int = 32, verbose: bool = True) -> None:
        """
        Fits the model to the training data.
        :param X_train: Training features.
        :param y_train: Training labels.
        :param batch_size: Size of mini-batches.
        :param verbose: Whether to print verbose output.
        """
        print(f"Fitting {self.__class__.__name__} with LR={self.learning_rate}, Epochs={self.epochs}, Opt={self.optimizer}")
        print(f"  Fit called with batch_size={batch_size}, verbose={verbose}")
        # Simulate training...
        self.weights = [random.random() for _ in range(5)]

    def predict(self, X_test) -> List[float]:
        if self.weights is None:
            raise RuntimeError("Model not fitted.")
        print(f"Predicting with {self.__class__.__name__}")
        return [sum(self.weights) * random.random() for _ in X_test]

def get_hyperparameters_from_init(cls):
    """Extracts hyperparameters and their default values from a class's __init__ method."""
    signature = inspect.signature(cls.__init__)
    hyperparameters = {}
    for name, param in signature.parameters.items():
        if name == 'self':
            continue
        # Extract default values
        if param.default is not inspect.Parameter.empty:
            hyperparameters[name] = param.default
        else:
            hyperparameters[name] = "REQUIRED" # Indicate no default
    return hyperparameters

def find_subclasses_in_module(module, base_class):
    """Finds all subclasses of a given base_class within a module."""
    subclasses = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, base_class) and obj != base_class:
            subclasses.append(obj)
    return subclasses

# Usage 1: Inspecting hyperparameters
print("--- Inspecting Hyperparameters ---")
model_params = get_hyperparameters_from_init(MyModel)
print(f"Hyperparameters for MyModel: {model_params}")

# Example: Inspecting arguments of a method
fit_signature = inspect.signature(MyModel.fit)
print("\nSignature of MyModel.fit:")
for name, param in fit_signature.parameters.items():
    print(f"  {name}: kind={param.kind}, default={param.default}, annotation={param.annotation}")


# Usage 2: Dynamic model discovery
print("\n--- Dynamic Model Discovery ---")
# To make this work, MyModel needs to be defined in a module that can be imported
# For demonstration, we'll use the current module (__main__)
current_module = sys.modules[__name__]
discovered_models = find_subclasses_in_module(current_module, SomeBaseClass)

print(f"Discovered subclasses of SomeBaseClass: {[cls.__name__ for cls in discovered_models]}")

if MyModel in discovered_models:
    print("Found MyModel. Instantiating it dynamically.")
    DynamicModelClass = MyModel
    dynamic_instance = DynamicModelClass(learning_rate=0.005, epochs=50)
    dynamic_instance.fit([[1,2], [3,4]], [0,1], verbose=False)
    print(f"Dynamic instance predictions: {dynamic_instance.predict([[5,6]])}")

# Usage 3: Getting source code
print("\n--- Getting Source Code ---")
# print("\nSource code for MyModel.fit:")
# print(inspect.getsource(MyModel.fit))
```

### Conclusion

Mastering these advanced Python language features empowers you to write more robust, maintainable, and flexible AI/ML codebases. While external libraries handle the heavy lifting of numerical computation and deep learning, a strong grasp of these Python fundamentals allows you to: 

*   **Build Better Frameworks:** Design your own tools and libraries with clear APIs.
*   **Improve Code Quality:** Reduce boilerplate, ensure consistency, and catch errors early.
*   **Develop More Sophisticated Architectures:** Handle complex data pipelines, model configurations, and experiment management with native Python tools.
*   **Boost Productivity:** Write less code, with clearer intent.

These techniques don't replace NumPy or PyTorch; they complement them, enabling you to build the intelligent glue code and architectural patterns that make your AI/ML projects successful and scalable.

