import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import json, os
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import trange



import plotly.graph_objects as go
def generate_spiral_data(classes, num_samples, dimensions=2, noise=0.2, plot=False):
    """
    Generate spiral data for 1D, 2D, or 3D.

    Args:
        classes (int): Number of classes.
        num_samples (int): Total number of samples.
        dimensions (int): Dimensionality of the data (1, 2, or 3).
        noise (float): Amount of noise to add to the data.
        plot (bool): Whether to plot the data (for 2D or 3D only).

    Returns:
        tuple: (X, y) where X is the data and y is the labels.
    """
    if dimensions < 1 or dimensions > 3:
        raise ValueError("Dimensions must be 1, 2, or 3.")

    X = []
    y = []
    points_per_class = num_samples // classes

    for class_number in range(classes):
        r = np.linspace(0.0, 1, points_per_class)
        t = np.linspace(class_number * (2 * np.pi / classes), 
                        (class_number + 1) * (2 * np.pi / classes), points_per_class)
        t += np.random.randn(points_per_class) * noise

        if dimensions == 1:
            X.extend(r.reshape(-1, 1))
        elif dimensions == 2:
            x1 = r * np.sin(t)
            x2 = r * np.cos(t)
            X.extend(np.c_[x1, x2])
        elif dimensions == 3:
            x1 = r * np.sin(t)
            x2 = r * np.cos(t)
            x3 = t
            X.extend(np.c_[x1, x2, x3])

        y.extend([class_number] * points_per_class)

    X = np.array(X)
    y = np.array(y)

    if plot:
        if dimensions == 2:
            fig = go.Figure()
            for class_number in range(classes):
                fig.add_trace(go.Scatter(
                    x=X[y == class_number, 0],
                    y=X[y == class_number, 1],
                    mode='markers',
                    name=f"Class {class_number}"
                ))
            fig.update_layout(
                title="2D Spiral Data",
                xaxis_title="X1",
                yaxis_title="X2",
                showlegend=True
            )
            fig.show()
        elif dimensions == 3:
            fig = go.Figure()
            for class_number in range(classes):
                fig.add_trace(go.Scatter3d(
                    x=X[y == class_number, 0],
                    y=X[y == class_number, 1],
                    z=X[y == class_number, 2],
                    mode='markers',
                    name=f"Class {class_number}"
                ))
            fig.update_layout(
                title="3D Spiral Data",
                scene=dict(
                    xaxis_title="X1",
                    yaxis_title="X2",
                    zaxis_title="X3"
                ),
                showlegend=True
            )
            fig.show()

    return X, y


class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, bias=True):
        super(DenseLayer, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, bias=bias))
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
        self.layers.append(nn.Linear(hidden_size, output_size, bias=bias))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


class DataClassification(nn.Module):
    def __init__(self, 
                 input_size,
                 class_output_size,
                 class_hidden_size,
                 class_num_layers,
                 abnormal_hidden_size,
                 abnormal_num_layers,
                 bias=True,
                 device_map="auto"):
        super(DataClassification, self).__init__()
        self.input_size = input_size
        self.class_output_size = class_output_size
        self.class_hidden_size = class_hidden_size
        self.class_num_layers = class_num_layers
        self.abnormal_hidden_size = abnormal_hidden_size
        self.abnormal_num_layers = abnormal_num_layers
        self.bias = bias
        self.device_map = "cuda" if torch.cuda.is_available() and device_map == "auto" else "cpu"
        self.classification = DenseLayer(input_size, class_output_size, class_hidden_size, class_num_layers, bias).to(self.device_map)
        self.abnormal = DenseLayer(input_size, input_size, abnormal_hidden_size, abnormal_num_layers, bias).to(self.device_map)

    def forward(self, x):
        x1 = self.classification(x)
        x2 = self.abnormal(x)
        return x1, x2

    def predict(self, data, threshold=1.0):
        data = torch.tensor(data, dtype=torch.float32, device=self.device_map)
        abnormal = []
        mse = nn.MSELoss()
        with torch.no_grad():
            class_, output = self(data)
            _, predicted = torch.max(class_, 1)
            for i in range(len(output)):
                loss = mse(output[i], data[i])
                abnormal.append(loss.item() > threshold)
        return predicted.cpu().numpy(), abnormal

    def trainer(self, X_train, y_train, epochs=600, learning_rate=0.001, logging_step=50):
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device_map)
        y_train = torch.tensor(y_train, dtype=torch.long, device=self.device_map)

        def accuracy_fn(y_true, y_pred):
            correct = torch.eq(y_true, y_pred).sum().item()
            return (correct / len(y_pred)) * 100

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in trange(epochs):
            self.train()
            class_out, _ = self(X_train)
            loss = loss_fn(class_out, y_train)
            _, y_pred = torch.max(class_out, 1)
            acc = accuracy_fn(y_train, y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % logging_step == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.5f} | Accuracy: {acc:.2f}%")

        print(f"Final | Loss: {loss.item():.5f} | Accuracy: {acc:.2f}%")

    def save_model(self, model_name):
        os.makedirs(model_name, exist_ok=True)
        torch.save(self.state_dict(), f"{model_name}/model.pth")
        config = {
            "input_size": self.input_size,
            "class_output_size": self.class_output_size,
            "class_hidden_size": self.class_hidden_size,
            "class_num_layers": self.class_num_layers,
            "abnormal_hidden_size": self.abnormal_hidden_size,
            "abnormal_num_layers": self.abnormal_num_layers,
            "bias": self.bias,
        }
        with open(f"{model_name}/config.json", "w") as f:
            json.dump(config, f, indent=4)
        print(f"Model saved to {model_name}")

    def load_model(self, model_name):
        with open(f"{model_name}/config.json", "r") as f:
            config = json.load(f)
        self.__init__(**config)
        self.load_state_dict(torch.load(f"{model_name}/model.pth", map_location=self.device_map))
        self.to(self.device_map)
        print(f"Model loaded from {model_name}")


if __name__ == "__main__":
    # Generate 2D spiral data for testing
    X, y = generate_spiral_data(classes=3, num_samples=1000, dimensions=2, noise=0.1, plot=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = DataClassification(
        input_size=2, class_output_size=3, 
        class_hidden_size=16, class_num_layers=2, 
        abnormal_hidden_size=16, abnormal_num_layers=2, bias=True
    )
    model.trainer(X_train, y_train, epochs=500, logging_step=50)
    model.save_model("spiral_model")

    # Load and predict
    model.load_model("spiral_model")
    predictions, abnormalities = model.predict(X_test)
    print(f"Predictions: {predictions}")
    print(f"Abnormalities: {abnormalities}")
