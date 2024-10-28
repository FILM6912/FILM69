import torch
from torch import nn
import torch.optim as optim
import json,os
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

class AbnormalLayer(nn.Module):
    def __init__(self, input_size=2,  hidden_size=2028, num_layers=16, bias=False):
        super(AbnormalLayer, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, bias=bias))
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
        self.layers.append(nn.Linear(hidden_size, input_size,bias=bias))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))

        x = self.layers[-1](x)
        return x

class ClassificationLayer(nn.Module):
    def __init__(self, input_size=2, output_size=1, hidden_size=2028, num_layers=16, bias=False):
        super(ClassificationLayer, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, bias=bias))
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
        self.layers.append(nn.Linear(hidden_size, output_size,bias=bias))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))

        x = self.layers[-1](x)
        return x

class DataClassification(nn.Module):
    def __init__(self, 
    input_size=2,
    class_output_size=2,
    class_hidden_size=8,
    class_num_layers=2,
    abnormal_hidden_size=64,
    abnormal_num_layers=3,
    bias=True,
    device_map="auto"):
        super(DataClassification, self).__init__()
        self.input_size=input_size
        self.class_output_size=class_output_size
        self.class_hidden_size=class_hidden_size
        self.class_num_layers=class_num_layers
        self.abnormal_hidden_size=abnormal_hidden_size
        self.abnormal_num_layers=abnormal_num_layers
        self.bias=bias
        if device_map=="auto":self.device_map = "cuda" if torch.cuda.is_available() else "cpu"
        else:self.device_map=device_map
        self.classification=ClassificationLayer(input_size, class_output_size, class_hidden_size, class_num_layers, bias)
        self.abnormal=AbnormalLayer(input_size, abnormal_hidden_size, abnormal_num_layers, bias)
       
    def forward(self, x):
        x1=self.classification(x)
        x2=self.abnormal(x)
        return x1,x2
    
    def load_model(self,model_name,device_map="auto"):
        with open(model_name+"/config.json", 'r') as file:
            data = json.load(file)

        if device_map=="auto":self.device_map = "cuda" if torch.cuda.is_available() else "cpu"
        else:self.device_map=device_map
        
        self.classification=ClassificationLayer(data["input_size"], data["class_output_size"], data["class_hidden_size"], data["class_num_layers"], data["bias"])
        self.abnormal=AbnormalLayer(data["input_size"], data["abnormal_hidden_size"], data["abnormal_num_layers"], data["bias"])
        self.load_state_dict(torch.load(model_name+"/model.pth",weights_only=True))
        self = self.to(self.device_map)
    
    def predict(self,data, threshold=1):
        data=torch.tensor(data,dtype=torch.float32, device=self.device_map)
        abnormal=[]
        mse = nn.MSELoss()
        with torch.no_grad():
            class_,output = self(data)
            _, predicted = torch.max(class_, 1)
            for i in range(len(output)):
                loss = mse(output[i].unsqueeze(0), data[i].unsqueeze(0))
                abnormal.append(loss.item() > threshold)
                
        return predicted,abnormal
    
    def trainer(self,X_train,y_train,epochs = 600,learning_rate=0.001):
        X_train=torch.tensor(X_train,dtype=torch.float32, device=self.device_map)
        y_train=torch.tensor(y_train,dtype=torch.long, device=self.device_map)
    

        def accuracy_fn(y_true, y_pred):
            correct = torch.eq(y_true, y_pred).sum().item()
            acc = (correct / len(y_pred)) * 100
            return acc
        
        torch.manual_seed(42)
        loss_fn = nn.CrossEntropyLoss() 
        optimizer =  optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            out1,out2 = self(X_train)
            _, y_pred = torch.max(out1, 1)

            loss = loss_fn(out1, y_train)
            acc = accuracy_fn(y_true=y_train,y_pred=y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.eval()
            # with torch.inference_mode():
            #   test_logits,_ = model_3(X_test)
            #   test_logits=test_logits.squeeze()
            #   test_pred = torch.round(torch.sigmoid(test_logits))
            #   test_loss = loss_fn(test_logits, y_test)
            #   test_acc = accuracy_fn(y_true=y_test,y_pred=test_pred)

            if epoch % 50 == 0:
                # print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
                print(f"Epoch: {epoch}\t|\tLoss: {loss:.5f} \t|\t Accuracy: {acc:.2f}%")
                # print(f"Epoch: {epoch}")
                
    def total_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params
    
    def save_model(self,model_name):
        os.makedirs(model_name, exist_ok=True)
        torch.save(self.state_dict(), f'{model_name}/model.pth')
        
        data = {
        "input_size":self.input_size,
        "class_output_size":self.class_output_size,
        "class_hidden_size":self.class_hidden_size,
        "class_num_layers":self.class_num_layers,
        "abnormal_hidden_size":self.abnormal_hidden_size,
        "abnormal_num_layers":self.abnormal_num_layers,
        "bias":self.bias,
        }
        
        with open(model_name+"/config.json", 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Model saved to {model_name}.pth")
        

if __name__=="__main__":
    n_samples = 1000
    X, y = make_circles(n_samples,noise=0.03,random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42) # make the random split reproducible
    
    model_3 = DataClassification(
        input_size=2,
        class_output_size=2,
        class_hidden_size=8,
        class_num_layers=2,
        abnormal_hidden_size=64,
        abnormal_num_layers=3,
        bias=True
    )
    
    total_params=model_3.total_params()
    print(f"Total parameters: {total_params:,}")
    for name, param in model_3.named_parameters():
        print(name)
        
    model_3.trainer(X_train,y_train,epochs=600)
    y_pre,ad=model_3.predict(X_test)
    
    model_3.save_model("model")
    model = DataClassification()
    model.load_model("model",device_map="cuda")
    model.predict(X_test)