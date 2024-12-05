import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from collections import defaultdict
import pickle as pkl
import pandas as pd
import sys





torch.manual_seed(42)  
np.random.seed(42)



class GMM:

    """
    
        GMM class for performing clustering using the expectation maximization algorithm
    
    """

    def __init__(self, X, init_means, index_to_label):

        self.index_to_label = index_to_label

        self.X = X
        self.means = init_means
        self.n = X.shape[0]
        self.k = init_means.shape[0]
        self.d = X.shape[1]
        self.weights = np.array([1 / self.k for _ in range(self.k)])
        self.covs = np.array([np.eye(self.d)] * self.k)
        self.taus = None

    def gaussian(self, x, mu, sigma):
        sigma_inv = np.linalg.inv(sigma)
        delta = x - mu
        z = -0.5 * np.dot(delta, sigma_inv @ delta)
        Z = ((2 * np.pi) ** (self.d / 2)) * np.sqrt(np.linalg.det(sigma))
        return (1 / Z) * np.exp(z)

    def expectation(self):
        self.taus = np.zeros((self.n, self.k))
        for i in range(self.n):
            for j in range(self.k):
                self.taus[i][j] = self.weights[j] * self.gaussian(self.X[i], self.means[j], self.covs[j])
        self.taus /= self.taus.sum(axis=1, keepdims=True)

    def maximization(self):
        Nk = self.taus.sum(axis=0)
        self.means = np.dot(self.taus.T, self.X) / Nk[:, np.newaxis]
        for j in range(self.k):
            diff = self.X - self.means[j]
            self.covs[j] = (self.taus[:, j][:, np.newaxis] * diff).T @ diff / Nk[j]
        self.weights = Nk / self.n

    def log_likelihood(self):
        ll = 0
        for i in range(self.n):
            prob = 0
            for j in range(self.k):
                prob += self.weights[j] * self.gaussian(self.X[i], self.means[j], self.covs[j])
            ll += np.log(prob)
        return ll

    def fit(self, tol=1e-3, max_iter=50):
        log_likelihoods = []
        for iter in range(max_iter):
            self.expectation()
            self.maximization()
            ll = self.log_likelihood()
            log_likelihoods.append(ll)
            if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                break

            print(f"iter : {iter}, ll : {ll}")

        return log_likelihoods


    def predict(self, x):
        responsibilities = np.array([
            self.weights[j] * self.gaussian(x, self.means[j], self.covs[j])
            for j in range(self.k)
        ])
        responsibilities /= responsibilities.sum()  
        return np.argmax(responsibilities)  



    def display(self):
        from matplotlib.patches import Ellipse
        colors = plt.cm.get_cmap('viridis', self.k)
        cluster_labels = [self.predict(x) for x in self.X]


        for i in range(self.k):
            cluster_points = self.X[np.array(cluster_labels) == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=5, color=colors(i), label=f'Cluster {self.index_to_label[i]}')

            plt.scatter(self.means[i, 0], self.means[i, 1], s=100, c='red', marker='x', label=f'Mean {self.index_to_label[i]}' if i == 0 else None)

            eigenvalues, eigenvectors = np.linalg.eigh(self.covs[i])
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            

            width, height = 2.5 * np.sqrt(eigenvalues)

            ellipse = Ellipse(
                xy=self.means[i], width=width, height=height, angle=angle,
                edgecolor="black", facecolor='none', linewidth=3
            )
            plt.gca().add_patch(ellipse)

        plt.legend()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Gaussian Mixture Model Clustering with Ellipses')
        plt.savefig("gmm.png")
        plt.clf()



class MNIST(Dataset):

    def __init__(self, data_path, keep_labels=[1,4,8]):
        data = np.load(data_path)
        images = data['data']

        self.images = []
        for image in images:
       
                self.images.append(torch.from_numpy(image).float() / 255)  


        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.images[idx].view(1, 28, 28)  
    

class VALMNIST(Dataset):

    def __init__(self, data_path, keep_labels=[1,4,8]):
        data = np.load(data_path)
        images = data['data']
        labels = data['labels']
        
        self.images = []
        self.labels = []
        for image, label in zip(images, labels):
            if label in keep_labels:
                self.images.append(torch.from_numpy(image).float() / 255)  
                self.labels.append(label)

        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.images[idx].view(1, 28, 28) , self.labels[idx]  


class VALMNISTIMAGES(Dataset):

    def __init__(self, data_path):
        data = np.load(data_path)
        images = data['data']
 
        
        self.images = []
        self.labels = []
        for image in images:

                self.images.append(torch.from_numpy(image).float() / 255)  


        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.images[idx].view(1, 28, 28) 



import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        

        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)   
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 
        self.fc_mu = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256) , 
            nn.ReLU(), 
            nn.Linear(256, 128) , 
            nn.ReLU(), 
            nn.Linear(128, 2)

        ) 
        self.fc_logvar = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256) , 
            nn.ReLU(), 
            nn.Linear(256, 128) , 
            nn.ReLU(), 
            nn.Linear(128, 2)

        
        )
        

        self.fc_dec = nn.Sequential( 
            nn.Linear(2, 128), 
            nn.ReLU(), 
            nn.Linear(128, 256), 
            nn.ReLU(), 
            nn.Linear(256,  64 * 7 * 7)
        )
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)  
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)   

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_dec(z))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        x = x.view(-1,784)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), (mu, logvar)




def loss_function(recon_x, x, mu, logvar, beta = 0.5):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD







def show_reconstruction(model, val_loader, device,  n=15):
    model.eval()
    cnt = 0
    for data, labels in val_loader : 
        print(labels)
        
        data = data.to(device)
        recon_data, _ = model(data)

        print(data.shape, recon_data.shape)
        
        fig, axes = plt.subplots(2, 2)
        for i in range(1):
    
            axes[0, i].imshow(data[i].cpu().numpy().squeeze(), cmap='gray')
            axes[0, i].axis('off')

            axes[1, i].imshow(recon_data[i].cpu().view(28, 28).detach().numpy(), cmap='gray')
            axes[1, i].axis('off')
        plt.savefig(f"recon{cnt}.png")
        plt.clf()
        cnt+=1

        recon_data = recon_data.view(-1,1,28,28)
        print(data.shape, recon_data.shape)

        for o,r in zip(data, recon_data) : 

            o = o.view(28,28)
            r = r.view(28,28)
            o = o.cpu().detach().numpy()
            r = r.cpu().detach().numpy()
            print(o.shape, r.shape)
            mse = np.mean((o- r) ** 2)


            ssim_value, _ = ssim(o, r, full=True, data_range=1.1)
            print(1-mse,ssim_value)
        


import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_2d_manifold(vae, latent_dim=2, n=20, digit_size=28, device='cuda'):
    figure = np.zeros((digit_size * n, digit_size * n))

    # Generate a grid of values between 0.05 and 0.95 percentiles of a normal distribution
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    vae.eval()  # Set VAE to evaluation mode
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], device=device, dtype=torch.float32)
                
                # Pass z to VAE Decoder 
                # Write your code here
                digit = vae.decode(z_sample).view(28,28).cpu()

                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit


    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gnuplot2')
    plt.axis('off')
    plt.savefig("manifold.png")
    plt.clf()




def train(data_path, val_path,  model_path , params_path, device = "mps" ):

    """

    data_path : path to training dataset (unlabelled)
    val_path : path to validation dataset for mean initialization of gmm (labelled)
    
    model_path : path where vae model is to be saved
    params_path : path where gmm model is to be saved

    
    
    """

    mnist = MNIST(data_path)
    val_mnist = VALMNIST(val_path)


    vae = VAE().to(device)
    vae.train()

    batch_size = 32
    epochs = 8
    learning_rate = 3e-4
    beta = 0.55

    train_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    scheduler = scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)


    for epoch in range(epochs):
        train_loss = 0
        for images in train_loader:
            images = images.to(device)
            recon, (mu, logvar) = vae(images)
            loss = loss_function(recon, images, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}")


    torch.save(vae.state_dict(), model_path)

    
    val_loader = DataLoader(val_mnist, 15)

    # show_reconstruction(vae, val_loader,device)
    # plot_2d_manifold(vae, latent_dim=2, n=20, digit_size=28, device=device)


    points = []
    label_images = defaultdict(list)


    with torch.no_grad() : 

        loader = DataLoader(mnist, 1)
        

        for image in loader : 

            image = image.to(device)
            point , _ = vae.encode(image)
            point = point.detach().cpu().numpy()
            point = point[0]
            points.append(point)


        loader = DataLoader(val_mnist, 1)

        for image, label in loader : 

            image = image.to(device)
            point , _ = vae.encode(image)
            point = point.detach().cpu().numpy()
            point = point[0]
            label_images[int(label)].append(point)


    means = []
    index_to_label = defaultdict(int) 


    for label in label_images.keys() : 
        idx = len(means)
        mean = np.array([0,0], dtype=np.float64)
        for point in label_images[label] : 
            mean += point 

        mean /= len(label_images[label])
            
        means.append(mean)
        index_to_label[idx] = label

    means = np.array(means)
    points = np.array(points)

    gmm = GMM(points, means, index_to_label)
    log_likelihoods = gmm.fit()
    # gmm.display()

    with open(params_path, "wb") as f : 
        pkl.dump(gmm, f)




def reconstruction(data_path, model_path, save_path = "vae_reconstructed.npz", device = "mps") : 
    
    vae = VAE()
    vae.load_state_dict(torch.load(model_path))
    vae = vae.to(device)
    vae.eval()
    val_mnist = VALMNISTIMAGES(data_path)
    val_loader = DataLoader(val_mnist, batch_size=1)
    recons = []
    cnt = 0
    for data in val_loader : 

        
        data = data.to(device)
        recon_data, _ = vae(data)
        recon = recon_data.view(28,28)
        recon = recon.cpu().detach().numpy()
        recons.append(recon)        
        

    recons = np.array(recons)
    np.savez(save_path, data = recons)
    



        


def classification(data_path, model_path, params_path, save_path = "vae.csv", device = "mps") :

    vae = VAE()
    vae.load_state_dict(torch.load(model_path))
    vae = vae.to(device)
    vae.eval()

    with open(params_path, "rb") as f : 
        gmm = pkl.load(f)

    data = np.load(data_path)
    images = data['data']

    with torch.no_grad() : 

        with open(save_path, "w") as f : 

            f.write("Predicted_Label\n")


            for image in images : 

                image = torch.from_numpy(image).float() / 255
                image = image.to(device)
                point , _ = vae.encode(image.view(-1,28,28))
                point = point.detach().cpu().numpy()
                point = point[0]
                label = gmm.index_to_label[gmm.predict(point)]
                f.write(f"{label}\n")


     
def evaluate_gmm_performance(data_path, csv_path):

    data = np.load(data_path)
    labels_true = data['labels']
    labels_pred = np.array(pd.read_csv(csv_path))
    
    accuracy = accuracy_score(labels_true, labels_pred)
    precision_macro = precision_score(labels_true, labels_pred, average='macro')  
    recall_macro = recall_score(labels_true, labels_pred, average='macro') 
    f1_macro = f1_score(labels_true, labels_pred, average='macro') 

   
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }



if __name__ == "__main__":

    

    device = "cpu"
    if torch.cuda.is_available() : 
        device = "cuda"
    if torch.mps.is_available() : 
        device = "mps"

    args = sys.argv

    n = len(args)

    if n == 6 : 
        train(data_path=args[1], val_path=args[2], model_path=args[4],params_path= args[5], device=device)

    elif n == 5 : 
        classification(data_path=args[1], model_path=args[3],params_path= args[4],device=device)

    elif n == 4 : 
        reconstruction(data_path=args[1], model_path=args[3], device=device)

    else : 
        data_path = "/Users/poojanshah/Desktop/aia3/mnist_1_4_8_train.npz"
        val_path = "/Users/poojanshah/Desktop/aia3/mnist_1_4_8_val_recon.npz"
        model_path = "/Users/poojanshah/Desktop/aia3/model.pth"
        params_path = "/Users/poojanshah/Desktop/aia3/params.pkl"
        csv_path = "/Users/poojanshah/Desktop/aia3/vae.csv"

        train(data_path=data_path, val_path=val_path, model_path=model_path, params_path=params_path)
        classification(data_path, model_path, params_path)
        res=evaluate_gmm_performance(data_path, csv_path)
        print(res)

    
