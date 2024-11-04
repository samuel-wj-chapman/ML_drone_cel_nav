import os
import cv2
import yaml
import torch
import random
import numpy as np
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # For progress bars
import matplotlib.pyplot as plt
from clearml import Task

task = Task.init(project_name='ML_drone_cel_nav', task_name='mse loss + new time norm + batch norm and weight decay')

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, dim=(224, 224), channels=1):
    """
    Loads a single image as a Numpy array and resizes it as desired.
    """
    if channels == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, dim)
    return image

def build_input(image_dir, channels=1, dim=(224,224)):
    """
    Loads all of the images into a single numpy array.
    """
    X = []
    files = os.listdir(image_dir)
    for file in tqdm(files, desc='Loading images'):
        if file.endswith('.png'):
            image_path = os.path.join(image_dir, file)
            image = load_image(image_path, channels=channels, dim=dim)
            X.append(image)
    return (np.array(X) / 255.0)

from concurrent.futures import ThreadPoolExecutor


def load_images_parallel(image_dir, channels=1, dim=(224, 224)):
    """
    Loads all images in parallel using ThreadPoolExecutor.
    """
    files = [file for file in os.listdir(image_dir) if file.endswith('.png')]
    
    def load_and_process(file):
        image_path = os.path.join(image_dir, file)
        return load_image(image_path, channels=channels, dim=dim)
    
    with ThreadPoolExecutor() as executor:
        images = list(tqdm(executor.map(load_and_process, files), total=len(files), desc='Loading images'))
    
    return np.array(images) / 255.0




def build_labels(image_dir):
    """
    Parses file names to extract latitude, longitude, and time.
    """
    files = os.listdir(image_dir)
    y = []
    times = []
    for file in tqdm(files, desc='Parsing labels'):
        if file.endswith('.png'):
            file_split = file.split('+')
            lat = float(file_split[0])
            long = float(file_split[1])
            time_str = file_split[2].split('.')[0]
            # Adjust format string to match the time format in filenames
            # For time format '2024-11-10T21:17:11', the format string is '%Y-%m-%dT%H:%M:%S'
            time_obj = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
            y.append((lat, long))
            times.append(time_obj)
    return np.array(y), np.array(times)

def normalize_y(y, lat_min, lat_range, long_min, long_range):
    """
    Converts latitudes and longitudes to values bounded by [0,1]
    """
    y_norm = np.zeros(y.shape)
    y_norm[:,0] = (y[:,0] - lat_min) / lat_range
    y_norm[:,1] = (y[:,1] - long_min) / long_range
    return y_norm

def normalize_times(times, t_min, t_max):
    """
    Converts times to a float bounded by [0,1]
    """
    t_min = np.datetime64(t_min)
    t_max = np.datetime64(t_max)
    # Ensure t_max is after t_min
    assert t_max > t_min, "t_max should be after t_min"
    # Convert time range to total seconds
    time_range = (t_max - t_min) / np.timedelta64(1, 's')
    # Convert times to seconds from t_min
    seconds_from_t0 = (times - t_min) / np.timedelta64(1, 's')
    # Normalize
    normalized_times = seconds_from_t0 / time_range
    # Ensure normalized times are between 0 and 1
    assert np.all(normalized_times >= 0) and np.all(normalized_times <= 1), "Normalized times not in [0,1]"
    return normalized_times


class CelestialDataset(Dataset):
    def __init__(self, images, times, labels):
        self.images = images
        self.times = times
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        time = self.times[idx]
        label = self.labels[idx]
        return image, time, label
    

class CelestialCNN(nn.Module):
    def __init__(self):
        super(CelestialCNN, self).__init__()
        # Convolutional layers with 'same' padding
        self.conv1 = nn.Conv2d(1, 5, kernel_size=10, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(5, 1, kernel_size=10, padding='same')
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Dynamically compute the input size for fc1
        self.feature_size = self._get_conv_output_size()
        self.fc1 = nn.Linear(self.feature_size + 1, 256)  # +1 for the time feature
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def _get_conv_output_size(self):
        """
        Computes the size of the output from the convolutional layers.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 224, 224)
            x = self.conv1(dummy_input)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.flatten(x)
            feature_size = x.shape[1]
        return feature_size

    def forward(self, image, time):
        x = self.conv1(image)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        time = time.view(-1, 1)
        x = torch.cat((x, time), dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class CelestialCNN(nn.Module):
    def __init__(self):
        super(CelestialCNN, self).__init__()
        # Convolutional layers with 'same' padding
        self.conv1 = nn.Conv2d(1, 5, kernel_size=10, padding='same')
        self.bn1 = nn.BatchNorm2d(5)  # Add batch normalization after conv1
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(5, 1, kernel_size=10, padding='same')
        self.bn2 = nn.BatchNorm2d(1)  # Add batch normalization after conv2
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        
        # Dynamically compute the input size for fc1
        self.feature_size = self._get_conv_output_size()
        self.fc1 = nn.Linear(self.feature_size + 1, 256)  # +1 for the time feature
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def _get_conv_output_size(self):
        """
        Computes the size of the output from the convolutional layers.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 224, 224)
            x = self.conv1(dummy_input)
            x = self.bn1(x)  # Apply batch normalization
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)  # Apply batch normalization
            x = self.relu2(x)
            x = self.flatten(x)
            feature_size = x.shape[1]
        return feature_size

    def forward(self, image, time):
        x = self.conv1(image)
        x = self.bn1(x)  # Apply batch normalization
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)  # Apply batch normalization
        x = self.relu2(x)
        x = self.flatten(x)
        time = time.view(-1, 1)
        x = torch.cat((x, time), dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def haversine_loss(y_true, y_pred, denorm, R=3443.92):
    """
    Computes the mean Haversine distance between true and predicted coordinates.
    Uses PyTorch operations to ensure compatibility with autograd.
    """
    lat_min, lat_range, long_min, long_range = denorm

    # Denormalize the coordinates
    lat1 = y_true[:,0] * lat_range + lat_min
    lat2 = y_pred[:,0] * lat_range + lat_min
    long1 = y_true[:,1] * long_range + long_min
    long2 = y_pred[:,1] * long_range + long_min

    # Convert to radians
    pi = torch.tensor(np.pi).to(y_true.device)
    phi1 = lat1 * pi / 180.0
    phi2 = lat2 * pi / 180.0
    delta_phi = (lat2 - lat1) * pi / 180.0
    delta_lambda = (long2 - long1) * pi / 180.0

    # Haversine formula
    a = torch.sin(delta_phi / 2.0) ** 2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda / 2.0) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    # Distance in nautical miles
    d = R * c
    return torch.mean(d)

def main():
    # Configuration variables
    train_val_path = '/dataset/train'  # Update this path
    test_path = '/dataset/test'  # Update this path
    #latstart = 50
    #latend = 61
    #longstart = -15
    #longend = 6

    latstart = 50
    latend = 54
    longstart = -5
    longend = -1

    dtstart = '2024-11-10T20:00:00'
    dtend = '2024-11-11T00:00:00'
    tv_split = 0.1
    lr = 0.002 #changed from 001
    epochs = 1000  # Set the number of epochs
    batch_size = 32  # Set the batch size

    # Derived variables
    latrange = latend - latstart
    longrange = longend - longstart
    denorm_params = (latstart, latrange, longstart, longrange)

    # Load and preprocess data
    print('Loading and preprocessing training and validation sets...')
    X = load_images_parallel(train_val_path, channels=1, dim=(224, 224)) #purely to speed up processing
    #X = build_input(train_val_path, channels=1, dim=(224,224))
    X = np.expand_dims(X, 1)  # Add channel dimension
    y, times = build_labels(train_val_path)
    y_norm = normalize_y(y, latstart, latrange, longstart, longrange)
    times_norm = normalize_times(times.astype('datetime64'), dtstart, dtend)
    print("Normalized Times min:", times_norm.min(), "max:", times_norm.max())


    # Split into training and validation sets
    print('Splitting into training and validation sets...')
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - tv_split))
    train_indices = indices[:split]
    val_indices = indices[split:]

    x_train = X[train_indices]
    y_train = y_norm[train_indices]
    t_train = times_norm[train_indices]

    x_val = X[val_indices]
    y_val = y_norm[val_indices]
    t_val = times_norm[val_indices]

    from torch.utils.data import DataLoader

    # Create Datasets and DataLoaders
    train_dataset = CelestialDataset(torch.tensor(x_train, dtype=torch.float32),
                                    torch.tensor(t_train, dtype=torch.float32),
                                    torch.tensor(y_train, dtype=torch.float32))
    val_dataset = CelestialDataset(torch.tensor(x_val, dtype=torch.float32),
                                torch.tensor(t_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))

    # Use multiple workers for data loading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # Check a batch of images
    images, times_batch, labels = next(iter(train_loader))
    print("Images shape:", images.shape)
    print("Images min:", images.min().item(), "max:", images.max().item())




    # Convert tensor to numpy array and remove channel dimension
    img = images[0].squeeze().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {labels[0]}")
    plt.show()


    print("Labels min:", y_train.min(axis=0), "max:", y_train.max(axis=0))
    print("Times min:", t_train.min(), "max:", t_train.max())


    from torch.optim.lr_scheduler import ReduceLROnPlateau

    # Initialize model, loss function, and optimizer
    model = CelestialCNN().to(device)
    # Use haversine_loss as the criterion
    #def criterion(y_pred, y_true):
    #    return haversine_loss(y_true, y_pred, denorm_params)
    criterion = nn.MSELoss()
    # Add weight decay to the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)

    patience = 40  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    print('Training the model...')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # Use tqdm to wrap the DataLoader
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]', leave=False)
        for images, times_batch, labels in train_loader_tqdm:
            images = images.to(device)
            times_batch = times_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, times_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Update progress bar
            train_loader_tqdm.set_postfix({'Loss': f'{loss.item():.4f}'})
        task.get_logger().report_scalar("Loss", "Training Loss", iteration=epoch, value=running_loss/len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, times_batch, labels in val_loader:
                images = images.to(device)
                times_batch = times_batch.to(device)
                labels = labels.to(device)

                outputs = model(images, times_batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Training Loss: {running_loss/len(train_loader):.6f}, "
              f"Validation Loss: {val_loss/len(val_loader):.6f}")
        task.get_logger().report_scalar("Loss", "Validation Loss", iteration=epoch, value=val_loss/len(val_loader))
        scheduler.step(val_loss)
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_celestial_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break




    # Save the trained model
    torch.save(model.state_dict(), 'celestial_model.pth')

    # Test the model
    print('Loading and preprocessing test set...')
    X_test = build_input(test_path, channels=1, dim=(224,224))
    X_test = np.expand_dims(X_test, 1)  # Add channel dimension
    y_test, times_test = build_labels(test_path)
    y_test_norm = normalize_y(y_test, latstart, latrange, longstart, longrange)
    times_test_norm = normalize_times(times_test.astype('datetime64'), dtstart, dtend)

    test_dataset = CelestialDataset(torch.tensor(X_test, dtype=torch.float32),
                                    torch.tensor(times_test_norm, dtype=torch.float32),
                                    torch.tensor(y_test_norm, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Evaluate the model on the test set and compute average error in miles
    print('Evaluating on test set...')
    model.eval()
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for images, times_batch, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            times_batch = times_batch.to(device)
            labels = labels.to(device)

            outputs = model(images, times_batch)
            predictions.append(outputs.cpu())
            ground_truth.append(labels.cpu())

    predictions = torch.cat(predictions, dim=0)
    ground_truth = torch.cat(ground_truth, dim=0)

    # Compute average error in miles using Haversine formula
    avg_error_nm = haversine_loss(ground_truth, predictions, denorm_params).item()
    avg_error_miles = avg_error_nm   # Convert nautical miles to miles

    print(f"Average error on test set: {avg_error_miles:.2f} n miles")
    task.get_logger().report_scalar("Accuracy", "NM", value=avg_error_miles, iteration=epoch)


if __name__ == '__main__':
    main()

