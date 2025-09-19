from pathlib import Path
import time
import pickle

import idx2numpy as idx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

#default parameters
SPLIT = 0.95
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCH_NUMS = 70
NUM_CLASSES  = 51
NEURONS_L1 = 256
NEURONS_L2 = 128

#We're building a small neural network on the mnist dataset to recognize individual characters
#as an abstraction to eventually build a full mathematical-expression evaluator.
#this first step is done without pytorch, for the sake of understanding the base network. 
    
def init_weights_and_biases(neurons_l1, neurons_l2, num_classes):   
    #weight matrices
    W0 = np.random.randn(neurons_l1, 784) * np.sqrt(2.0 / 784)
    W1 = np.random.randn(neurons_l2, neurons_l1) * np.sqrt(2.0 / neurons_l1)
    W2 = np.random.randn(NUM_CLASSES, neurons_l2) * np.sqrt(2.0 / neurons_l2)

    
    #bias matrices
    b0 = np.zeros((neurons_l1, 1))
    b1 = np.zeros((neurons_l2, 1))
    b2 = np.zeros((51, 1))
    
    return W0, W1, W2, b0, b1, b2


def reLu(x):
    return np.maximum(0, x)

#no need to implement temperature, as we always want the best prediction
def softmax(array):
    shifted_array = array - np.max(array, axis=0, keepdims=True)
    exp_array = np.exp(shifted_array)
    return exp_array / np.sum(exp_array, axis=0, keepdims=True)

def forward_prop(input, W0, W1, W2, b0, b1, b2):
    Z1 = W0.dot(input) + b0
    A1 = reLu(Z1)
    
    Z2 = W1.dot(A1) + b1
    A2 = reLu(Z2)
    
    Z3 = W2.dot(A2) + b2
    A3 = softmax(Z3)
    
    #return the output nodes
    return Z1, Z2, Z3, A1, A2, A3

def one_hot(Y):
    one_hot = np.zeros((NUM_CLASSES, Y.size))
    one_hot[Y, np.arange(Y.size)] = 1
    return one_hot


def reLu_derivative(x):
    #if relu > 0 its derivative is dx/dx = 1, else its d0/dx = 0
    return (x > 0)

def back_prop(Z1, Z2, A1, A2, A3, W1, W2, input, target_output):
    target_output = one_hot(target_output)
    
    #update the third layer
    dZ3 = target_output - A3 #for softmax and MSE
    #TODO
    dW2 = dZ3.dot(A2.T)
    db2 = np.sum(dZ3, axis=1, keepdims=True)    
    #update the second layer
    dZ2 = W2.T.dot(dZ3) * reLu_derivative(Z2)
    dW1 = dZ2.dot(A1.T)
    db1 = np.sum(dZ2, axis=1, keepdims=True)    
    #update the first layer
    dZ1 = W1.T.dot(dZ2) * reLu_derivative(Z1)
    dW0 = dZ1.dot(input.T)
    db0 = np.sum(dZ1, axis=1, keepdims=True)
    
    return dW0, dW1, dW2, db0, db1, db2


def update_parameters(W0, W1, W2, b0, b1, b2, dW0, dW1, dW2, db0, db1, db2, learning_rate):
    W0 += learning_rate * dW0
    W1 += learning_rate * dW1
    W2 += learning_rate * dW2
    
    b0 += learning_rate * db0
    b1 += learning_rate * db1
    b2 += learning_rate * db2
    
    return W0, W1, W2, b0, b1, b2
    
    
def batch_cost(A3, target_output, num_classes):
    one_hot_encoded = one_hot(target_output)
    epsilon = 1e-15
    A3 = np.clip(A3, epsilon, 1 - epsilon)
    loss = -np.sum(one_hot_encoded * np.log(A3)) / target_output.size
    return loss



def outputIntToAscii(n): 
    label_to_ascii = {
        0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57,
        10: 65, 11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74,
        20: 75, 21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84,
        30: 85, 31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101,
        40: 102, 41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116, 47: 42, 48: 43, 49: 45,
        50: 47
    }
    return chr(label_to_ascii[n])


def train(X_train, Y_train, l1, l2, num_classes, epochs, batch_size, lr):
    #W stands for Weights, b for bias
    start_time = time.time()
    W0, W1, W2, b0, b1, b2 = init_weights_and_biases(l1, l2, num_classes)
    costs = []
    accuracies = []
    times = []
    m = X_train.shape[1]  #number of training examples
    
    for epoch in range(epochs):
        epoch_cost = 0
        correct_predictions = 0
        
        #process batches
        for i in range(0, m, batch_size):
            end = min(i + batch_size, m)
            X_batch = X_train[:, i:end]
            Y_batch = Y_train[i:end]
            
            #forward propagation
            Z1, Z2, Z3, A1, A2, A3 = forward_prop(X_batch, W0, W1, W2, b0, b1, b2)
            
            #cost computation
            epoch_cost += batch_cost(A3, Y_batch, num_classes)
            
            #compute accuracy
            predictions = np.argmax(A3, axis=0)
            correct_predictions += np.sum(predictions == Y_batch)
            
            #backward propagation
            dW0, dW1, dW2, db0, db1, db2 = back_prop(Z1, Z2, A1, A2, A3, W1, W2, X_batch, Y_batch)
            
            #update parameters
            W0, W1, W2, b0, b1, b2 = update_parameters(W0, W1, W2, b0, b1, b2, 
                                                       dW0, dW1, dW2, db0, db1, db2, 
                                                       lr)

        epoch_cost = epoch_cost / m
        epoch_accuracy = correct_predictions / m
        costs.append(epoch_cost)
        accuracies.append(epoch_accuracy)
        times.append(time.time() - start_time)
        
        # Check for early stopping
        if epoch >= 5:
            recent_accuracies = accuracies[-5:]
            if max(recent_accuracies) - min(recent_accuracies) < 0.005 and accuracies[-1] < accuracies[-5]:
                print("Early stopping triggered.")
                break
        
        print('|', end='', flush=True)
        # Print progress every few epochs
        if (epoch + 1) % epochs == 0:
            print('\\')
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Cost: {epoch_cost:.4f}")
            print(f"Training Accuracy: {epoch_accuracy:.2%}")
            # Print a few predictions vs actual
            test_idx = np.random.randint(0, m)
            _, _, _, _, _, A3 = forward_prop(X_train[:, test_idx:test_idx+1], W0, W1, W2, b0, b1, b2)
            pred = outputIntToAscii(np.argmax(A3))
            actual = outputIntToAscii(Y_train[test_idx])
            print(f"Sample prediction: {pred}, Actual: {actual}\n")
    
    return W0, W1, W2, b0, b1, b2, costs, accuracies, times
    
    
def load_n_lines_from_dataset(n, file_path: Path = DATA_DIR / "processed_dataset.csv"):
    try:
        #137_600 is the size of the dataset
        #read_csv automatically skips the header row!
        df = pd.read_csv(file_path, skiprows=lambda i: i > 0 and np.random.rand() > n / 137_600, nrows=n)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    return df


def split_data(data, train_size):
    np.random.shuffle(data)
    
    #number of training examples
    n_train = int(data.shape[0] * train_size)
    
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    #transpose to have features as columns
    X_train = train_data[:, 1:].T / 255.0  #normalize
    Y_train = train_data[:, 0].T
    
    X_test = test_data[:, 1:].T / 255.0    #normalize
    Y_test = test_data[:, 0].T
    
    return (X_train, Y_train, X_test, Y_test)

def save_weights_and_biases(W0, W1, W2, b0, b1, b2, split, lr, bs, en, nc, nl1, nl2):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filename = MODELS_DIR / f"weights_biases_S{split}_LR{lr}_BS{bs}_E{en}_NC{nc}_L1{nl1}_L2{nl2}.npz"
    np.savez(str(filename), W0=W0, W1=W1, W2=W2, b0=b0, b1=b1, b2=b2)

def main(train_test_split, lr, bs, epochs, num_classes, l1, l2):
    
    #transfer the csv to a giant matrix
    cache_file = DATA_DIR / "processed_dataset_cache.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            df = pickle.load(f)
    else:
        df = load_n_lines_from_dataset(137_600)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
    data = np.array(df)
    
    #define the training and testing datasets
    X_train, Y_train, X_test, Y_test = split_data(data, train_test_split)
    
    #train the model
    W0, W1, W2, b0, b1, b2, costs, tr_acc, times = train(X_train,
                                                             Y_train,
                                                             l1,
                                                             l2,
                                                             num_classes,
                                                             epochs,
                                                             bs,
                                                             lr
                                                             )
    
    # Test the model on the test set
    _, _, _, _, _, A3_test = forward_prop(X_test, W0, W1, W2, b0, b1, b2)
    test_predictions = np.argmax(A3_test, axis=0)
    te_acc = np.sum(test_predictions == Y_test) / Y_test.size
    
    print(f"Test Accuracy: {te_acc:.2%}")
    print('\n')
    
    
    return costs, tr_acc, te_acc, times, W0, W1, W2, b0, b1, b2
    

if __name__ == "__main__":
    costs, tr_acc, te_acc, times, W0, W1, W2, b0, b1, b2 = main(SPLIT, LEARNING_RATE, BATCH_SIZE, EPOCH_NUMS, NUM_CLASSES, NEURONS_L1, NEURONS_L2)
    save_weights_and_biases(W0, W1, W2, b0, b1, b2, SPLIT, LEARNING_RATE, BATCH_SIZE, EPOCH_NUMS, NUM_CLASSES, NEURONS_L1, NEURONS_L2)
