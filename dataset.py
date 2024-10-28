import pandas as pd
import numpy as np
import idx2numpy as idx
from PIL import Image
import os

def load_emnist():
    '''Load the balanced EMNIST dataset.'''
    emnist_train = idx.convert_from_file('data/emnist/emnist-balanced-train-images-idx3-ubyte')
    emnist_train_labels = idx.convert_from_file('data/emnist/emnist-balanced-train-labels-idx1-ubyte')
    emnist_test = idx.convert_from_file('data/emnist/emnist-balanced-test-images-idx3-ubyte')
    emnist_test_labels = idx.convert_from_file('data/emnist/emnist-balanced-test-labels-idx1-ubyte')
    
    # Reshape the images to 2D arrays (num_images, pixels)
    emnist_train = emnist_train.reshape(emnist_train.shape[0], -1)
    emnist_test = emnist_test.reshape(emnist_test.shape[0], -1)
    
    return emnist_train, emnist_train_labels, emnist_test, emnist_test_labels

def process_bhmsds():
    input_dir = 'data/bhmsds_symbols'
    output_dir = 'data/bhmsds_processed_symbols'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('L')
            
            img = img.resize((28, 28))
            pixels = np.array(img)
            if (pixels[0, :] < 255).any() or (pixels[-1, :] < 255).any() or \
               (pixels[:, 0] < 255).any() or (pixels[:, -1] < 255).any():
                img = Image.fromarray(np.pad(pixels, pad_width=1, mode='constant', 
                                           constant_values=255)[1:-1, 1:-1])
            
            # Invert the image
            img = Image.eval(img, lambda x: 255 - x)
            
            # Save the processed image
            img.save(os.path.join(output_dir, filename))

def get_bhmsds_label(filename):
    '''Get the label of a BHMSDS image.'''
    type = filename.split('-')[0]
    if type == 'dot':
        return 47
    if type == 'plus':
        return 48
    if type == 'minus':
        return 49
    if type == 'slash':
        return 50
    else:
        raise ValueError(f"Invalid bhmsds file name: {type}")

def load_bhmsds():
    '''Load the BHMSDS dataset.'''
    bhmsds_train = []
    bhmsds_train_labels = []
    
    for filename in os.listdir('data/bhmsds_processed_symbols'):
        if filename.endswith('.png'):
            img_path = os.path.join('data/bhmsds_processed_symbols', filename)
            img = Image.open(img_path).convert('L')
            img = img.resize((28, 28))
            pixels = np.array(img)
            bhmsds_train.append(pixels.flatten())
            bhmsds_train_labels.append(get_bhmsds_label(filename))
    
    bhmsds_train = np.array(bhmsds_train)
    bhmsds_train_labels = np.array(bhmsds_train_labels)
    
    return bhmsds_train, bhmsds_train_labels

def main():
    '''Build the dataset from EMNIST and the BHMSDS dataset, then export it as a .csv file.'''
    # Load EMNIST data
    emnist_train, emnist_train_labels, emnist_test, emnist_test_labels = load_emnist()
    
    # Process BHMSDS data if needed
    if not os.path.exists('data/bhmsds_processed_symbols') or \
       not os.listdir('data/bhmsds_processed_symbols'):
        process_bhmsds()
    
    # Load BHMSDS data
    bhmsds_train, bhmsds_train_labels = load_bhmsds()
    
    # Concatenate all image data and labels
    all_images = np.vstack((emnist_train, emnist_test, bhmsds_train))
    all_labels = np.concatenate((emnist_train_labels, emnist_test_labels, bhmsds_train_labels))
    
    # Create DataFrame with label and individual pixel columns
    df = pd.DataFrame(all_images, columns=[str(i) for i in range(784)])  # Create columns 0-783 for pixels
    df.insert(0, 'label', all_labels)  # Insert labels as the first column
    
    # Save to CSV
    df.to_csv('data/processed_dataset.csv', index=False)
    print("Dataset saved successfully!")
    print("\nFirst few rows:")
    print(df.head())
    print("\nLast few rows:")
    print(df.tail())
    print(f"\nTotal number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)-1}")  # -1 to exclude label column

if __name__ == '__main__':
    main()