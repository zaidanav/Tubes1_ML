# Tubes1_ML

## Deskripsi Singkat
Repository ini berisi code implementasi Model FFNN from scratch beserta hasil Experimentnya

## Set up and run
### Cara run Notebook 
1. klik Experiment.ipynb
2. Run cell dari bagian import sampai Visualization
3. pilih bagian experiment yang ingin di run

### Membuat model
Note : jangan lupa data import dan preprocessing

#### buat parameter

base_architecture = {
    'layers': [64, 32, 10],
    'activations': ['relu', 'relu', 'softmax'],
    'loss': 'categorical_cross_entropy'
}


params = {
    'epochs': 20,
    'batch_size': 32,
    'learning_rate': 0.1
}


#### Definisikan model

model = FFNN_model.FeedforwardNeuralNetwork(
    input_size=X_train.shape[1],
    layer_sizes=base_architecture['layers'],
    activations=base_architecture['activations'],
    loss=base_architecture['loss'],
    weight_init= 'xavier',
    weight_init_params={'seed': 42})

#### Tentukan hyperparameter training

history = model.train(
        X_train=X_train,
        y_train=y_train_onehot,
        X_val=X_test,
        y_val=y_test_onehot,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
        verbose=1)

Note : plot_history dan evaluate_model didefinisikan pada notebook bagian Visualization

#### plot hasil training

plot_history(history)
    
print(f"\nEvaluating model")

accuracy, _ = evaluate_model(model, X_test, y_test_onehot, y_test)



## Pembagian Tugas

1. Farhan Raditya Aji / 13522142 : Experiment 5 , Bonus 1 dan 2, laporan
2. Muhammad Zaidan Sa'dun Robbani / 13522146 : Experiment 3 dan 4, laporan
3. Rafif Ardhinto Ichwantoro / 13522159 : Membuat implementasi Class FFNN, Experiment 1 & 2, laporan 