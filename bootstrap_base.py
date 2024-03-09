def bootstrap(X, y, n_samples=100):
    models = []
    metrics = {"Precision": [], "Recall": [], "F1": []}
    sample_indices= []
 
    for i in range(n_samples):
        # Bootstrap resampling
        indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        sample_indices.append(indices)
        X_sample = X.iloc[indices, :].values
        y_sample = y.loc[indices].values
 
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.33, random_state=42
        )
 
        # Define and compile neural network model
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dropout(rate=0.25))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
 
        # Train the model
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)
 
        # Evaluate the model
        y_pred = model.predict(X_test)
        thresh = (sum(y_train)/len(y_train))[0] #or np.mean(y_train)  Compute threshold
        metrics["Precision"].append(precision_score(y_test, (y_pred > thresh), average="macro"))
        metrics["Recall"].append(recall_score(y_test, (y_pred > thresh), average="macro"))
        metrics["F1"].append(f1_score(y_test, (y_pred > thresh), average="macro"))
 
        #print_report(y_test, y_pred, thresh)
 
        # Save the trained model
        models.append(model)
 
    # Create a DataFrame to store results
    pred_df = pd.DataFrame({
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "Models": models,
        "Sample_indices": sample_indices,
    })
 
    return models, pred_df