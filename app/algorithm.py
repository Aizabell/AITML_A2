from sklearn.model_selection import KFold

class LinearRegression:
    def __init__(self, regularization, lr=0.001, method='batch', num_epochs=50, batch_size=50,
                 cv=KFold(n_splits=3), init_method="zero", use_momentum=False, momentum=0.9):
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method = method  # 'sto', 'mini', or 'batch'
        self.cv = cv
        self.regularization = regularization
        self.init_method = init_method  # "zero" or "xavier"
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.prev_step = 0  # for momentum
        self.kfold_scores = []  # to store cv results

    def initialize_weights(self, n_features):
        if self.init_method == "zero":
            return np.zeros(n_features)
        elif self.init_method == "xavier":
            lower, upper = -(1.0/np.sqrt(n_features)), (1.0/np.sqrt(n_features))
            return np.random.uniform(lower, upper, size=n_features)
        else:
            raise ValueError("Invalid initialization method.")

    def mse(self, y_true, y_pred):
        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)
        return np.mean((y_pred - y_true)**2)

    def r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def predict(self, X):
        return X @ self.theta

    def _train(self, X, y):
        # Compute prediction and gradient
        yhat = self.predict(X)
        m = X.shape[0]
        grad = (1/m) * (X.T @ (yhat - y))
        if self.regularization:
            grad += self.regularization.derivation(self.theta)
        # Update with learning rate and (optional) momentum
        step = self.lr * grad
        if self.use_momentum:
            step += self.momentum * self.prev_step
            self.prev_step = step
        self.theta = self.theta - step
        return self.mse(y, yhat)

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.kfold_scores = []  # reset cv scores

        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            X_cv_train, y_cv_train = X_train[train_idx], y_train[train_idx]
            X_cv_val, y_cv_val = X_train[val_idx], y_train[val_idx]

            # Reset previous momentum and early-stopping tracking per fold
            self.prev_step = 0
            self.val_loss_old = np.inf

            # Initialize weights for this fold
            self.theta = self.initialize_weights(X_cv_train.shape[1])

            # For logging per-fold, you could also start a nested mlflow run if desired.
            for epoch in range(self.num_epochs):
                # Shuffle (make sure to shuffle both X and y together)
                perm = np.random.permutation(X_cv_train.shape[0])
                X_cv_train = X_cv_train[perm]
                y_cv_train = y_cv_train[perm]

                # Choose gradient descent method
                if self.method == 'sto':
                    # Stochastic: update for each sample
                    for i in range(X_cv_train.shape[0]):
                        X_batch = X_cv_train[i].reshape(1, -1)
                        y_batch = np.array([y_cv_train[i]])
                        train_loss = self._train(X_batch, y_batch)
                elif self.method == 'mini':
                    # Mini-batch: update in batches of self.batch_size
                    for i in range(0, X_cv_train.shape[0], self.batch_size):
                        X_batch = X_cv_train[i:i+self.batch_size]
                        y_batch = y_cv_train[i:i+self.batch_size]
                        train_loss = self._train(X_batch, y_batch)
                else:
                    # Batch: update once per epoch using all training data
                    train_loss = self._train(X_cv_train, y_cv_train)

                # Validate on the fold's validation set
                y_val_pred = self.predict(X_cv_val)
                val_loss = self.mse(y_cv_val, y_val_pred)
                # Early stopping if validation loss is not changing significantly
                if np.allclose(val_loss, self.val_loss_old, atol=1e-6):
                    break
                self.val_loss_old = val_loss

            self.kfold_scores.append({
                "mse": val_loss,
                "r2": self.r2_score(y_cv_val, self.predict(X_cv_val))
            })

    def plot_feature_importance(self, feature_names):
        if len(self.theta) != len(feature_names):
            raise ValueError("Feature names must match the number of coefficients.")
        # Skip bias term if present (assumes first coefficient is bias)
        coef = self.theta.copy()
        if len(coef) == len(feature_names) + 1:
            coef = coef[1:]
        # Get absolute importance and sort indices
        abs_coef = np.abs(coef)
        sorted_idx = np.argsort(abs_coef)[::-1]
        plt.figure(figsize=(10, 5))
        plt.barh([feature_names[i] for i in sorted_idx], abs_coef[sorted_idx], color='blue')
        plt.xlabel("Absolute Coefficient Value")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()
        plt.show()