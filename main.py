import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,log_loss, confusion_matrix

dataset = pd.read_csv("data/cancer.csv")

dataset.head(5)
dataset.shape
dataset.describe()

plt.figure(figsize=(10,10))
for index, column in enumerate(dataset.columns):
    plt.subplot(3,3,index+1)
    plt.hist(dataset[column])
    plt.title(column)
    #plt.show()

# Spliting the dataset into train and test pandas dataframes
train_split = 1200
train_df = dataset.iloc[:train_split, :]
test_df = dataset.iloc[train_split:, :]

features_indices = len(dataset.columns) - 1
X_df = dataset.iloc[:, :features_indices]
y_df = dataset.iloc[:, -1]

# Nominalize the dataset
X_df_mean = X_df.mean() # mean
X_df_std = X_df.std() # standard deviation

X_df_norm = (X_df - X_df_mean) / X_df_std
X_df_norm.head(5)    



# Using sklearn to implement random forest algorithm
def train_and_eval_model(model_name: str, X_df: pd.DataFrame, y_df: pd.DataFrame, train_size: float):
    # Turn the dataframes to numpy arrays
    X = np.float64(X_df.values)
    y = np.float64(y_df.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=True, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    confusion_mtx = confusion_matrix(y_test, y_pred)

    return model, {'Model_name': model_name, 'Accuracy': accuracy, 'AUROC': auroc, 'F1Score': f1score, 'Confusion matrix': confusion_mtx}


# Training a model without data normalization
model_0, model_0_metrics = train_and_eval_model(model_name = 'model_0', X_df=X_df, y_df=y_df, train_size=0.8)

# Training a model with data normalization
model_1, model_1_metrics = train_and_eval_model(model_name = 'model_1', X_df=X_df_norm, y_df=y_df, train_size=0.8)



#Compare the results of both models
def compare_models(*model_metrics) -> pd.DataFrame:
    compare_models_df = pd.DataFrame(model_metrics, columns=model_metrics[0].keys())
    compare_models_df.set_index('Model_name', inplace=True)
    return compare_models_df

compare_models(model_0_metrics, model_1_metrics)



# Creating a model with a train_size of 0.9
model_2, model_2_metrics = train_and_eval_model(model_name='model_2', X_df=X_df_norm, y_df=y_df, train_size=0.9)
compare_models(model_0_metrics, model_1_metrics, model_2_metrics)



age = 20 
gender = 0 
bmi = 22 
smoking = 0 
genetic_risk = 1 
physical_activity = 20 
alcohol_intake = 5 
cancer_history = 0

raw_record = [age, gender, bmi, smoking, genetic_risk, physical_activity, alcohol_intake, cancer_history]
record = np.array(raw_record, type(float()))

record = (record - X_df_mean.values)/X_df_std.values
record = record.reshape(1,8)

prediction = model_2.predict(record)
if prediction == 0:
    print('You have been diagnosed to be cancer negative.')
else:
    print('Sorry, you have been diagnosed to be cancer positive.')

