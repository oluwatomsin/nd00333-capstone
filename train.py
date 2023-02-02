from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset

def prepare_data(data):
    print(data.dtypes)
    #all of them are numerical--> not encoding needed.
    #leave target as it is
    return (data.loc[:,data.columns!='DEATH_EVENT'], data['DEATH_EVENT'])

#getting run context
run = Run.get_context()

#getting training data
ws = run.experiment.workspace
dataset = Dataset.get_by_name(ws, name='heart_disease')
ds = dataset.to_pandas_dataframe()
print(ds)
#preparing data --> there is not too much preparation needed due to the nature of the data itself. Nevertheles, having a modular code for this processing task is recommendable,
x, y = prepare_data(ds) 

# TODO: Split data into train and test sets.
#spliting data --> 80-20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)





def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)
    #saving model
    joblib.dump(model, 'outputs/model.joblib')
    parser = argparse.ArgumentParser()

if __name__ == '__main__':
    main()