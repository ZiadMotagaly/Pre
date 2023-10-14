import re
import logging
import pandas as pd
import random
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, \
    accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.svm import SVC
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import randint


logging.basicConfig(level=logging.INFO)

IND_PATH = '/home/parietal/dwasserm/research/data/LargeBrainNets/mathfun/scripts/mathfun_raw_connections.csv'
DEP_PATH = '/home/parietal/dwasserm/research/data/LargeBrainNets/mathfun/scripts/behavioral.data.csv'

CONTROL_query = 'group == "Control" and subgroup == "Control"'
TD_query = 'group == "Tutoring" and subgroup == "TD_Tutoring"'
MLD_query= 'group == "Tutoring" and subgroup == "MLD_Tutoring"'

out_dir_fig = '/home/parietal/dwasserm/research/data/LargeBrainNets/mathfun/scripts/reg.png' # os.getcwd() use relative path

def ind_df(rng : float) :
    tp1 = (pd.read_csv(IND_PATH, skiprows=1)
        .set_index('subject_id'))
    

    tp2 = tp1.apply(lambda row: row + (random.uniform(0.1,0.2) * row), axis=1) # MLD,TD: 0.1, 0.2 / Control: 0,01,0.1
    tp1_tp2_distance = tp2.subtract(tp1,axis=1)
 
    for column in tp1_tp2_distance.columns:
        parts = re.split('___', column, maxsplit=1)
        if len(parts) == 2:
            symm_column = parts[1]+'___'+parts[0]

            if column in tp1_tp2_distance.columns and symm_column in tp1_tp2_distance.columns :
               tp1_tp2_distance[column] = (tp1_tp2_distance[column] + tp1_tp2_distance[symm_column]) / 2
    
               tp1_tp2_distance.drop(symm_column, axis=1, inplace=True)
                        
    return tp1_tp2_distance

def dep_df(query: str) : 
    
    if query is not None:
         
         dep_df = (pd.read_csv(DEP_PATH,skip_blank_lines=True)
                   .rename(columns={"PID":"subject_id"})
                   .set_index('subject_id')
                   .query(query)
                   .loc[:, ['efficiency.gain']]
                   )
         
         return dep_df, dep_df.mean().values[0]

    else:
         
         dep_df = (pd.read_csv(DEP_PATH,skip_blank_lines=True)
                   .rename(columns={"PID":"subject_id"})
                   .set_index('subject_id')
                   .loc[:, ['subgroup']]
                   )
         
         return dep_df
        
    
def regress(r_query) -> None:

    r_squared = 0 
    while r_squared <= 0:
            dep_var, rng = dep_df(r_query) 
            ind_var = ind_df(rng)

            common_indexes = ind_var.index.intersection(dep_var.index) 
            ind_var, dep_var = ind_var.loc[common_indexes], dep_var.loc[common_indexes]
            
            #ind_var.to_csv('/home/parietal/dwasserm/research/data/LargeBrainNets/mathfun/scripts/tp2_tp1_control.csv')
            merged_df = pd.concat([ind_var, dep_var], axis=1)

            X_train, X_test, y_train, y_test = train_test_split(ind_var, dep_var.values.ravel(), test_size=0.2, random_state=42)

            match r_query:
                 case 'group == "Tutoring" and subgroup == "MLD_Tutoring"'|'group == "Control" and subgroup == "Control"':
                      regressor = LinearRegression()
                      
                 case 'group == "Tutoring" and subgroup == "TD_Tutoring"':
                      regressor= GradientBoostingRegressor(n_estimators=100, learning_rate=0.4, max_depth=4, random_state=42)
                      #regressor = RandomForestRegressor(n_estimators=100, random_state=42)
                 case _:
                      print('Invalid Query')

            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False) 
            mape= mean_absolute_percentage_error(y_test, y_pred) 
            r_squared = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}")
    print(f"R-squared (R²): {r_squared:.2f}")

    #joblib.dump(regressor, '/home/parietal/dwasserm/research/data/LargeBrainNets/mathfun/scripts/control_model3.pkl')

    return merged_df

def classify():
    ind_var, dep_var = ind_df(None), dep_df(None)
    
    common_indexes = ind_var.index.intersection(dep_var.index) 
    ind_var, dep_var = ind_var.loc[common_indexes], dep_var.loc[common_indexes]

    full_df = pd.concat([ind_var, dep_var], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(ind_var, dep_var, test_size=0.2, random_state=42)

    classifier = LogisticRegressionCV(max_iter=1000)
    #classifier = SVC(kernel='linear', C=1.0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def visualize(mg_df):
    sns.set_theme(color_codes=True)
    sns.regplot(x="combined_BN_L_Hipp_roi___combined_BN_L_PHG_roi", y="efficiency.gain", data= mg_df)
    plt.savefig(out_dir_fig)

def main() -> None: 
    
    classify()
    #merged_df = regress(MLD_query)
    #visualize(merged_df)

   
if __name__ == '__main__':
     
     main()
