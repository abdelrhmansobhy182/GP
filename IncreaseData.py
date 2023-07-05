import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter



def increaseData(data):
     Y = data['class']
     X = data.loc[:, data.columns != 'class']
     print(X)
     print(Y)
     # Apply SMOTE to oversample the minority class
     smote = SMOTE(random_state=42)
     X_resampled, y_resampled = smote.fit_resample(X,Y)

     # Concatenate the resampled features and target into a single dataframe
     df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled)], axis=1)
     df_resampled.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                             'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                             'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                             'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia',
                             'class']



     # Save the resampled dataset to a CSV file
     df_resampled.to_csv('resampled_dataset.csv', index=False)
     return df_resampled