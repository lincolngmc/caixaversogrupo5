import pandas as pd
from sklearn.model_selection import train_test_split
# Carregamento do Dataset 
df = pd.read_csv('breast_data.csv', sep=',', index_col=None)

#Selecionar colunas necessarias (excluímos o ID e a última coluna por estar vazia)
df = df[['diagnosis','radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]

# Criação da coluna Target que é derivada da 'Diagnosis' onde 1 é para M (Malígno) e 0 para B (Benígno)
df['target'] =  df['diagnosis'].map({'M': 1, 'B': 0})

#Exclusao da Coluna Diagnosis
df = df.drop('diagnosis', axis=1)

#Checagem de valores nulos
df.isnull().sum()

# Definição de Target e Features
X = df['target']
y = df.drop('target', axis=1)



#df.to_csv('breast2.csv', sep=";", index=None)

