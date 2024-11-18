import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker import Session
import joblib

class ModelTrainer:
    def __init__(self, config):
        """
        Inizializza il trainer con configurazioni specifiche
        
        :param config: Dizionario di configurazione
        """
        self.config = config
        self.sagemaker_session = Session()
        self.role = config.get('sagemaker', {}).get('role')
        self.region = config.get('sagemaker', {}).get('region', 'eu-west-1')
        
        self.input_path = config.get('pipeline', {}).get('training', {}).get('input_path')
        self.output_path = config.get('pipeline', {}).get('training', {}).get('output_path')
        self.instance_type = config.get('sagemaker', {}).get('instance_type', 'ml.m5.xlarge')
    
    def create_training_script(self, output_path='/opt/ml/training_script.py'):
        """
        Genera uno script di training personalizzato
        """
        training_code = '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_model(input_path):
    # Carica dati preprocessati
    df = pd.read_csv(input_path)
    
    # Separazione features e target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Addestramento modello
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Valutazione
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Salvataggio modello
    model_path = os.path.join('/opt/ml/model', 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Modello salvato in {model_path}")

if __name__ == '__main__':
    import sys
    input_path = sys.argv[1]
    train_model(input_path)
'''
        
        with open(output_path, 'w') as f:
            f.write(training_code)
        
        return output_path

    def train(self):
        """
        Avvia il job di training su SageMaker
        """
        sklearn_estimator = SKLearn(
            entry_point=self.create_training_script(),
            role=self.role,
            instance_count=1,
            instance_type=self.instance_type,
            framework_version='0.23-1',
            output_path=self.output_path,
            code_location=self.output_path
        )

        sklearn_estimator.fit({
            'train': self.input_path
        })