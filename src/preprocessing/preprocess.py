import pandas as pd
import numpy as np
import boto3
import sagemaker
from sagemaker.preprocessing import SKLearnProcessor
from sagemaker import Session

class DataPreprocessor:
    def __init__(self, config):
        """
        Inizializza il preprocessore con configurazioni specifiche
        
        :param config: Dizionario di configurazione
        """
        self.config = config
        self.sagemaker_session = Session()
        self.role = config.get('sagemaker', {}).get('role')
        self.region = config.get('sagemaker', {}).get('region', 'eu-west-1')
        
        self.input_path = config.get('pipeline', {}).get('preprocessing', {}).get('input_path')
        self.output_path = config.get('pipeline', {}).get('preprocessing', {}).get('output_path')
    
    def preprocess(self):
        """
        Esegue il preprocessing dei dati utilizzando SKLearnProcessor
        """
        sklearn_processor = SKLearnProcessor(
            framework_version='0.23-1',
            role=self.role,
            instance_type='ml.m5.xlarge',
            instance_count=1,
            sagemaker_session=self.sagemaker_session
        )
        
        sklearn_processor.run(
            code='preprocessing_script.py',  # Script Python con la logica di preprocessing
            inputs=[
                sagemaker.processing.ProcessingInput(
                    source=self.input_path, 
                    destination='/opt/ml/processing/input'
                )
            ],
            outputs=[
                sagemaker.processing.ProcessingOutput(
                    source='/opt/ml/processing/output',
                    destination=self.output_path
                )
            ]
        )

def create_preprocessing_script(output_path='/opt/ml/processing/preprocessing_script.py'):
    """
    Genera uno script di preprocessing da utilizzare nel job
    """
    preprocessing_code = '''
import pandas as pd
import numpy as np

def preprocess_data(input_path, output_path):
    # Carica il dataset
    df = pd.read_csv(input_path)
    
    # Esempio di preprocessing
    # Gestione valori nulli
    df.dropna(inplace=True)
    
    # Encoding categoriche
    df = pd.get_dummies(df)
    
    # Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    columns_to_scale = df.select_dtypes(include=['float64', 'int64']).columns
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    # Salva il dataset preprocessato
    df.to_csv(f'{output_path}/preprocessed_data.csv', index=False)

if __name__ == '__main__':
    import sys
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    preprocess_data(input_path, output_path)
'''
    
    with open(output_path, 'w') as f:
        f.write(preprocessing_code)
    
    return output_path