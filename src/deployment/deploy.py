import sagemaker
from sagemaker.sklearn import SKLearnModel
from sagemaker import Session

class ModelDeployer:
    def __init__(self, config):
        """
        Inizializza il deployer con configurazioni specifiche
        
        :param config: Dizionario di configurazione
        """
        self.config = config
        self.sagemaker_session = Session()
        self.role = config.get('sagemaker', {}).get('role')
        self.region = config.get('sagemaker', {}).get('region', 'eu-west-1')
        
        self.model_path = config.get('pipeline', {}).get('training', {}).get('output_path')
        self.instance_type = config.get('sagemaker', {}).get('instance_type', 'ml.m5.xlarge')
    
    def deploy(self, endpoint_name='my-ml-endpoint'):
        """
        Distribuisce il modello come endpoint SageMaker
        
        :param endpoint_name: Nome dell'endpoint
        """
        model = SKLearnModel(
            model_data=f'{self.model_path}/model.joblib',
            role=self.role,
            framework_version='0.23-1',
            entry_point='inference.py'
        )
        
        predictor = model.deploy(
            instance_type=self.instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name
        )
        
        return predictor
    
    def create_inference_script(self, output_path='/opt/ml/inference.py'):
        """
        Genera lo script di inferenza per l'endpoint
        """
        inference_code = '''
import joblib
import numpy as np
import json
import os

def model_fn(model_dir):
    """
    Carica il modello
    """
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    return model

def input_fn(serialized_input_data, content_type):
    """
    Deserializza i dati di input
    """
    if content_type == 'application/json':
        input_data = json.loads(serialized_input_data)
        return np.array(input_data)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """
    Esegue le predizioni
    """
    return model.predict(input_data)

def output_fn(prediction, content_type):
    """
    Serializza l'output
    """
    if content_type == 'application/json':
        return json.dumps(prediction.tolist())
    raise ValueError(f"Unsupported content type: {content_type}")
'''
        
        with open(output_path, 'w') as f:
            f.write(inference_code)
        
        return output_path