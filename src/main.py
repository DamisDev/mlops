import yaml
from src.preprocessing.preprocess import DataPreprocessor
from src.training.train import ModelTrainer
from src.deployment.deploy import ModelDeployer

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Carica configurazione
    config = load_config()
    
    # Preprocessing
    preprocessor = DataPreprocessor(config)
    preprocessor.preprocess()
    
    # Training
    trainer = ModelTrainer(config)
    trainer.train()
    
    # Deployment
    deployer = ModelDeployer(config)
    predictor = deployer.deploy()

if __name__ == '__main__':
    main()