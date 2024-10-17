"""This example envisages traning a common Convolutional Neural Network on a number of distributed datasets using
classical FedOpt algorithm. Local datasets are uniformly distributed, but some of them are transformed (noised) 
according to predefined schema. During the training, three contribution metrics are defined: LOO, LSAA and EXLSAA."""

import pickle
import os
import datasets
from forcha.components.orchestrator.evaluator_orchestrator import Evaluator_Orchestrator
from forcha.components.settings.evaluator_settings import EvaluatorSettings
from model import MNIST_Expanded_CNN

def simulation():
    cwd = os.getcwd()
    settings = EvaluatorSettings(
        simulation_seed=121,
        global_epochs=80,
        local_epochs=3,
        number_of_nodes=10,
        sample_size=10,
        optimizer='SGD',
        batch_size=32,
        learning_rate=0.01,
        alpha_sample=True,
        loo_sample=True,
        line_search_length=5,
        parallelization = True,
        save_central_model=False,
        save_nodes_models=False
        )
    
    with open(f'/home/maciejzuziak/Uniform_Contribution_Paper-1/PhD_experiments/V/datasets/mnist/het/MNIST_10_dataset_pointers', 'rb') as file:
        data = pickle.load(file)
    # DATA: Selecting data for the orchestrator
    orchestrator_data = data[0]
    # DATA: Selecting data for nodes
    nodes_data = data[1]
    #nodes_data = data[1]
    model = MNIST_Expanded_CNN()
    orchestrator = Evaluator_Orchestrator(
        settings=settings, 
        full_debug=True,
        parallelization=False
        )
    
    orchestrator.prepare_orchestrator(
         model=model, 
         validation_data=orchestrator_data)
    orchestrator.prepare_training(nodes_data=nodes_data)
    signal = orchestrator.train_protocol()
    return signal

if __name__ == "__main__":
       simulation()