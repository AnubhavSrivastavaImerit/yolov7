import os
import mlflow
from mlflow.tracking import MlflowClient
import pathlib
import configparser

class manager:
    def __init__(self):
        self.config_path = os.path.join(os.path.expanduser("~"),".config/manager-config.ini")
        if not pathlib.Path(self.config_path).exists():
            raise FileNotFoundError
            
        # environment variable setup  
        self.config_detail = configparser.ConfigParser()
        self.config_detail.read(self.config_path)
        mlflow_config_reader = self.config_detail["mlflow-config"]
        aws_config_reader = self.config_detail["aws-config"]
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_config_reader["mlflow_tracking_uri"]
        os.environ["EXPERIMENT-NAME"] = mlflow_config_reader["experiment-name"]
        os.environ["AWS_ACCESS_KEY"] = aws_config_reader["aws_access_key"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_config_reader["aws_secret_access_key"]
        
        self.client =  MlflowClient()
        experiment = mlflow.get_experiment_by_name(os.environ["EXPERIMENT-NAME"] )
        # Need to check how to set experiemnt id
        self.run = self.client.create_run(experiment.experiment_id)
        print("Name: {}".format(experiment.name))
        print("Experiment ID: {}".format(experiment.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
        print("Running experiement with uuid  : {} ".format(self.run.info.run_id, ))

        mlflow.start_run()


    def log_param(self,key,value):
        self.client.log_param(self.run.run_id,key,value)
        return

    def log_artifact(self,path):
        self.client.log_artifact(self.run.run_id,path,"./")
        return

    def log_artifacts(self,dir:str):
        self.client.log_artifacts(self.run.run_id,dir,"./")
        return

    def __del__(self):
        self.log_artifact("./dataset.txt","./")
        mlflow.end_run()

    








