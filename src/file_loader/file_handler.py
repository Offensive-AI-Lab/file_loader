import json
import pickle
import os
import logging
from importlib import import_module
from .bucket_loader import BucketLoader
import dill
import importlib.util

class FileLoader:
    """
        A utility class for loading files and objects from local storage or a GCP bucket.

        Attributes:
        ----------
        from_bucket (bool): A flag indicating whether to load files from a GCP bucket (True) or locally (False).
        metadata (dict or str): Metadata related to the files and objects being loaded.
        bucket_loader (BucketLoader): An instance of the BucketLoader class for handling GCP bucket operations.

        Methods:
        --------
        to_pickle(file_name, obj):
            Save an object as a pickle file.

        from_pickle(file_name):
            Load an object from a pickle file.

        check_file_in_local(dir_path, expected_file):
            Check if a file exists in the local directory.

        get_model():
            Load the machine learning model from either local storage or a GCP bucket.

        get_dataloader():
            Load the data loader from either local storage or a GCP bucket.

        get_loss():
            Load the loss function, either a custom one or a built-in one, from either local storage or a GCP bucket.

        get_file(expected_file):
            Load a file with the given name from either local storage or a GCP bucket.

        get_estimator():
            Load an estimator object based on the specified ML model type, implementation, and algorithm.

        save_file(obj, path, as_pickle=False, as_json=False):
            Save an object to a file, either as a pickle or JSON file.

        """

    def __init__(self, metadata,
                 path_to_files_dir,
                 path_to_model_files_dir,
                 path_to_dataloader_files_dir,
                 path_to_dataset_files_dir,
                 path_to_loss_files_dir,
                 path_to_req_files_dir,
                 from_bucket,
                 bucket_name=None,
                 account_service_key_name=None):
        """
                Initialize a FileLoader instance.

                Parameters:
                -----------
                metadata (dict or str): Metadata related to the files and objects being loaded.

                """
        self.path_to_files_dir = path_to_files_dir
        self.path_to_model_files_dir = path_to_model_files_dir
        self.path_to_dataloader_files_dir = path_to_dataloader_files_dir
        self.path_to_dataset_files_dir = path_to_dataset_files_dir
        self.path_to_loss_files_dir = path_to_loss_files_dir
        self.path_to_req_files_dir = path_to_req_files_dir
        self.from_bucket = bool(from_bucket)
        self.bucket_name = bucket_name
        self.account_service_key_path = self.path_to_files_dir + "/" + account_service_key_name
        if self.from_bucket:
            self.bucket_loader = BucketLoader(metadata,
                                              self.path_to_files_dir,
                                              self.path_to_model_files_dir,
                                              self.path_to_dataloader_files_dir,
                                              self.path_to_dataset_files_dir,
                                              self.path_to_loss_files_dir,
                                              self.path_to_req_files_dir,
                                              self.bucket_name,
                                              self.account_service_key_path)
        # self.from_bucket = False
        self.metadata = metadata
        if isinstance(self.metadata, str):
            self.metadata = json.loads(self.metadata)
        if isinstance(self.metadata, dict):
            self.__ML_model_file_id = self.metadata['ml_model']['meta']['definition']['uid']
            self.__loss_function_file_id = self.metadata['ml_model']['loss']['uid']
            self.__dataloader_file_id = self.metadata['dataloader']['definition']['uid']
            # self.input_validator = InputValidator(metadata)

    @staticmethod
    def to_pickle(file_name, obj):
        """
                Save an object as a pickle file.

                Parameters:
                -----------
                file_name (str): The name of the pickle file.
                obj: The object to be saved.

                Returns:
                --------
                None

                """
        with open(f'./{file_name}.pickle', 'wb') as f:
            dill.dump(obj, f)

    @staticmethod
    def from_pickle(file_name):
        with open(file_name, 'rb') as f:
            loaded_obj = dill.load(f)
            # class_name = loaded_obj.__class__.__module__

        return loaded_obj

    def load_module_from_file(self,module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    def check_file_in_local(self, dir_path, expected_file):
        """
                        Check if a file exists in the local directory.

                        Parameters:
                        -----------
                        dir_path (str): The directory path to check.
                        expected_file (str): The name of the file to check for.

                        Returns:
                        --------
                        bool: True if the file exists, False otherwise.

                        """

        def check_single(dir_path, expected_file):
            print(f"Checking for {expected_file} in {dir_path}")
            scaner = os.scandir(path=dir_path)
            for entry in scaner:
                if entry.is_dir() or entry.is_file():
                    if entry.name == expected_file:
                        return True
            return False

        if isinstance(expected_file, list):
            return all([check_single(dir_path, file) for file in expected_file])
        else:
            return (check_single(dir_path, expected_file))

    def get_model(self):
        """
                Load the machine learning model from either local storage or a GCP bucket.

                Returns:
                --------
                model: The loaded machine learning model.

                """

        def get_model_from_local(framework):
            if framework == 'sklearn':
                import sklearn
                if self.from_bucket == True:
                    bin_model_path = self.path_to_model_files_dir + "/model.pickle"
                else:
                    bin_model_path = self.metadata['ml_model']['meta']['parameters']['path']
                # class_name = self.metadata['ml_model']['meta']['definition']['class_name']
                # module = __import__('user_files.model.model_def', fromlist=[class_name])
                # model_class = getattr(module, class_name)
                # model_object = getattr(import_module(".".join([os.getenv("FILES_PATH"), "model.model_def"])), class_name)()
                model = FileLoader.from_pickle(bin_model_path)
            elif framework == 'tensorflow' or framework == 'keras':
                if self.from_bucket == True:
                    PATH = self.path_to_model_files_dir + "/model.keras"

                else:
                    PATH = self.metadata['ml_model']['meta']['parameters']['path']
                # model = tf.keras.models.load_model(PATH, custom_objects={"MNISTSequentialModel": MNISTSequentialModel})
                class_name = self.metadata['ml_model']['meta']['definition']['class_name']
                if self.from_bucket == True:
                    model_path = os.path.join(self.path_to_model_files_dir, 'model_def.py')

                    # Load the dataloader module
                    model_module = self.load_module_from_file('model_def', model_path)
                    model_object = getattr(model_module, class_name)()
                else:
                    model_object = getattr(import_module(self.metadata['ml_model']['meta']['definition']['path']),  class_name)()


                model = model_object.load(PATH)
            elif framework == 'xgboost':
                import joblib
                import xgboost as xgb
                if self.from_bucket == True:
                    model = xgb.XGBClassifier()
                    PATH = self.path_to_model_files_dir + "/model.json"
                else:
                    model = xgb.XGBClassifier()
                    PATH = self.metadata['ml_model']['meta']['parameters']['path']
                # model_xgb = xgb.XGBClassifier()
                model.load_model(PATH)
            elif framework == 'catboost':
                from catboost import CatBoostClassifier
                if self.from_bucket == True:
                    model = CatBoostClassifier()
                    PATH = self.path_to_model_files_dir + "/model.cbm"
                else:
                    model = CatBoostClassifier()
                    PATH = self.metadata['ml_model']['meta']['parameters']['path']
                model.load_model(PATH)
            elif framework == 'pytorch':
                import torch
                class_name = self.metadata['ml_model']['meta']['definition']['class_name']
                if self.from_bucket == True:
                    model_path = os.path.join(self.path_to_model_files_dir, 'model_def.py')

                    # Load the dataloader module
                    model_module = self.load_module_from_file('model_def', model_path)
                    model = getattr(model_module, class_name)()
                    # model = getattr(import_module(".".join([os.getenv("FILES_PATH"), "model.model_def"])), class_name)()
                    PATH = self.path_to_model_files_dir + "/parameters.pth"
                else:
                    model = getattr(import_module(self.metadata['ml_model']['meta']['definition']['path']), class_name)()

                    PATH = self.metadata['ml_model']['meta']['parameters']['path']
                model.load_state_dict(torch.load(PATH))
            else:
                raise Exception(f"Expected framework to be torch, tensorflow, keras or sklearn. got {framework}")
            return model

        # check if the ML model's file is in the folder
        framework = self.metadata['ml_model']['meta']['framework']
        if framework == 'sklearn':
            expected_file = "model.pickle"
        elif framework == 'tensorflow' or framework == 'keras':
            expected_file = "model.keras"
        elif framework == "pytorch":
            expected_file = [f"parameters.pth", "model_def.py"]
        elif framework == "xgboost":
            expected_file = "model.json"
        elif framework == "catboost":
            expected_file = "model.cbm"

        else:
            raise Exception(f"Expected framework to be torch, tensorflow, keras or sklearn. got {framework}")

        dir_path = self.path_to_model_files_dir
        print(f"Checking for {expected_file} in {dir_path}")
        file_in_local = self.check_file_in_local(dir_path=dir_path, expected_file=expected_file)

        if file_in_local:
            model = get_model_from_local(framework)
            return model

        elif self.from_bucket:
            try:
                self.bucket_loader.get_model()

            except Exception as err:
                raise Exception(f"Failed to get ML model from bucket:\nError: {err}").with_traceback(err.__traceback__)
            model = get_model_from_local(framework)
            return model

        else:
            raise FileNotFoundError("ML model file not found")

    def get_dataloader(self):
        def get_dataset_path():
            path_to_dataset = None
            files_in_dataset_folder = os.listdir(self.path_to_dataset_files_dir)
            print(f"Files in dataset folder: {files_in_dataset_folder}")
            for file in files_in_dataset_folder:
                if file.startswith("test_set") and not file.endswith("zip"):
                    path_to_dataset = file
                    print(f"Found dataset file: {path_to_dataset}")
            if path_to_dataset is None:
                return None
            print(f"Path to dataset: {self.path_to_dataset_files_dir}/{path_to_dataset}")
            return self.path_to_dataset_files_dir + "/" + path_to_dataset

        def get_dataloader_from_local():
            dataloader_path = os.path.join(self.path_to_dataloader_files_dir, 'dataloader_def.py')

            # Load the dataloader module
            dataloader_module = self.load_module_from_file('dataloader_def', dataloader_path)

            class_name = self.metadata['dataloader']['definition']['class_name']  # see if it from pytorch, tensorflow, sklearn CSV  or custom
            if self.from_bucket == True:
                path_to_dataset = get_dataset_path()
            else:
                path_to_dataset = self.metadata['test_set']['path']
            if self.metadata['ml_model']['meta']['framework'] == "pytorch":
                from torch.utils.data import DataLoader



                if self.from_bucket == True:
                    print(path_to_dataset)
                    # dataset = getattr(import_module(".".join([os.getenv("FILES_PATH"), "dataloader.dataloader_def"])),class_name)(path_to_dataset, batch_size=2)
                    # dataset = PyTorchImgLoader(path_to_dataset, batch_size, transform=transform)
                    dataset_class = getattr(dataloader_module, class_name)
                    dataset = dataset_class(path_to_dataset)
                else:
                    dataset = getattr(import_module(self.metadata['dataloader']['definition']['path']), class_name)(path_to_dataset)
                dataloader = DataLoader(dataset, shuffle=False)
            else:
                try:
                    if self.metadata['ml_model']['meta']['framework'] == "tensorflow" or self.metadata['ml_model']['meta']['framework'] == "keras":
                        input = self.metadata['ml_model']['dim']['input']
                        dataloader_class = getattr(dataloader_module, class_name)
                        dataloader = dataloader_class(path_to_dataset, input=input)

                        # dataloader = getattr(import_module(".".join([os.getenv("FILES_PATH"), "dataloader.dataloader_def"])),class_name)(path_to_dataset, batch_size=2, input=input)
                    else:
                        dataloader_class = getattr(dataloader_module, class_name)
                        dataloader = dataloader_class(path_to_dataset)
                        # dataloader = getattr(import_module(".".join([os.getenv("FILES_PATH"), "dataloader.dataloader_def"])),class_name)(path_to_dataset, batch_size=2)



                except KeyError:
                    raise ValueError(f"Class '{class_name}' not found or imported.")

            return dataloader

        if get_dataset_path() is None:
            if self.from_bucket:
                self.bucket_loader.get_dataset()
            else:
                raise Exception("Could not found dataset in local")
            if get_dataset_path() is None:
                raise Exception("Could not found dataset in cloud storge")
        # check if the ML model's file is in the folder
        dataloader_expected_file = "dataloader_def.py"
        file_in_local = self.check_file_in_local(dir_path=self.path_to_dataloader_files_dir,expected_file=dataloader_expected_file)


        if file_in_local:
            dataloader = get_dataloader_from_local()
            return dataloader

        elif self.from_bucket:
            try:
                self.bucket_loader.get_dataloader()
            except Exception as err:
                raise Exception(f"Failed to get dataloader from bucket:\nError: {err}").with_traceback(
                    err.__traceback__)

            dataloader = get_dataloader_from_local()
            return dataloader

        else:
            raise FileNotFoundError("dataloader file not found")

    def get_loss(self):  # Ask about this
        """
                Load the loss function, either a custom one or a built-in one, from either local storage or a GCP bucket.

                Returns:
                --------
                loss_func: The loaded loss function.

                """

        def get_loss_from_local():
            class_name = self.metadata['ml_model']['loss']['type']
            if self.from_bucket == True:
                loss_path = os.path.join(self.path_to_loss_files_dir, 'loss.py')

                # Load the dataloader module
                loss_module = self.load_module_from_file('loss', loss_path)
                loss_func = getattr(loss_module, class_name)()
                # loss_func = getattr(import_module(".".join([os.getenv("FILES_PATH"), "loss.py"])), class_name)()
            else:
                loss_func = getattr(import_module(self.metadata['ml_model']['loss']['path']), class_name)()
            return loss_func

        if self.metadata['ml_model']['loss']['type'] == 'CustomLoss':

            expected_file = "loss.py"
            file_in_local = self.check_file_in_local(dir_path=self.path_to_loss_files_dir, expected_file=expected_file)

            if file_in_local:

                loss = get_loss_from_local()
                return loss

            elif self.from_bucket:
                try:
                    self.bucket_loader.get_loss()
                except Exception as err:
                    raise Exception(f"Failed to get loss from bucket:\nError: {err}").with_traceback(err.__traceback__)

                loss = get_loss_from_local()
                return loss

            else:
                raise FileNotFoundError("loss file not found")
        else:
            class_name = self.metadata['ml_model']['loss']['type']
            framework = self.metadata['ml_model']['meta']['framework']
            if framework == "pytorch":
                loss = getattr(import_module("torch.nn.modules.loss"), class_name)()
            elif framework == "tensorflow" or framework == "keras":
                if class_name is None or class_name == "":
                    return None
                loss = getattr(import_module("tensorflow.keras.losses"), class_name)()
            elif framework is None:
                return None
            else:
                raise TypeError("Framework must be pytorch or tensorflow for loss function")
            return loss

    # def get_file(self, expected_file):
    #     """
    #             Load a file with the given name from either local storage or a GCP bucket.
    #
    #             Parameters:
    #             -----------
    #             expected_file (str): The name of the file to load.
    #
    #             Returns:
    #             --------
    #             file: The loaded file or object.
    #
    #             """
    #     def get_file_from_local(expected_file):
    #         file_path = get_files_package_root() + expected_file
    #         if isinstance(expected_file, str):
    #             file_name, file_format = expected_file.split(".")
    #             if file_format == "pickle" or file_format == "pkl":
    #                 file = FileLoader.from_pickle(file_path)
    #                 return file
    #             elif file_format == 'json':
    #                 with open(file_path, 'r') as f:
    #                     return json.load(f)
    #             elif file_format == 'txt':
    #                 with open(file_path, 'r') as f:
    #                     return f.read()
    #             else:
    #                 raise Exception(f"Expected file format pickle,pkl,json or txt. got {file_format}")
    #
    #     # check if the ML model's file is in the folder
    #     file_in_local = self.check_file_in_local(dir_path=get_files_package_root(),expected_file=expected_file)
    #
    #     if file_in_local:
    #         try:
    #             file = get_file_from_local(expected_file)
    #             return file
    #         except Exception as err:
    #             error_logger.error(f"Error occurred while getting {expected_file} from local:\nError: {err}")
    #             return
    #     elif self.from_bucket:
    #         try:
    #             self.bucket_loader.get_file(expected_file)
    #             file = get_file_from_local(expected_file)
    #             return file
    #         except Exception as err:
    #             raise Exception(f"Failed to get dataloader from bucket:\nError: {err}")
    #     else:
    #         raise FileNotFoundError(f""
    #                                 f"{expected_file} file not found")

    def get_estimator(self):
        """
               Load an estimator object based on the specified ML model type, implementation, and algorithm.

               Returns:
               --------
               estimator: The loaded estimator object.

               """

        def get_obj_from_str(obj_as_str):
            # Checking torch estimators
            if obj_as_str == "PyTorchClassifier":
                from art.estimators.classification import PyTorchClassifier
                return PyTorchClassifier
            # elif obj_as_str == "PyTorchRegressor":
            #     from art.estimators.regression.pytorch import PyTorchRegressor
            #     return PyTorchRegressor
            # Checking tensorflow estimators
            elif obj_as_str == "TensorFlowV2Classifier":
                from art.estimators.classification import TensorFlowV2Classifier
                return TensorFlowV2Classifier
            # elif obj_as_str == "KerasRegressor":
            #     from art.estimators.regression import KerasRegressor
            #     return KerasRegressor
            elif obj_as_str == "KerasClassifier":
                from art.estimators.classification import KerasClassifier
                return KerasClassifier
            # Checking sklearn estimators
            # elif obj_as_str == "ScikitlearnRegressor":
            #     from art.estimators.regression import ScikitlearnRegressor
            #     return ScikitlearnRegressor
            elif obj_as_str == "SklearnClassifier":
                from art.estimators.classification import SklearnClassifier
                return SklearnClassifier
                # Checking xgboost estimators
            elif obj_as_str == "XGBoostClassifier":
                from art.estimators.classification import XGBoostClassifier
                return XGBoostClassifier
            # Checking catboost estimators
            elif obj_as_str == "CatBoostARTClassifier":
                from art.estimators.classification import CatBoostARTClassifier
                return CatBoostARTClassifier

            else:
                raise Exception(f"{obj_as_str}  is not an estimator name")

        def assign_vars(cls, args_dict, ML_model):
            framework = self.metadata['ml_model']['meta']['framework']
            if args_dict.get("optimizer"):
                optimizer_name = self.metadata['ml_model']['optimizer']['type']
                lr = self.metadata['ml_model']['optimizer']['learning_rate']
                if framework == "pytorch":
                    optimizer = getattr(import_module(f"torch.optim.{optimizer_name.lower()}"), optimizer_name)(ML_model.parameters(), lr=lr)
                elif framework == "tensorflow" or framework == "keras":
                    optimizer = getattr(import_module(f"tensorflow.keras.optimizers"), optimizer_name)(learning_rate=lr)
                else:
                    optimizer = None
                # optimizer = SGD(ML_model.parameters(), lr=lr)
                args_dict["optimizer"] = optimizer

            if args_dict.get("loss"):
                try:
                    loss = self.get_loss()
                    if self.metadata['ml_model']['meta']['framework'] == "tensorflow" or self.metadata['ml_model']['meta']['framework'] == "keras":

                        args_dict["loss_object"] = loss
                        args_dict.pop("loss")
                    else:
                        args_dict["loss"] = loss
                except Exception as err:

                    raise Exception(
                        f"Error while getting loss fucntion to estimator.").with_traceback(err.__traceback__)
            if framework == "sklearn":
                model_name = type(ML_model).__bases__[0]
                ML_model.__class__.__module__ = model_name.__module__
                ML_model.__class__.__name__ = model_name.__name__
            if self.metadata['ml_model']['meta']['framework'] == "keras":
                import tensorflow as tf

                # Disable eager execution
                tf.compat.v1.disable_eager_execution()
                ML_model.compile(loss=args_dict["loss"], optimizer=args_dict["optimizer"], metrics=['accuracy'])
            obj = cls(**args_dict, model=ML_model)
            return obj

        print("Getting estimator...")
        with open(self.path_to_files_dir + "/Estimator_params.json", 'r') as f:
            content = json.load(f)
            if isinstance(content, str):
                estimator_params = json.loads(content)
            elif isinstance(content, dict):
                estimator_params = content
        model = self.get_model()
        # model = NeuralNetworkClassificationModel(29,2)
        estimator_str = estimator_params['object']
        estimator_obj = get_obj_from_str(estimator_str)
        print(f"Estimator: {estimator_obj}")
        params = estimator_params['params']
        print(f"Estimator params: {params}")
        estimator = assign_vars(cls=estimator_obj, args_dict=params, ML_model=model)
        return estimator

    # def get_req_file(self):
    #     if self.metadata["req_file"]:
    #         req_file_of_user = self.metadata["req_file"]
    #     framework = self.metadata['ml_model']['meta']['framework']
    #     if framework == "pytorch":
    def merge_requirements(self, base_file, user_file, output_file):
        base_requirements = {}
        user_requirements = {}

        # Read base requirements
        with open(base_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '==' in line:
                        package, version = line.split('==')
                        base_requirements[package.lower()] = version
                    else:
                        package = line.split('==')[0].strip()
                        base_requirements[package.lower()] = ''

        # Read user requirements
        with open(user_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '==' in line:
                        package, version = line.split('==')
                        user_requirements[package.lower()] = version
                    else:
                        package = line.split('==')[0].strip()
                        user_requirements[package.lower()] = ''

        # Merge requirements
        merged_requirements = base_requirements.copy()
        merged_requirements.update(user_requirements)

        # Write merged requirements to output file
        with open(output_file, 'w') as f:
            for package, version in merged_requirements.items():
                if version:
                    f.write(f"{package}=={version}\n")
                else:
                    f.write(f"{package}\n")

    def get_req_file(self):
        def get_req_from_local():
            user_file = "user_requirements.txt"
            file_in_local = self.check_file_in_local(dir_path=self.path_to_req_files_dir, expected_file=user_file)
            if file_in_local:
                base_file = self.path_to_req_files_dir + "/requirements.txt"
                user_file = self.path_to_req_files_dir + "/user_requirements.txt"
                output_file = self.path_to_req_files_dir + "/requirements.txt"
                self.merge_requirements(base_file, user_file, output_file)
                with open(output_file, 'r') as f:
                    req_file = f.read()
                    req_file_path = self.path_to_req_files_dir + "/requirements.txt"
                return req_file_path
            else:
                with open(self.path_to_req_files_dir + "/requirements.txt", 'r') as f:
                    req_file = f.read()
                    req_file_path = self.path_to_req_files_dir + "/requirements.txt"
                return req_file_path

        # if self.metadata["req_file"]:
        #     req_file_of_user = self.metadata["req_file"]
        # framework = self.metadata['ml_model']['meta']['framework']
        expected_file = "requirements.txt"
        # if framework == "pytorch":
        file_in_local = self.check_file_in_local(dir_path=self.path_to_req_files_dir,expected_file=expected_file)

        if file_in_local:

            req_file = get_req_from_local()
            return req_file

        elif self.from_bucket:
            try:
                self.bucket_loader.get_req_file()
            except Exception as err:
                raise Exception(f"Failed to get loss from bucket:\nError: {err}").with_traceback(err.__traceback__)

            req = get_req_from_local()
            return req

        else:
            raise FileNotFoundError("loss file not found")

    def save_file(self, obj, path, as_pickle=False, as_json=False):
        """
        saves the file
        :param obj : the object to save
        :param path: path to save the file in
        :param as_pickle: if the file is binary or not
        :param as_json : if the file is a json
        :return: None
        """
        if as_pickle:
            with open(path, 'wb') as f:
                dill.dump(obj, f)
        elif as_json:
            with open(path, 'w') as f:
                json.dump(obj, f)

        else:
            raise Exception("Expected as_pickle or as_json to be True")