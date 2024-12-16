import json
import os
import traceback
import logging
from gcloud import storage

class BucketLoader:
    """
    Attributes:
    __________
    bucket: str, The name of the bucket holding the files
    access_key_id: str, access key to the GCP account
    secret_access_key: str, secret access key to the GCP account
    region: Bucket region
    ML_model_file_id: str, name of the file containing the ML model without the file type(e.g pickle)
    loss_function_file_id: str, name of the file containing the loss function without the file type(e.g pickle)
    optimizer_file_id: str, name of the file containing the optimizer without the file type(e.g pickle)
    dataloader_file_id: str, name of the file containing the dataloader without the file type(e.g pickle)
    requirements_file_id: str, name of the file containing the requirements with the file type(e.g txt)

    Methods:
    --------
    upload_to_gcp(): static method, uploading files to GCP bucket.
    download_from_GCP(): static method, downloading files from GCP bucket.
    to_pickle(): static method, converting objects to pickle files.
    from_pickle(): static method, converting pickle files to objects.
    upload(): wrapper to upload_to_gcp method, that handles different.
    get_model(): returning the ML model from GCP.
    get_loss(): returning the loss function as an object from GCP.
    get_optimizer(): returning the optimizer as an object from GCP.
    get_dataloader(): returning the dataloader as an object from GCP.
    get_requirements(): returning the requirements file as txt from GCP.
    """
    def __init__(self,metadata,
                 path_to_files_dir,
                 path_to_model_files_dir,
                 path_to_dataloader_files_dir,
                 path_to_dataset_files_dir,
                 path_to_loss_files_dir,
                 path_to_req_files_dir,
                 bucket_name,
                 account_service_key_path):
        """
        The bucket loader is an object that connect with aws and preform downloads
        and uploads from GCP bucket.

        Parameters
        ----------
        metadata : str/dict
            The json from user containing all metadata and structure as above.

            Expected json file:

           { 'ml_model':{
                      'meta':{
                            'definition':{
                                        'uid': str(id), 'path': str, 'class_name': str('name')
                                        },
                            'parameters': {
                                        'uid': str(id), 'path': str
                                        },
                            'framework': str(type),
                            'ml_type': str(type)
                            }
                            },


                      'dim':{
                            'input': tuple('rows,cols'),'num_classes': int('num'), 'clip_values': tuple('min,max')
                            },

                      'loss':{
                            'uid': str(id), 'type': str(type),'path': str
                            },
                      'optimizer':{
                            type: str('type'), 'learning_rate': int
                            },

              'dataloader':{
                        'definition':{'uid': str(id), 'path': str, 'class_name': str
                                    }
                            },
              'auth': {
                      'bucket_name': str('name'),
                      'account_service_key': str('key')
                      }
            }
          Example:
           { 'ML_model':{
                      'meta':{
                            'definition':{
                                        'uid': "modDefID", 'path': gc//bucketName//, 'class_name': myClass
                                        },
                            'parameters': {
                                        'uid': parametersID, 'path': gc//bucketName//
                                        },
                            'framework': tensorflow,
                            'ML_type': classification
                            }
                            },


                      'dim':{
                            'input': (10,10),'num_classes': 10, 'clip_values': (100,200)
                            },

                      'loss':{
                            'uid': lossID, 'type': crossEntropy ,'path': gc//bucketName//
                            },
                      'optimizer':{
                            type: ADAM, 'learning_rate': 0.01
                            },

              'dataloader':{
                        'definition':{'uid': dataloaderID, 'path': gc//bucketName//, 'class_name': dataClass
                                    }
                            },
              'auth': {
                      'bucket_name': MyBucket,
                      'account_service_key': ACCOUNT_SERVICE_KEY
                      }
            }


        Methods
        ----------
        All the methods are getter, setters and one method of that open connection.
        Getters downloading files from S3 bucket, while setters uploading files to GCP.

        """
        self.bucket_name = bucket_name
        self.account_service_key_path = account_service_key_path
        self.path_to_files_dir = path_to_files_dir
        self.path_to_model_files_dir = path_to_model_files_dir
        self.path_to_dataloader_files_dir = path_to_dataloader_files_dir
        self.path_to_dataset_files_dir = path_to_dataset_files_dir
        self.path_to_loss_files_dir = path_to_loss_files_dir
        self.path_to_req_files_dir = path_to_req_files_dir
        self.metadata = metadata
        if isinstance(self.metadata, str):
            self.metadata = json.loads(self.metadata)
        if isinstance(self.metadata, dict):
            # Extracting the parts regrading the model it's self
            try:
                self.__ML_model_file_id = self.metadata['ml_model']['meta']['parameters']['uid']
                self.__ML_model_file_URL = self.metadata['ml_model']['meta']['parameters']['path']
                # Extracting the parts regrading the model definition file
                self.__ML_model_class_name = self.metadata['ml_model']['meta']['definition']['class_name']
                self.__ML_model_script_file_id = self.metadata['ml_model']['meta']['definition']['uid']
                self.__ML_model_script_file_URL = self.metadata['ml_model']['meta']['definition']['path']
                # Extracting the parts regrading the loss function
                self.__loss_function_file_id = self.metadata['ml_model']['loss']['uid']
                self.__loss_function_file_URL = self.metadata['ml_model']['loss']['path']
                # Extracting the parts regrading the optimizer
                self.__optimizer_type = self.metadata['ml_model']['optimizer']['type']
                # Extracting the parts regrading the dataloader
                self.__dataloader_file_id = self.metadata['dataloader']['definition']['uid']
                self.__dataloader_file_URL = self.metadata['dataloader']['definition']['path']
                # Extracting the parts regrading the test set
                self.__test_set_id = self.metadata['test_set']['uid']
                self.__test_set_URL = self.metadata['test_set']['path']
                #check if exists
                if 'req_file' in self.metadata and isinstance(self.metadata['req_file'], dict):
                    self.__requirements_file_id = self.metadata['req_file']['uid']
                    self.__requirements_file_URL = self.metadata['req_file']['path']
                # Extracting the parts regrading the GCP authentication
                # self.__account_service_key = self.metadata['gcp_auth']['account_service_key']
                ACCOUNT_SERVICE_KEY = self.account_service_key_path
                os.environ["ACCOUNT_SERVICE_KEY"] = ACCOUNT_SERVICE_KEY
            except Exception as err:
                logging.error(f"Metadata is not valid!\nError:\n{err}")
                logging.error(traceback.format_exc())

        else:
            raise TypeError('meta data need to be type dict or str')

    def __getstate__(self):
        state = {"__ML_model_file_id": self.__ML_model_file_id,
                 "__ML_model_script_file_id": self.__ML_model_script_file_id,
                 "__ML_model_class_name": self.__ML_model_class_name,
                 "__loss_function_file_id" : self.__loss_function_file_id,
                "__optimizer_type": self.__optimizer_type ,
                 "__dataloader_file_id" : self.__dataloader_file_id,
                 "__account_service_key": self.__account_service_key}
        return state

    def __setstate__(self, state):
        self.__ML_model_file_id = state['__ML_model_file_id']
        self.__ML_model_script_file_id = state['__ML_model_script_file_id']
        self.__ML_model_class_name = state['__ML_model_class_name']
        self.__loss_function_file_id = state['__loss_function_file_id']
        self.__optimizer_type = state['__optimizer_type']
        self.__dataloader_file_id = ['__dataloader_file_id']
        self.__account_service_key = state['__account_service_key']


    def get_client(self):
        """
        Function to get the client of the GCP bucket
        :return: the client of the GCP bucket
        """
        ACCOUNT_SERVICE_KEY = os.environ.get('ACCOUNT_SERVICE_KEY')
        # Check if ACCOUNT_SERVICE_KEY is defined
        if ACCOUNT_SERVICE_KEY is None:
            raise ValueError("ACCOUNT_SERVICE_KEY environment variable is not defined")
        # Check if BUCKET_NAME is defined
        if self.bucket_name is None:
            raise ValueError("BUCKET_NAME environment variable is not defined")
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ACCOUNT_SERVICE_KEY
        os.environ["DONT_PICKLE"] = 'False'
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        return bucket

    def upload_to_gcp(self, dest_file_path, src_file_name):
        """
        Function to upload files to the GCP bucket
        :param dest_file_path: the path of the file in the bucket
        :param src_file_name: the path of the file in the local machine
        :return:
        """
        try:
            bucket = self.get_client()
            blob = bucket.blob(dest_file_path)
            blob.upload_from_filename(src_file_name)
        except Exception as e:
            # Handle the exception here, you can log the error or take appropriate actions
            logging.error(f"An error occurred while uploading to GCP: {e}")
            logging.error(traceback.format_exc())
            # You can also raise the exception again if you want to propagate it



    def download_from_gcp(self, src_file_name,dest_file_name=None,folder=False):
        """
              Function to download files from the GCP bucket
              :param src_file_name: the path of the file in the bucket
              :param dest_file_name: the path of the file in the local machine
              :return:
              """
        bucket = self.get_client()
        src_file_name = BucketLoader.reformat_path(src_file_name, self.bucket_name)
        if folder:
            blobs = bucket.list_blobs()
            path = os.path.join(self.path_to_dataset_files_dir, "test_set")
            os.mkdir(path)
            for blob in blobs:
                file_name = blob.name
                if file_name.find(src_file_name) != -1 and not file_name.endswith(src_file_name + "/"):
                    dir = os.path.dirname(file_name)
                    full_dir = os.path.join(path, dir[len(src_file_name) + 1:])
                    if not os.path.exists(full_dir):
                        os.makedirs(full_dir)
                    dest = os.path.join(path , file_name[len(src_file_name) + 1:])


                    blob.download_to_filename(dest)
            return
        blob = bucket.blob(src_file_name)
        if dest_file_name:
            blob.download_to_filename(dest_file_name)
        else:
            blob.download_to_filename(src_file_name)

    @staticmethod
    def reformat_path(path, bucket_name):
        start_index = path.find(bucket_name)
        if start_index == -1:
            return path
        return path[start_index + len(bucket_name) + 1:]

    # Getters starts here
    # All getters methods are downloading files from GCP bucket.

    def get_model(self):
        """
                Function to get the ML model from the bucket
      """
        print(f"Downloading model script from GCP bucket.:{self.path_to_model_files_dir}")
        framework = self.metadata['ml_model']['meta']['framework']
        if framework == "pytorch" or framework == "tensorflow" or framework == "keras":
            model_script_dest = self.path_to_model_files_dir + "/model_def.py"

            self.download_from_gcp(src_file_name=self.__ML_model_script_file_URL,dest_file_name=model_script_dest)

        framework = self.metadata['ml_model']['meta']['framework']
        if framework == "sklearn":
            # downloading model pickle file
            model_bin_dest = self.path_to_model_files_dir + "/model.pickle"
            self.download_from_gcp(src_file_name=self.__ML_model_file_URL,dest_file_name=model_bin_dest)

        elif framework == "pytorch":
            model_dest = self.path_to_model_files_dir + "/parameters.pth"
            self.download_from_gcp(src_file_name=self.__ML_model_file_URL, dest_file_name=model_dest)

        elif framework == "tensorflow" or framework == "keras":
            model_dest = self.path_to_model_files_dir + "/model.keras"
            self.download_from_gcp(src_file_name=self.__ML_model_file_URL,  dest_file_name=model_dest)

        elif framework == "xgboost":
            model_dest = self.path_to_model_files_dir + "/model.json"
            self.download_from_gcp(src_file_name=self.__ML_model_file_URL, dest_file_name=model_dest)

        elif framework == "catboost":
            model_dest = self.path_to_model_files_dir + "/model.cbm"
            self.download_from_gcp(src_file_name=self.__ML_model_file_URL,  dest_file_name=model_dest)


    # def get_file(self, file):
    #     """
    #     Function to get all files in the bucket ecxc
    #     :param file: the name and format of the file to download
    #
    #     """
    #     if file== None:
    #         raise TypeError('file need to be type str')
    #     file_name, file_format = file.split(".")
    #     if file_name + file_format == self.__loss_function_file_id:
    #         url = self.__loss_function_file_URL
    #     else:
    #         raise Exception(f"Expected loss, optimizer or requirements. Got{file_name}")
    #     try:
    #         loss_dest = get_files_package_root() + file_name + file_format
    #         self.download_from_gcp(src_file_name=url,
    #                                dest_file_name=loss_dest)
    #     except Exception as err:
    #         error_logger.error(f"Downloading {file_name} from GCP bucket failed!\nError:\n{err}")

    def get_dataloader(self):
        """
                Function to get the dataloader from the bucket
                """

        dataloader_dest = self.path_to_dataloader_files_dir + f"/dataloader_def.py"
        self.download_from_gcp(src_file_name=self.__dataloader_file_URL,dest_file_name=dataloader_dest)



    def get_dataset(self):
        def get_data_format():
            dataset_path = self.__test_set_URL
            format = dataset_path.rfind(".")
            if format == -1:
                return None
            return dataset_path[format:]
            # client = self.get_client()
            # blobs = client.list_blobs()
            # print(blobs)
            # file_names = [blob.name for blob in blobs]
            # print(file_names)
            # true_file_name = self.__test_set_id
            # for file_name in file_names:
            #     file_format_index = file_name.rfind(".")
            #     file_format = file_name[file_format_index:]
            #     if (true_file_name + "/") in file_name:
            #         return None
            #     elif file_format_index == -1:
            #         continue
            #     elif file_name.endswith(true_file_name + file_format):
            #         print(file_name)
            #         return file_format
            #     else:
            #         continue

            # raise Exception("Dataset not found in Bucket")


        data_format = get_data_format()
        is_folder = False
        if data_format is None:
            data_format = ""
            is_folder = True

        dataset_dest = self.path_to_dataset_files_dir + "/test_set" + data_format
        print(f"Downloading test set from GCP bucket.:{dataset_dest}")
        self.download_from_gcp(src_file_name=self.__test_set_URL, dest_file_name=dataset_dest,folder=is_folder)

        if data_format[1:] == "zip":
            import zipfile
            folder_to_extract_zip = self.path_to_dataset_files_dir + "/test_set"
            with zipfile.ZipFile(dataset_dest, "r") as zip_ref:
                zip_ref.extractall(folder_to_extract_zip)
            os.remove(dataset_dest)



    def get_loss(self):
        loss_dest = self.path_to_loss_files_dir + f"/loss.py"
        self.download_from_gcp(src_file_name=self.__loss_function_file_URL, dest_file_name=loss_dest)


    def get_req_file(self):
        req_dest = self.path_to_req_files_dir + f"/user_requirements.txt"

        if self.__requirements_file_URL:
            self.download_from_gcp(src_file_name=self.__requirements_file_URL, dest_file_name=req_dest)

        req_dest_app = self.path_to_req_files_dir + f"/requirements.txt"
        framework = self.metadata['ml_model']['meta']['framework']
        if framework == "tensorflow" or framework == "keras":
            base_file = "https://storage.cloud.google.com/e2e-mabadata/requirements/requirements-tensorflow.txt"
            self.download_from_gcp(src_file_name=base_file, dest_file_name=req_dest_app)

        if framework == "pytorch":
            base_file = "https://storage.cloud.google.com/e2e-mabadata/requirements/requirements-torch.txt"
            self.download_from_gcp(src_file_name=base_file, dest_file_name=req_dest_app)

        if framework == "sklearn" or framework == "xgboost" or framework == "catboost":
            base_file = "https://storage.cloud.google.com/e2e-mabadata/requirements/requirements-xgboost_sklearn.txt"
            self.download_from_gcp(src_file_name=base_file, dest_file_name=req_dest_app)

            # user_file = req_dest
            # output_file = req_dest
            # self.merge_requirements(base_file, user_file, output_file)

    # def upload(self, obj, obj_type, to_pickle=True):
    #     hashmap = {"ML_model": self.__ML_model_file_id,
    #                "loss": self.__loss_function_file_id,
    #                "optimizer": self.__optimizer_file_id,
    #                "dataloader": self.__dataloader_file_id,
    #                "requirements-TF.txt": self.__requirements_file_id,
    #                "Estimator_params": "Estimator_params.json",
    #                "attack_defence_metadata": "attack_defence_metadata.json"}
    #     try:
    #         file_name = hashmap[obj_type]
    #         if to_pickle:
    #             logging.info(f'Dumping {obj_type} to pickle file...')
    #             self.to_pickle(file_name, obj)
    #             file_name += ".pickle"
    #         logging.info(f'Uploading {obj_type} to GCP bucket starts...')
    #         self.upload_to_gcp(file_name, file_name)
    #         logging.info('Upload was successful!')
    #     except Exception as err:
    #
    #         logging.error(f"Uploading {obj_type} to GCP failed!\nError:\n{err}")