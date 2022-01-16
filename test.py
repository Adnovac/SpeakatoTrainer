import unittest
import datetime
import os
import stat
import shutil

from src import SpeakatoTrainer


class TestExceptions(unittest.TestCase):

    test_1_model_path = f"models/Speakato_model_{datetime.datetime.now().date()}_test_1"
    test_2_model_path = f"models/Speakato_model_{datetime.datetime.now().date()}_test_2"
    test_3_model_path = f"models/Speakato_model_{datetime.datetime.now().date()}_test_3"
    test_4_model_path = f"models/Speakato_model_{datetime.datetime.now().date()}_test_4"
    polish_dataset_path = f"examples\polish_commands_dataset"
    test_language = "pl"
    test_mode = "1"

    def test_lang_int(self):
        """
        Testing if the wrong language raises an exception

        test_1 model doesn't exist, but if the test will go wrong, there is a possibility that folder could be created.
        """
        self.delete_folder(self.test_1_model_path)
        
        language = "jp"
        with self.assertRaises(Exception) as context:
            A = SpeakatoTrainer.SpeakatoTrainer(language, self.test_1_model_path , self.polish_dataset_path, self.test_mode)

        self.assertTrue('Language not available! Supported languages: pl/eng' in str(context.exception))

    def test_mode_int(self):
        """
        Testing if the wrong mode raises an exception

        test_4 model doesn't exist, but if the test will go wrong, there is a possibility that folder could be created.
        """
        self.delete_folder(self.test_4_model_path)
        
        mode = "3"
        with self.assertRaises(Exception) as context:
            A = SpeakatoTrainer.SpeakatoTrainer(self.test_language, self.test_1_model_path , self.polish_dataset_path, mode)

        self.assertTrue('Wrong mode selected! Supported modes: Mode 1: Create model, Mode 2: add new data to previously created model' in str(context.exception))
    
    def test_model_int(self):
        """
        Testing if the duplicate Model path raises an exception
        """
        self.delete_folder(self.test_2_model_path)

        A = SpeakatoTrainer.SpeakatoTrainer(self.test_language, self.test_2_model_path , self.polish_dataset_path, self.test_mode)

        with self.assertRaises(Exception) as context:
            B = SpeakatoTrainer.SpeakatoTrainer(self.test_language, self.test_2_model_path , self.polish_dataset_path, self.test_mode)

        self.assertTrue(f"Model: models/Speakato_model_{datetime.datetime.now().date()}_test_2 already exists!" in str(context.exception))
    
    def test_data_int(self):
        """
        Testing if the wrong dataset path raises an exception

        test_3 model doesn't exist, but if the test will go wrong, there is a possibility that folder could be created.
        """
        self.delete_folder(self.test_3_model_path)

        with self.assertRaises(Exception) as context:
            A = SpeakatoTrainer.SpeakatoTrainer(self.test_language, self.test_3_model_path , f"examples\japanese_commands_dataset", self.test_mode)

        self.assertTrue(f"Dataset: examples\japanese_commands_dataset doesn't exists!" in str(context.exception))
    
    def delete_folder(self, modelPath : str):
         if os.path.exists(modelPath):
            os.chmod(modelPath, stat.S_IWRITE)
            shutil.rmtree(modelPath)
    

if __name__ == '__main__':
    unittest.main()