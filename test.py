import unittest
import datetime
import os
import stat
import shutil

from src import SpeakatoTrainer


class TestLanguage(unittest.TestCase):
    def test_list_int(self):
        """
        Testing if the wrong language raises an exception

        test_1 model doesn't exist, but if the test will go wrong, there is a possibility that folder could be created.
        """
        if os.path.exists(f"models/Speakato_model_{datetime.datetime.now().date()}_test_1"):
            os.chmod( f"models/Speakato_model_{datetime.datetime.now().date()}_test_1", stat.S_IWRITE )
            shutil.rmtree( f"models/Speakato_model_{datetime.datetime.now().date()}_test_1")
        
        language = "jp"
        with self.assertRaises(Exception) as context:
            A = SpeakatoTrainer.SpeakatoTrainer(language, f"models/Speakato_model_{datetime.datetime.now().date()}_test_1" , f"examples\polish_commands_dataset")

        self.assertTrue('Language not available. Supported languages: pl/eng' in str(context.exception))

class TestModelPath(unittest.TestCase):
    def test_list_int(self):
        """
        Testing if the duplicate Model path raises an exception
        """
        if os.path.exists(f"models/Speakato_model_{datetime.datetime.now().date()}_test_2"):
            os.chmod( f"models/Speakato_model_{datetime.datetime.now().date()}_test_2", stat.S_IWRITE )
            shutil.rmtree( f"models/Speakato_model_{datetime.datetime.now().date()}_test_2")
        A= SpeakatoTrainer.SpeakatoTrainer("pl", f"models/Speakato_model_{datetime.datetime.now().date()}_test_2" , f"examples\polish_commands_dataset")

        with self.assertRaises(Exception) as context:
            B = SpeakatoTrainer.SpeakatoTrainer("pl", f"models/Speakato_model_{datetime.datetime.now().date()}_test_2" , f"examples\polish_commands_dataset")

        self.assertTrue(f"Model: models/Speakato_model_{datetime.datetime.now().date()}_test_2 already exists!" in str(context.exception))

class TestExampleDatasetPath(unittest.TestCase):
    def test_list_int(self):
        """
        Testing if the wrong dataset path raises an exception

        test_3 model doesn't exist, but if the test will go wrong, there is a possibility that folder could be created.
        """
        if os.path.exists(f"models/Speakato_model_{datetime.datetime.now().date()}_test_3"):
            os.chmod( f"models/Speakato_model_{datetime.datetime.now().date()}_test_3", stat.S_IWRITE )
            shutil.rmtree( f"models/Speakato_model_{datetime.datetime.now().date()}_test_3")

        with self.assertRaises(Exception) as context:
            A = SpeakatoTrainer.SpeakatoTrainer("pl", f"models/Speakato_model_{datetime.datetime.now().date()}_test_3" , f"examples\japanese_commands_dataset")

        self.assertTrue(f"Dataset: examples\japanese_commands_dataset doesn't exists!" in str(context.exception))

if __name__ == '__main__':
    unittest.main()