import unittest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

# Import module under test
import hierarchical.config as config_module


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Reset global state before each test
        config_module._config = None

        # Create a temporary config file
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)
        self.config_path = self.tmp_dir / "test_config.yaml"

        self.config_data = {
            "paths": {"data_dir": "data/raw", "output_dir": "/abs/output"},
            "banks": ["BankA", "BankB"],
            "model": {"hidden_dim": 64},
        }

        with open(self.config_path, "w") as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        config_module._config = None
        self.tmp_dir_obj.cleanup()

    def test_load_explicit_path(self):
        cfg = config_module.load_config(self.config_path)
        self.assertEqual(cfg["model"]["hidden_dim"], 64)
        self.assertIsNotNone(config_module._config)

    @patch.dict(os.environ, {"EMBEDDER_CONFIG": ""})  # Ensure unset
    def test_load_env_var(self):
        os.environ["EMBEDDER_CONFIG"] = str(self.config_path)
        try:
            cfg = config_module.load_config()
            self.assertEqual(cfg["banks"], ["BankA", "BankB"])
        finally:
            del os.environ["EMBEDDER_CONFIG"]

    def test_get_config_raises_if_not_loaded(self):
        # If we just call get_config() and no default exists, what happens?
        # The code tries load_config().
        # If actual default file missing, it raises FileNotFoundError.
        # We can mock DEFAULT_CONFIG_PATH to a non-existent path to verify crash.
        with patch(
            "hierarchical.config.DEFAULT_CONFIG_PATH", Path("/non/existent/path.yaml")
        ):
            with self.assertRaises(FileNotFoundError):
                config_module.get_config()

    def test_get_path_resolution(self):
        config_module.load_config(self.config_path)

        # Absolute path should remain absolute
        # (Assuming Linux paths)
        out_path = config_module.get_path("output_dir")
        self.assertTrue(out_path.is_absolute())
        self.assertEqual(str(out_path), "/abs/output")

        # Relative path should become absolute (cwd join)
        data_path = config_module.get_path("data_dir")
        self.assertTrue(data_path.is_absolute())
        expected = Path.cwd() / "data/raw"
        self.assertEqual(data_path, expected)

    def test_get_path_errors(self):
        config_module.load_config(self.config_path)

        # Required missing
        with self.assertRaises(ValueError):
            config_module.get_path("missing_key", required=True)

        # Optional missing
        self.assertIsNone(config_module.get_path("missing_key", required=False))

    def test_helper_getters(self):
        config_module.load_config(self.config_path)
        self.assertEqual(config_module.get_banks(), ["BankA", "BankB"])
        self.assertEqual(config_module.get_model_config()["hidden_dim"], 64)
        self.assertEqual(config_module.get_training_config(), {})  # Empty in our mock


if __name__ == "__main__":
    unittest.main()
