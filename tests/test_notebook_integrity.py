import base64
import json
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "frozenlake_comparison.ipynb"


class NotebookIntegrityTests(unittest.TestCase):
    def test_notebook_exists_and_is_valid_json(self):
        self.assertTrue(NOTEBOOK_PATH.is_file())
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(notebook.get("nbformat"), 4)
        self.assertIsInstance(notebook.get("cells"), list)

    def test_code_cells_parse_without_execution(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"cell_{index}", "exec")

    def test_module_imports_and_original_configuration_remain(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn("from q_learning import QLearningAgent", source)
        self.assertIn("from policy_gradient import PolicyGradientAgent", source)
        root_setup = source.index("PROJECT_ROOT = next")
        self.assertLess(root_setup, source.index("from q_learning import QLearningAgent"))
        self.assertLess(root_setup, source.index("from policy_gradient import PolicyGradientAgent"))
        self.assertIn("NUM_EPISODES = 1000", source)
        self.assertIn('map_name="8x8", is_slippery=True', source)
        self.assertIn("alpha=0.1", source)
        self.assertIn("learning_rate=0.01", source)
        self.assertIn("GAMMA = 0.99", source)

    def test_algorithm_imports_resolve_from_root_and_notebooks_directory(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        source = next(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            and "PROJECT_ROOT = next" in "".join(cell.get("source", []))
        )
        setup = source[
            source.index("from pathlib import Path") : source.index(
                "from policy_gradient import PolicyGradientAgent"
            )
        ]
        import_check = (
            "import sys\n"
            + setup
            + "from q_learning import QLearningAgent\n"
            + "from policy_gradient import PolicyGradientAgent\n"
            + "print(PROJECT_ROOT.resolve())\n"
        )

        for working_directory in (PROJECT_ROOT, PROJECT_ROOT / "notebooks"):
            with self.subTest(working_directory=working_directory.name):
                result = subprocess.run(
                    [sys.executable, "-B", "-c", import_check],
                    cwd=working_directory,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    Path(result.stdout.strip()).resolve(), PROJECT_ROOT.resolve()
                )

    def test_saved_results_and_plot_are_preserved(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        saved_text = []
        png_outputs = []
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                saved_text.extend(output.get("text", []))
                image = output.get("data", {}).get("image/png")
                if image:
                    image_text = "".join(image) if isinstance(image, list) else image
                    png_outputs.append(base64.b64decode(image_text))

        self.assertIn("Q-Learning mean training reward: 0.0000", "".join(saved_text))
        self.assertIn("Policy Gradient mean training reward: 0.0010", "".join(saved_text))
        self.assertTrue(any(image.startswith(b"\x89PNG\r\n\x1a\n") for image in png_outputs))


if __name__ == "__main__":
    unittest.main()
