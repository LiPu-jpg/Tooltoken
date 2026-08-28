import unittest

from latent_register.toolgen_eval_adapter import install_template


class ToolGenEvalAdapterTest(unittest.TestCase):
    def test_installs_repository_template_in_target_registry(self):
        registry = {"llama-3": "stale"}
        calls = []

        def loader(name):
            calls.append(name)
            return {"name": name, "source": "toolgen"}

        install_template(registry, "llama-3", loader)
        self.assertEqual(calls, ["llama-3"])
        self.assertEqual(
            registry["llama-3"], {"name": "llama-3", "source": "toolgen"}
        )


if __name__ == "__main__":
    unittest.main()
