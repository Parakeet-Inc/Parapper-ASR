import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "verify_cat_onnx_distribution.py"
REQUIREMENTS = Path(__file__).parents[1] / "requirements-cat-onnx.txt"
ENVIRONMENT_CHECKER = Path(__file__).parents[1] / "cat_export_environment.py"

RUNTIME_FILES = [
    "chat_template.jinja",
    "genai_config.json",
    "model_q4.onnx",
    "model_q4.onnx.data",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]

PUBLICATION_FILES = [
    "LICENSE",
    "MODEL_CARD.md",
    "THIRD_PARTY_NOTICES.md",
    "build-metadata.json",
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class VerifyCatOnnxDistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        self.fake_modules = self.root / "fake-modules"
        self.fake_modules.mkdir()
        self._write_fake_onnx_module()
        self._write_fake_genai_module()
        self._write_valid_distribution()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_fake_onnx_module(self):
        (self.fake_modules / "onnx.py").write_text(
            """
import os

class TensorProto:
    FLOAT = 1
    UINT4 = 21

class Attribute:
    def __init__(self, name, value):
        self.name = name
        self.i = value

class Node:
    def __init__(self, op_type, inputs=(), bits=None, name='', attrs=None):
        self.op_type = op_type
        self.input = list(inputs)
        self.name = name
        self.attribute = [] if bits is None else [Attribute('bits', bits)]
        self.attribute.extend(
            Attribute(key, value) for key, value in (attrs or {}).items()
        )

class Initializer:
    def __init__(self, name, dims, data_type):
        self.name = name
        self.dims = dims
        self.data_type = data_type

class Graph:
    def __init__(self):
        q4_count = int(os.environ.get('FAKE_Q4_COUNT', '120'))
        self.node = [Node('MatMulNBits', bits=4) for _ in range(q4_count)]
        self.node.append(Node('MatMulNBits', bits=8, name='/lm_head/MatMul_Q4'))
        if os.environ.get('FAKE_EMBEDDING_FP32') == '1':
            self.node.append(Node('Gather', ('model.embed_tokens.weight',), name='/model/embed_tokens/Gather'))
        else:
            block_size = int(os.environ.get('FAKE_EMBEDDING_BLOCK_SIZE', '16'))
            self.node.append(Node(
                'GatherBlockQuantized',
                (
                    'model.embed_tokens.weight_Q4',
                    'input_ids',
                    'model.embed_tokens.weight_scales',
                    'model.embed_tokens.weight_zero_points',
                ),
                name='/model/embed_tokens/Gather_Q4',
                attrs={'block_size': block_size, 'gather_axis': 0, 'quantize_axis': 1},
            ))
        self.node.append(Node('Gather', ('position_ids',)))
        self.initializer = [
            Initializer(
                'model.embed_tokens.weight_Q4',
                [102400, 1280],
                TensorProto.UINT4,
            ),
            Initializer(
                'model.embed_tokens.weight_scales',
                [102400, 80],
                TensorProto.FLOAT,
            ),
            Initializer(
                'model.embed_tokens.weight_zero_points',
                [102400, 80],
                TensorProto.UINT4,
            ),
        ]

class Model:
    def __init__(self):
        self.graph = Graph()

def load(path, load_external_data=False):
    return Model()
""".strip()
            + "\n",
            encoding="utf-8",
        )

    def _write_fake_genai_module(self):
        (self.fake_modules / "onnxruntime_genai.py").write_text(
            """
from pathlib import Path

class Model:
    def __init__(self, model_dir):
        data = Path(model_dir, 'model_q4.onnx.data').read_bytes()
        if data != b'model_q4.onnx.data\\n':
            raise RuntimeError('external tensor data rejected')
""".strip()
            + "\n",
            encoding="utf-8",
        )

    def _write_valid_distribution(self):
        for name in RUNTIME_FILES + PUBLICATION_FILES:
            path = self.model_dir / name
            path.write_bytes((name + "\n").encode())

        (self.model_dir / "genai_config.json").write_text(
            json.dumps({"model": {"decoder": {"filename": "model_q4.onnx"}}}),
            encoding="utf-8",
        )
        (self.model_dir / "special_tokens_map.json").write_text("{}\n", encoding="utf-8")
        (self.model_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (self.model_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        (self.model_dir / "LICENSE").write_text(
            "MIT License\n\nCopyright (c) 2026 CyberAgent AI Lab\n",
            encoding="utf-8",
        )
        (self.model_dir / "MODEL_CARD.md").write_text(
            "CAT-Translate k_quant Q4 block16 embedding\n"
            "b555f93ef67846b6ed2773e0d2f16ceb0d30adb9\n",
            encoding="utf-8",
        )
        (self.model_dir / "THIRD_PARTY_NOTICES.md").write_text(
            "cyberagent/CAT-Translate-0.8b\nMIT License\n"
            "b555f93ef67846b6ed2773e0d2f16ceb0d30adb9\n",
            encoding="utf-8",
        )
        (self.model_dir / "build-metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source": {
                        "repository": "cyberagent/CAT-Translate-0.8b",
                        "revision": "b555f93ef67846b6ed2773e0d2f16ceb0d30adb9",
                        "license": "MIT",
                    },
                    "export": {
                        "variant": "k_quant",
                        "precision": "int4",
                        "execution_provider": "cpu",
                        "embedding": "groupwise_q4_block16",
                        "embedding_quantization": {
                            "bits": 4,
                            "block_size": 16,
                            "is_symmetric": False,
                            "operator": "GatherBlockQuantized",
                            "command": [
                                "python",
                                "quantize_cat_embedding_gather.py",
                                "<INTERMEDIATE_DIR>",
                                "<OUT_DIR>",
                                "--block-size",
                                "16",
                            ],
                        },
                        "command": [
                            "python",
                            "-m",
                            "onnxruntime_genai.models.builder",
                            "-i",
                            "<SOURCE_DIR>",
                            "-o",
                            "<INTERMEDIATE_DIR>",
                            "-c",
                            "<CACHE_DIR>",
                            "-p",
                            "int4",
                            "-e",
                            "cpu",
                            "--extra_options",
                            "filename=model_q4.onnx",
                            "hf_token=false",
                            "hf_remote=false",
                            "int4_algo_config=k_quant",
                        ],
                        "duration_seconds": 123.4,
                    },
                    "environment": {
                        "python": "3.12.10",
                        "packages": {
                            "onnxruntime-genai": "0.14.1",
                            "onnxruntime": "1.27.0",
                            "onnx": "1.22.0",
                            "onnx-ir": "0.2.1",
                            "transformers": "4.57.6",
                            "huggingface-hub": "0.36.2",
                            "torch": "2.12.1+cpu",
                            "tokenizers": "0.22.2",
                            "sentencepiece": "0.2.1",
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def _run(self, *args, env=None):
        command_env = os.environ.copy()
        command_env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(self.fake_modules), command_env.get("PYTHONPATH")])
        )
        command_env.update(env or {})
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.model_dir), *args],
            capture_output=True,
            text=True,
            env=command_env,
            check=False,
        )

    def test_candidate_graph_writes_manifest_and_checksum_that_reverify(self):
        generated = self._run("--write-manifest")

        self.assertEqual(generated.returncode, 0, generated.stderr)
        manifest = json.loads(
            (self.model_dir / "distribution-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["graph"],
            {
                "mat_mul_n_bits": {"4": 120, "8": 1},
                "gather": 1,
                "gather_block_quantized": 1,
                "embedding": {
                    "data_type": "UINT4",
                    "shape": [102400, 1280],
                    "scale": {
                        "data_type": "FLOAT",
                        "shape": [102400, 80],
                    },
                    "zero_point": {
                        "data_type": "UINT4",
                        "shape": [102400, 80],
                    },
                    "block_size": 16,
                    "gather_axis": 0,
                    "quantize_axis": 1,
                },
            },
        )
        self.assertEqual(set(manifest["files"]), set(RUNTIME_FILES + PUBLICATION_FILES))
        self.assertTrue((self.model_dir / "SHA256SUMS").is_file())

        verified = self._run()
        self.assertEqual(verified.returncode, 0, verified.stderr)
        self.assertIn("distribution verified", verified.stdout)

        (self.model_dir / "tokenizer.model").write_bytes(b"tampered\n")
        tampered = self._run()
        self.assertNotEqual(tampered.returncode, 0)
        self.assertIn("manifest does not match", tampered.stderr)

    def test_missing_runtime_file_fails_before_manifest_is_written(self):
        (self.model_dir / "tokenizer.model").unlink()

        result = self._run("--write-manifest")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing required distribution file", result.stderr)
        self.assertIn("tokenizer.model", result.stderr)
        self.assertFalse((self.model_dir / "distribution-manifest.json").exists())

    def test_fp32_embedding_graph_is_rejected_as_non_candidate(self):
        result = self._run("--write-manifest", env={"FAKE_EMBEDDING_FP32": "1"})

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Q4 block16 embedding", result.stderr)
        self.assertFalse((self.model_dir / "distribution-manifest.json").exists())

    def test_embedding_block32_graph_is_rejected_as_non_candidate(self):
        result = self._run(
            "--write-manifest",
            env={"FAKE_EMBEDDING_BLOCK_SIZE": "32"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("block_size=16", result.stderr)
        self.assertFalse((self.model_dir / "distribution-manifest.json").exists())

    def test_non_k_quant_bit_layout_is_rejected(self):
        result = self._run("--write-manifest", env={"FAKE_Q4_COUNT": "121"})

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected MatMulNBits bit layout", result.stderr)

    def test_corrupt_external_tensor_data_is_rejected_before_manifest_is_written(self):
        (self.model_dir / "model_q4.onnx.data").write_bytes(b"corrupt\n")

        result = self._run("--write-manifest")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime model", result.stderr)
        self.assertFalse((self.model_dir / "distribution-manifest.json").exists())

    def test_local_path_or_credential_marker_is_rejected(self):
        metadata = json.loads(
            (self.model_dir / "build-metadata.json").read_text(encoding="utf-8")
        )
        metadata["export"]["source_dir"] = r"C:\\Users\\alice\\.cache\\huggingface"
        metadata["export"]["hf_token"] = "secret-value"
        (self.model_dir / "build-metadata.json").write_text(
            json.dumps(metadata) + "\n", encoding="utf-8"
        )

        result = self._run("--write-manifest")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("local path or credential", result.stderr)

    def test_generated_config_with_local_absolute_path_is_rejected(self):
        (self.model_dir / "genai_config.json").write_text(
            json.dumps(
                {
                    "model": {
                        "decoder": {
                            "filename": r"C:\Users\alice\exports\model_q4.onnx"
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        result = self._run("--write-manifest")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("local path or credential", result.stderr)

    def test_placeholder_license_is_rejected(self):
        (self.model_dir / "LICENSE").write_text("MIT\n", encoding="utf-8")

        result = self._run("--write-manifest")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source MIT license", result.stderr)

    def test_requirements_versions_match_build_metadata_contract(self):
        pinned = {}
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("--"):
                continue
            package, version = line.split("==", 1)
            pinned[package] = version
        metadata = json.loads(
            (self.model_dir / "build-metadata.json").read_text(encoding="utf-8")
        )
        environment_checker = load_module("cat_export_environment", ENVIRONMENT_CHECKER)
        verifier = load_module("verify_cat_onnx_distribution", SCRIPT)

        self.assertEqual(pinned, metadata["environment"]["packages"])
        self.assertEqual(pinned, environment_checker.EXPECTED_PACKAGES)
        self.assertEqual(pinned, verifier.EXPECTED_PACKAGES)


if __name__ == "__main__":
    unittest.main()
