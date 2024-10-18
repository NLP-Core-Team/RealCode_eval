# RealCode_eval
**RealCode_eval** is a benchmark to perform **execution-based** evaluation of LLM code generation capabilities in **real Github repositories**. The model-generated code is evaluated by running tests in respective repositories.

**RealCode v3** includes two indenpendent benchmarks: **FG (Function Generation)** and **SG (Scope Generation)**. Each benchmark has **1000 tasks** built from **154 Python GitHub repositories**. 

To avoid data contamination for popular Code LLMs we only use repositories created in 2024. 

## Realcode v3 Function Generation
Each task of Realcode v3 FG requires the model to generate the body of a function (or of a class method), based on a function signature, a **docstring** and the rest of source code file.

<details>
<summary> Example
</summary>

```python
"""
Feature Extraction Methods for DimSense
"""

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.base import BaseEstimator, TransformerMixin


class AutoencoderExtractor(BaseEstimator, TransformerMixin):
    """
    AutoencoderExtractor provides feature extraction using autoencoders.
    """
    def __init__(self, encoding_dim=10):
        """
        Initialize the AutoencoderExtractor.

        Parameters:
        - encoding_dim (int): Dimension of the encoded representation.
        """
        self.tf = None 
        self.encoding_dim = encoding_dim

    def _import_tensorflow(self):
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            raise ImportError("TensorFlow is required for using AutoencoderExtractor.")      

    def build_autoencoder(self):
        if self.tf is not None:
            input_layer = self.tf.keras.layers.Input(shape=(self.input_dim,))
            encoded = tf.keras.layers.Dense(self.encoding_dim, activation='relu')(input_layer)
            decoded = tf.keras.layers.Dense(self.input_dim, activation='sigmoid')(encoded)
            autoencoder = tf.keras.models.Model(input_layer, decoded)
            autoencoder.compile(optimizer='adam', loss='mean_squared_error')
            return autoencoder
        else: return None

    def fit_transform(self, X):
        """
        Fit the autoencoder model and transform the data.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - X_extracted (array-like): Extracted features.
        """
# >>> THIS NEEDS TO BE GENERATED >>>>
        if self.tf is None:
            self._import_tensorflow()
        self.input_dim = X.shape[1]
        self.autoencoder = self.build_autoencoder()
        self.autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, verbose=0)
        encoder = tf.keras.models.Model(inputs=self.autoencoder.input, outputs=self.autoencoder.layers[1].output)
        X_extracted = encoder.predict(X)
        return X_extracted
# <<<< <<<<

    def set_encoding_dim(self, encoding_dim):
        """
        Set the dimension of the encoded representation.

        Parameters:
        - encoding_dim (int): Dimension of the encoded representation.
        """
        self.encoding_dim = encoding_dim
        self.autoencoder = self.build_autoencoder()

    ...
```
</details>

## Realcode v3 Scope Generation
Each task of Realcode v3 SG requires the model to generate an arbitrary block of code (a body of a function, a for-loop, an if-statement, etc.), based on the rest of the source code file. Unlike FG task, the docstring (or any other description of the code in natural language) of the code may not be provided.

<details>
<summary> 
Example
</summary>

```python
import click
from geekbot_cli.api_client import APIClient
from geekbot_cli.config_manager import ConfigManager
from geekbot_cli.cli import CLI
import sys

@click.command()
@click.option('--clear-api-key', is_flag=True, help='Removes the saved API key from keyring')
def main(clear_api_key):
    """
    Entry point for the CLI that can now handle `--clear-api-key` to remove the saved API key.
    """
    config_manager = ConfigManager()
    if clear_api_key:
        if click.confirm('Are you sure you want to remove the API key?'):
# >>> THIS NEEDS TO BE GENERATED >>>>
            config_manager.delete_api_key()
            click.echo("API key has been removed.")
# <<<< <<<<
        else:
            click.echo("Operation cancelled.")
    else:
        # Normal CLI operation
        try:
            api_client = APIClient()
            cli = CLI(api_client, config_manager)
            cli.start()
        except Exception as e:
            click.echo(f"Error: {e}")
            sys.exit(1)

if __name__ == '__main__':
    main()
```
</details>

## Evaluation
We use the following evaluation procedure for each task and the generated code snippet:
1. The generated code snippet is placed in the appropriate position of the source code file. 
2. The entire repository gets copied to ./workdir, including the file from Step 1.
3. All tests are executed in the copied repository.
4. If the number of passed tests differs from the prerecorded number of the passed tests in the repository, we consider the generated code incorrect. If the two numbers are equal, the code is correct. 

# Getting started
Every repository in RealCode has dependencies and, as a result, necessitates properly configured environments. We utilize Conda to create individual environments for each repository.

**1.** Install requirements in your main environment
```python
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

**2.** Download repositories and meta files
```
wget https://zenodo.org/records/13378983/files/realcode_v3_repos_upd.tar.gz
tar -xvf ../RealCode_eval/realcode_v3_repos_upd.tar.gz -C data
```
Expected file structure:
```
data/realcode_v3/realcode_v3_SG.json
data/realcode_v3/realcode_v3_FG.json
data/realcode_v3/*Repository names*
```

**3.** Build environments for each repository in the benchmark (takes about an hour)
```bash
cd prepare_data
python run.py
cd ..
```

**4.** Check installation **(IMPORTANT!)**
```bash
pytest tests/test_evaluator.py
```
> [!NOTE]
> Number of passed tests in the repositories may vary depending on your system. If this test fails on your system feel free to open an issue. We need your feedback to create a more stable version of the benchmark.

**5.** Run the evaluation of your model (see config/config.yaml for details). E.g. for [codeparrot-small](https://huggingface.co/codeparrot/codeparrot-small):
```bash
CUDA_VISIBLE_DEVICES=0 python main.py +model=codeparrot generation_params.max_new_tokens=512 max_context_length=500
```
> [!WARNING]
> **Generated code is executed without any isolation in the benchmark! The repositories themselves are checked to be safe, but the generated code may not!**

# Examples
* Run deepseek-ai/deepseek-coder-1.3b-base with left context only, 1024 tokens in prompt:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py +model=deepseek size=1.3b max_context_length=1024
``` 
* Run deepseek-ai/deepseek-coder-1.3b-base with **left and right context**, 1024 tokens in prompt:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py +model=deepseek size=1.3b max_context_length=1024
``` 
* (Recommended mode) Run deepseek-ai/deepseek-coder-1.3b-base with left and right context, 1024 tokens in prompt, **3:1 left to right context ratio**:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py +model=deepseek size=1.3b max_context_length=1024 left_context_ratio=3
``` 
* You can evalute your awesome model from a local HuggingFace checkpoint:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py +model=local model_path=*path_to_HF_checkpoint* model_name=*my_awesome_model* max_context_length=1024
``` 
> [!NOTE]
> You may need to edit the config/model/local.yaml with FIM tokens for your model

* The model is inferenced with device_map='auto' by default. If you instead wish to distribute tasks between several GPUs, you can use accelerate:
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 main.py +model=deepseek size=1.3b max_context_length=1024
``` 

