# RealCode_eval
**RealCode_eval** is a benchmark to perform execution-based evaluation of LLM code generation for real github repositories.
# Data
**RealCode** is a dataset of 219 Python functions\* from 22 github repositories published between June and August on 2023. All these functions are covered with tests in their respective repositories. \
\* our term "function" also includes methods in classes

<details>
<summary> 

### Example of a task
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



# Evaluation benchmark

The task for a model in **RealCode_eval** is to write the body of a function that is declared in a file within one of the repositories. The benchmark supplies the model with the remainder of the file or even the entire repository. If the number of tests passed with the generated body matches the precomputed number of passed tests for the repository, the generation is deemed correct. The Pass@k metric (from [Codex paper](https://arxiv.org/abs/2107.03374)) is employed for evaluation.

Every repository in RealCode has dependencies and, as a result, necessitates properly configured environments. We utilize Conda to create distinct environments for each repository.


# Evaluation of public code LMs
> [!NOTE]
> These results were obtained on the pre-release version of the dataset, which contained two more functions (221 instead of 219)

## LM mode, left context only, 1024 tokens
| +model    | size    |   Pass@1 |
|:----------|:--------|---------:|
| starcoder | 1b      | 0.3873  |
| starcoder | 7b      | 0.4814 |
| codellama | 7b      | 0.4760 |
| codellama | 13b     | 0.4841 |
| codellama | 34b     | 0.4932 |
| phi1      | 1b      | 0.3529 |
| mistral   | 7b     | 0.4208 |
| deepseek-coder  | 1.3b    | 0.4144  |
| deepseek-coder  | 5.7bmqa | 0.4669 |
| deepseek-coder  | 6.7b    | 0.4914 |
| deepseek-coder  | 33b     | 0.4932 |

## Infill mode, 512 tokens left context, 512 tokens right context
| +model       | size         |   Pass@1 |
|:-------------|:-------------|---------:|
| codellama    | 7b           | 0.4941 |
| codellama    | 13b          | 0.5339 |
| deepseek-coder     | 1.3b         | 0.3113 |
| deepseek-coder     | 5.7bmqa      | 0.5330 |
| deepseek-coder     | 6.7b         | 0.4832 |
| deepseek-coder     | 33b          | 0.5484 |
| starcoder    | 1b           | 0.4506 |
| starcoder    | 7b           | 0.5149 |
| starcoder    | 15b | 0.5248 |

> [!NOTE]
> If an "oracle" takes max Pass@1 for each function from the configurations presented in LM and Infill tables, he would score Pass@1=0.7085
## Repository-level mode\*, 15k tokens in context, 13.5k in left context for infill
| model    | generator_mode   |   size |   Pass@1 |
|:---------|:-----------------|:-------|---------:|
| deepseek | lm               |      1.3b | 0.5438 | 
| deepseek | lm               |      5.7bmqa | 0.5601 |
| deepseek | infill           |      5.7bmqa | 0.5891 |
| deepseek | lm               |      6.7b | 0.5954 |
| deepseek | infill           |      6.7b | 0.5809 |

\* both the files imported by current file and the files that import current file are added to the left context


# Getting started
Prerequisites:
* Linux
* Conda 

1. Install requirements in your main environment
```python
pip install -r requirements.txt
```

2. Download repositories and dataset
```bash
wget https://zenodo.org/records/13378983/files/realcode_v3_repos_upd.tar.gz
tar -xvf ../RealCode_eval/realcode_v3_repos_upd.tar.gz -C data
```
3. Build conda environments for each repository
```
cd prepare_data
python run.py
cd ..
```

4. Check installation
```bash
pytest tests
```

> [!NOTE]
> Number of passed tests in the repositories may vary depending on your system. If this test fails on your system feel free to open an issue. We need your feedback to create a more stable version of the benchmark.

5. Run the evaluation of your model (see **config/config.yaml** for details). E.g. for [codeparrot-small](https://huggingface.co/codeparrot/codeparrot-small) (Pass@1 should be 0.16):
```bash
CUDA_VISIBLE_DEVICES=0 python main.py +model=codeparrot generation_params.max_new_tokens=512 max_context_length=500
```

> [!WARNING]
> **Generated code is executed without any isolation! Use at your own risk!**

6. The evaluation results will be saved at ```./results/```

# Examples


* Run codellama-7b with left context only, 1024 tokens in prompt
```bash
CUDA_VISIBLE_DEVICES=0 python main.py +model=codellama size=7b max_context_length=1024
``` 
* Run starcoder-3b with left and right context, 512 tokens in left context, 512 tokens in right context
```bash
CUDA_VISIBLE_DEVICES=0 python main.py +model=starcoder size=3b generator_mode=infill max_context_length=1024
``` 
* Run starcoder-3b with left and right context, 750 tokens in left context, 250 tokens in right context
```bash
CUDA_VISIBLE_DEVICES=0 python main.py +model=starcoder size=3b max_context_length=1000 left_context_ratio=3
``` 
* Run model from local checkpoint at ```/downloaded/checkpoints/my_ckpt```
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    +model=local model_base_path=/downloaded/checkpoints model_short_name=my_ckpt max_context_length=1024
```
* Run repository level code infilling with deepseekcoder-1.3b:
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    +model=deepseek size=1.3b +context_parser=import_copy \
    generator_mode=infill max_context_length=14000 left_context_ratio=19
```
* Use hydra multirun for consecutive runs:
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    +model=codellama size=7b,13b generator_mode=lm,infill max_context_length=2048 --multirun
```

See ```config/config.yaml``` for other options

# Notes
* Around 60% of the used repositories are related to the field of AI/LLMs/ML. We did not perform any specific topic-based filtering, it comes from the topic distribution of the github python repositories in summer 2023
* The code in RealCode repositories was not be seen during pretraining of starcoder or codellama as these models were trained before the summer 2023. Deepseek-coder may have seen this code in pretraining.
* Repositories are rolled back to a specific commit during data preparation
* Not all the tests are passed in the repositories. We consider a generation to be correct if it passes the same number of tests, as the ground truth body of a function
* We use ```device_map='auto'```, if you wish to use specific GPUs set ```CUDA_VISIBLE_DEVICES```, as in the examples

If you find RealCode_eval useful please consider giving a star to the repositories used for evaluation: \
https://github.com/Jakob-98/openai-functools \
https://github.com/biobootloader/mentat \
https://github.com/causalens/cai-causal-graph \
https://github.com/modelscope/modelscope-agent \
https://github.com/simonmesmith/agentflow \
https://github.com/defog-ai/sql-eval \
https://github.com/Wyvern-AI/wyvern \
https://github.com/danielbeach/tinytimmy \
https://github.com/a-r-r-o-w/stablefused \
https://github.com/langchain-ai/permchain \
https://github.com/NullPyDev/beholder \
https://github.com/opencopilotdev/opencopilot \
https://github.com/AgentOps-AI/agentops \
https://github.com/TengHu/ActionWeaver \
https://github.com/fynnfluegge/doc-comments.ai \
https://github.com/Tinny-Robot/DimSense \
https://github.com/mljar/plotai \
https://github.com/juliendenize/eztorch \
https://github.com/yihong0618/epubhv \
https://github.com/simonw/llm-cluster \
https://github.com/Pennyw0rth/NetExec \
https://github.com/Vaultexe/server


