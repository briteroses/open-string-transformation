# String Transformations Toolkit

This is a research codebase for redteaming of large language models, specifically focusing on encodings and obfuscations of text input and output. `string-transformation` provides an API for constructing arbitrary compositions of string obfuscation methods, which is useful for manual and automated redteaming and can be composed with other jailbreak methods.

## Getting Started
An extensive catalog of string obfuscations can be found in `string_transformations/string_transformations.py`; these include obfuscations discussed throughout the redteaming literature, including rotary ciphers, leetspeak, Morse code, Base64, ASCII, low-resource language translation, reversal, style injection, and more. Jailbreaking experiments based on the string transformation API can be found in `experiments/composition.py` and `experiments/adaptive_attack.py`.

The `models/` folder catalogues and provides an API for language model inference. The `judging/` folder provides an API for the LLM-as-a-judge in safety evaluations, and implements a custom LlamaGuard setup as well as the HarmBench setup.[[1]](#1)

## Usage
We use the following lightweight pattern for running an experiment such as `Composition`:

**main.py**
```python
def run(config):
    target_model = GPT4o()
    num_attack_trials = 20

    hyperparameter_grid = {
        "k_num_transforms": [1, 2],
        "maybe_transformation_instructions": [True],
        "other_transform": [Id, Reversal, Leetspeak, AlternatingCase, MorseCode, Binary, BaseN],
        "composition_target": ["query", "response"],
    }
    response_composition_experiment = CompositionExperiment(
        target_model=target_model,
        num_attack_trials=num_attack_trials,
        hyperparameter_grid=hyperparameter_grid,
    )
    response_composition_experiment.run_hyperparameter_grid(config)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run composition experiments with optional evaluation mode.")
    parser.add_argument("--val_or_eval_or_llamaguard", type=str, default="val", help="Specify the evaluation mode. 'val' = harmbench val classifier. 'eval' = harmbench official classifier. 'llamaguard' = Haize-Mistral with LlamaGuard prompt.")
    parser.add_argument("--evaluate_during", type=bool, default=False, help="Evaluate each hyperparameter setting as part of its run?")
    return parser.parse_args()


@app.local_entrypoint()
def main():
    config = vars(parse_arguments())
    cheap_modal_wrap_experiment.remote(run, config)

if __name__ == "__main__":
    config = vars(parse_arguments())
    run(config)
```

The script will run locally via `python main.py`, or, if you have a Modal account, the script can be run on a cheap remote CPU with `modal run -d main.py`. (If running on Modal, first change the Modal configuration to your liking at `utils/modal_utils.py`.)

It's also easy to use string transformations standalone. See the following example:
```python
from string_transformations.string_transformations import CaesarCipher, LatexMode, Leetspeak, Reversal

query = "What is the meaning of life?"
model = GPT4o()
greedy_one_sample = get_greedy_one_command(model)
greedy_more_tokens = {get_max_tokens_key(model): 512}

output = model.inference(query, **greedy_more_tokens)[0]
print(output)

composition = [Reversal.construct(), CaesarCipher.construct(), Leetspeak.construct(), LatexMode.construct(model.__class__)]

for transform in composition:
    output = transform(output)
    print(output)

for transform in reversed(composition):
    output = transform.invert(output)
    print(output)
```

## References
<a id="1">[1]</a>
Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., Sakhaee, E., Li, N., Basart, S., Li, B., Forsyth, D., & Hendrycks, D. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. arXiv preprint arXiv:2402.04249.
