---
name: apply-mm-torch-compile
description: Compiles multimodal encoders in vLLM using @support_torch_compile

---

# apply-mm-torch-compile

## Description
Interactive walkthrough for compiling multimodal encoders in vLLM using @support_torch_compile

## Instructions for Claude
When this skill is invoked, guide the user through each step interactively:
1. Ask which multimodal model they want to compile
2. Find the model's encoder class in the codebase
3. Walk through each step, confirming completion before moving on
4. Offer to make code changes when appropriate
5. Help troubleshoot any errors that arise

## Walkthrough Steps

### Step 1: Setup
Ask user to confirm:
- User has forked the vLLM repo
- Which model they're working on

### Step 2: Test Baseline Compilation
Run the model to verify it works before making changes:
```bash
python examples/offline_inference/vision_language.py -m <MODEL_SHORT_NAME>
```
or 
```bash
python examples/offline_inference/audio_language.py -m <MODEL_SHORT_NAME>
```

Then, patch the engine init:
```python
compilation_config={"compile_mm_encoder": "true"},
enforce_eager=False,  # Required for torch.compile
```

Verify the model runs correctly before proceeding.

### Step 3: Apply Direct Compile Call
Find the model's encoder/vision module in the codebase and add:
```python
self.vision_tower.compile(fullgraph=True)
```

Re-run Step 2 to verify compilation works.

> **CRITICAL:** The module name varies by model type:
> - Vision models: `self.vision_tower`
> - Audio models: `self.audio_tower`
> - Other components: `*MultimodalEmbedder`, etc.
>
> Start with 1 module and expand if successful. Work with the user to identify the correct module.

At this point, you may run into graph breaks or general compilation errors. Reference the [PyTorch Compile Programming Guide](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html) to resolve many graph breaks.

### Step 4: Apply @support_torch_compile Decorator

Now remove the torch.compile fromm step 3. and instead use 
the decorator to the module class with proper gating using
`enable_if` and `should_torch_compile_mm_vit`:

```python
from vllm.model_executor.models.vision import should_torch_compile_mm_vit

@support_torch_compile(
    dynamic_arg_dims={"x": 0},
    enable_if=should_torch_compile_mm_vit,
)
class YourEncoderModule(nn.Module):
    ...
```

**Common issues and fixes:**

**Critical** You should proactively try and apply these fixes as they are required for compilation

| Error | Cause | Fix |
|-------|-------|-----|
| `Expected tensors only, but got: <class 'int'>` | Cache load errors | Wrap init site with `set_model_tag` context manager |
| `__init__() takes 1 positional argument but 3 were given` | Positional args in init | Convert positional args to kwargs where module is instantiated |
| `AssertionError: Forward context is not set.` | Missing vllm_config | Wrap module's forward call site with `forward_context` |
| Shape specialization errors | Dynamic batch dimension | Mark leading dimensions as dynamic; use `mark_unbacked` when size is 1 |

Ask user to paste any error messages for help debugging.

> **NOTE:** If compilation issues persist, clearing the cache sometimes helps:
> ```bash
> rm -rf ~/.cache/torch_inductor/ ~/.cache/torch/ ~/.cache/vllm/
> ```

### Step 5: Benchmarking

**Metrics to measure:**
- **Compile time** - Reported in logs when running `vllm serve`. Use the times for the mm encoder (compiled first), not the text decoder.
- **Inference throughput** (tokens/sec) - Reported in a table after executing the bench command.

**Start the server:**
```bash
vllm serve <model_long_name> --compilation-config='{"compile_mm_encoder":"true"}'
```

**Run benchmarks:**

For vision models:
```bash
vllm bench serve \
  --backend openai-chat \
  --model <model_long_name> \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --num-prompts 1000
```

For audio models:
```bash
vllm bench serve \
  --backend openai-audio \
  --model <model_long_name> \
  --endpoint /v1/audio/transcriptions \
  --dataset-name hf \
  --dataset-path openslr/librispeech_asr \
  --hf-split test.clean \
  --num-prompts 1000
```

**Compare results** with and without torch.compile by setting `"compile_mm_encoder":"false"`.

## References
- [vLLM Torch Compile Multimodal Guide](https://docs.vllm.ai/en/latest/design/torch_compile_multimodal/#compilationconfig)

## Example Models (Reference Implementations)

### Qwen2.5-VL (Best Example)
**File:** `vllm/model_executor/models/qwen2_5_vl.py`

Shows the complete pattern for multimodal encoder compilation:

```python
# Import the helper function
from vllm.model_executor.models.vision import should_torch_compile_mm_vit

# Apply to vision encoder blocks
@support_torch_compile(
    dynamic_arg_dims={
        "x": 0,
        "cu_seqlens": 0,
        "rotary_pos_emb_cos": 0,
        "rotary_pos_emb_sin": 0,
    },
    enable_if=should_torch_compile_mm_vit,
)
class Qwen2_5_VisionBlock(nn.Module):
    ...

# Also applied to patch embedding and merger
@support_torch_compile(
    dynamic_arg_dims={"x": 0},
    enable_if=should_torch_compile_mm_vit,
)
class Qwen2_5_VisionPatchEmbed(nn.Module):
    ...

@support_torch_compile(
    dynamic_arg_dims={"x": 0},
    enable_if=should_torch_compile_mm_vit,
)
class Qwen2_5_VisionPatchMerger(nn.Module):
    ...
```

### Key Helper Function
**File:** `vllm/model_executor/models/vision.py:146`

```python
def should_torch_compile_mm_vit(vllm_config: VllmConfig) -> bool:
    """Callable to be passed to `@support_torch_compile`'s `enable_if` argument."""
    return vllm_config.compilation_config.compile_mm_encoder
```

### Other Multimodal Models with torch.compile
- `vllm/model_executor/models/qwen2_5_vl.py` - Qwen2.5-VL
- `vllm/model_executor/models/mllama4.py` - LLaMa4
