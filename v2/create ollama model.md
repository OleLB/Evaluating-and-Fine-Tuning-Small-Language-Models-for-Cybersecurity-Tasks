
After the fine-tuning process there should be a folder `*-lora-adapter`. Locate the folder and put it in a new parent folder

```
./mistral-model
	|___ ./mistral-lora-adapter
```

## Create a Modelfile

The Modelfile should be placed in the parent folder

```
./mistral-model
	|___ ./mistral-lora-adapter
	|___ ./Modelfile
```

Its content should look like:

```
FROM mistral-nemo:latest

ADAPTER "./mistral-lora-adapter"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

This is a recipe for how ollama should construct the model

## Rename file

rename "adapter_model.safetensors" to the format "model*.safetensors"

## Run ollama command

In a terminal, navigate to the parent folder, and run the command:
```
*/mistral-model> ollama create <new_model_name>:latest -f Modelfile
```

replace "<new_model_name>" with the desired name for your new ollama model

You should see something like:

```
gathering model components
copying file sha256:306cc1ce6bcdd45bff7436d9f832480c50c43653904e2edce70f8f3f90e68441 100%
copying file sha256:17c0ded2634b9068546e8292e08c8ccd1d0c9b8fe302a85e9b863a07ab411d2a 100%
copying file sha256:25c7c70e96fa52bc892a82e510266160947bda77d10f33ede625c46eed1d5225 100%
copying file sha256:1faccfa21dda875b3d657e7ed5f4d0fbbca3a40fe544b23a01bfdbc162ae3c55 100%
converting adapter
using existing layer sha256:b559938ab7a0392fc9ea9675b82280f2a15669ec3e0e0fc491c9cb0a7681cf94
using existing layer sha256:438402ddac7513a63010a594fbd69a11d74b7843ee686e9dc4589c01241c075a
using existing layer sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1
creating new layer sha256:66703f4c83afd8038f942671221a383d66f135f74acd606e39b15d5c16416da5
creating new layer sha256:0388ec7a89240f56b80189d70577da1446230ce5805cd5d5ef3e1772fad7adf6
writing manifest
success
```

The new model should now be available in ollama



