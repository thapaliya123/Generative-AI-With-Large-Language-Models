## Introduction
>> This weeks covers Fine-tuning and evaluation of large language models
- Explain how fine-tuning and instruction prompt dataset can improve performance on one or more tasks.
- Define catastrophic forgetting and explain techniques that can be used to overcome it.
- Define the term Parameter-efficient Fine Tuning (PEFT).
- Explain how PEFT decreases computational costs and overcomes catastrophic forgetting.

##  `Instruction Fine-Tuning:`

- When base large language model is created using lots of internet data, the trained model will know lots of information about text. But such Base model doesn't know how to responds to a certain questions based on prompt. Hence, we will Fine-tune the base model with Instruction related datasets that can understand the prompt passed by the user and responds accordingly.

- Instruction Fine-Tuning is useful because:
    - In-context learning may not work for smaller models.

    - Including examples via few-shot learning may leads to increase in context size of many LLMs.

    - when large language model can't answered to user prompt with optimal responds via in-context learning (one/few shot examples prompt).

- Instruction Fine-Tuning is a supervised learning i.e. given input prompts we will instruction fine-tune LLMs to output completion of the input prompts.

- `Full fine-tuning` updates all of the model parameters.
    -  Similar to Pre-training, Full fine-tuning may need enough memory and budget for training, since we are training all of the layers.


- **Instruction Fine-Tuning Steps:**
    1. Prepare the training datasets
        - several publicly available dataset like [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
        - 
    2. Training Splits 
        - Divide the datasets into training, validation, and test splits
    
    3. Modeling
        - Fine-Tune pre-trained LLMs with training split.
        - Validate and perform hyperparameter tuning with Validation split.
        - [Crossentropy Loss](https://wandb.ai/sauravmaheshkar/cross-entropy/reports/What-Is-Cross-Entropy-Loss-A-Tutorial-With-Code--VmlldzoxMDA5NTMx) is used as loss or objective function since `LLMs` outputs a probability distribution accross tokens from the Vocabulary.
        - Now, updates the model weights via Backpropagating Crossentropy loss function.
        - Finally, test instruction fine-tuned LLMs with test split, Compare model prediction and true output.

- Fine-Tuning original base model leads to newer version of model known as `instruct model`

- <img src='images/1.png' width='500'>