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


## Fine-tuning on a single task
- Fine-tuning on a single task is useful when there may arise situation when you to perform a single task like summarization, question answering, etc i.e. only the interested tasks.
- Generally, 500 to 1000 training examples can results into somewhat good model, However larger training samples leads to the optimal results.

- Author mentioned about the term `Catastrophic forgetting` which means the full-fine tuning process modifies the weights of the original llm. 
    - Leads to great performance on Fine Tuned Task (single task)
    - Degrades performance on other task (multiple task)

- For better generalization capability of model on multiple task, you can fine-tune on multiple tasks at the same time.
    - Good, multi-task fine-tuning may require 50,000 to 1,00,000 accross many tasks.
    - i.e. more data and more computation power to train

- Author also discussed about `PEFT (Parameter Efficient Fine-tuning)` instead of Full-Fine Tuning
    - PEFT preserves the original weight of the LLMs and train only a small number of task-specific adapter layers and parameters
    - Greater robust to catastrophic forgetting since most of the pre-trained weights are left unchanged.



## Multi-task instruction fine-tuning
> Extension of single task fune-tuning where training examples consisits example inputs and outputs for multiple tasks

- Dataset can consists of variety of tasks such as `summarization`, `Translation`, `NER`, etc.

- Generally the model is fine-tuned on these types of mixed datasets in order to improve the performance of the model on all of the mentioned tasks.

- **Cons:**
    - It requires a lot of data for fine-tuning, you may need around 50000 to 100000 or even greater. (As per instructor it is worth of gathering such large data, since final fine-tuned model will be of good performance)

- `FLAN` family of models are trained using mult-task instruction fine-tuning  dataset.
    - FLAN models refer to a specific set of instructions used to perform instruction fine-tuning.
    - FLAN stands for `Fine-Tuned Language Net`
    - Example: 
        - FLAN-T5 is T5 version fine-tuned on FLAN instruction set.
        - FLAN-PALN is the FLAN instruct version of the palm foundational model.

- As Discussed, FLAN-T5 is the Fine-tuned version of base T5 which is fine-tuned accross 473 datasets accross 146 task categories.
- <img src='images/2.png' width='500'>

- Instructor highlighted `samsum` datasets that is used to train model to improve dialogue summarization capability. You can look below for example datset:


| Dialogue (string)                            | Summary (string)                                    |
|----------------------------------------------|------------------------------------------------------|
| "Amanda: I baked cookies. Do you want some ? Jerry: Sure! Amanda: I'll bring you some tomorrow :-)" | "Amanda baked cookies and will bring Jerry some tomorrow." |

- **Samsum Data Link:** [Click Here](https://huggingface.co/datasets/samsum)

- Samsum is also the part of FLAN instruction set.

- `Sample FLAN-T5 prompt templates`
    - <img src='images/3.png' width='500>
    - As we can see, instruction to ask LLM to summarize are slightly different in different simples, this helps LLM to generalize better.

- To further improve T5's summarization capabilities it can be further fine-tuned with a domain specific instruction dataset like [dialogsum dataset](https://huggingface.co/datasets/knkarthick/dialogsum/viewer/default/train?row=0)


- Instructor mentioned, Summary before fine-tuning FLAN-T5 with our dataset is not that good as compared to fine-tuning FLAN-T5 with dialogsum datset that matches summarization similarly to that of human generated.

_`How to evaluate how well our model performed?`_