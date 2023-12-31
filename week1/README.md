## Introduction:
- In week 1 we'll go through the basic building block of transformer architecture.
    - We will only learn the things that will be useful in implementing the transformer in practice.
    - We will talk about self attention, multi-headed attention, Encoder, Decoder architecture and similar.
- We'll also talk about the Generative AI Project Life Cycle.
    - This sections helps to plan out, how to build your own Generative AI Project.
    - Generative AI project Life Cycles walks you through the individual stages and decisions you have to make when you're developing Generative AI applications.
- You need to be able to decide whether you are performing pertraining, Prompting, and fine-tuning.
- You will also able to make optimal decisions on choosing right LLMs based on their sizes (7B, 13B, 70B, and even greater)
    - Hint: You can use larger model if you want to generalize on different task. However you can use smaller model if you want to perform single tasks (like summarizing dialogue) or less.

- **Application of Generative AI**
    1. `Building a chatbot`
        - <img src='images/1.png' width='200'>
    2. `Generating images from text`
        - <img src='images/2.png' width='200'>
    3. `Using plugins to help you develop a code`
        - <img src='images/3.png' width='200'>

- Generative AI is a subset of traditional machine learning.

- **Different Large Language models**
    - Large language models have been trained in a trilions of words for many weeks and months with large number of computational resources.
    - These models will have billions of parameters.

    - <img src='images/4.png' width='400'>

    - In lab we will cover FLAN-T5 via prompting and fine-tuning.


- **LLM inference**
    - The text you pass to an LLM is known as prompt
    - The length of the longest sequence of tokens that a LLM can use to generate a token is known as context window of a LLM
    - The output of the model is known as completion.
    - The act of using the model to generate text is known as Inference.
    - <img src='images/5.png' width='400'>


## LLM use cases and track
- Chatbot
- Easy Writer
- Summarization
- Language Translation
- Translate Natural Language to Machine Code
- Named Entity Recognition
- Retrieval Augmented Generation (RAG)
    - Useful to retrieve data from outside a foundation model and augments your prompts by adding the relevant retrieved information.

- `Instructor mentions as the size of the LLM increases the understanding of the Language by model will also increases.`

- These use cases are possible only because of the architecture that powers that them i.e. `transformer`.


## Text Generation before Transfomers
- Before Transformer architecture that arises in 2017, RNNs were used for generating text.

- Generative algorithms have been around for some time, but previous models like recurrent neural networks (RNNs) had limitations due to compute and memory requirements for generative tasks.


## Transformer Architecture
>> You can look [Jay Alammar "The illustrated transformer"](https://jalammar.github.io/illustrated-transformer/) for interactive visualizations.

- <img src='images/6.png' width='400'>  

 - The transformer architecture revolutionized natural language tasks, outperforming earlier RNN-based models and enabling powerful generative capabilities.

 - The two main problems of RNNs solved by Transformer architecture are:
    - Long-term dependencies
    - Parallelize Computation 

 - The key feature of transformers is self-attention, which allows the model to understand the relevance and context of all words in a sentence by applying attention weights to their relationships.

 - The transformer consists of two parts: the encoder and the decoder, which work together and share similarities in processing.

 - Text must be tokenized to represent words as numbers before passing it through the embedding layer, where each token is represented as a vector in a high-dimensional embedding space.

 - Positional encoding is added to preserve word order information during processing.

 - The self-attention layer has multiple heads that independently learn different aspects of language, such as relationships between entities or the activities in a sentence.

 - The output of the self-attention layer goes through a feed-forward network, and the final softmax layer produces probability scores for each word in the vocabulary.

 - The word with the highest probability score is the most likely predicted token, but various methods can be used to select the final output token.


 ## Generating text with transformers
 >> In this section you will see how generate sequence given sequence using transformer architecture (similar to machine translation original objective of the transformer architecture designers).

 - In the original transformer paper, they experiment on machine translation task i.e. translating french text to english text.

- `Instructor used below figure to explain the concepts:`
    - <img src='images/10.png' width='500'>

 - `Transformers Encoders Flow:`
    - Tokenize input text using same tokenizers that was used to train the network (i.e. Byte Pair Encoding or BPE)
    - The Tokenize inputs are passed through the embedding layers, that will map tokenize input id to embedding vectors via embedding lookup table.
    - The sum of generated embedding vectors and position encoding vectors are then passed to the first multi-headed attention layers in the Encoders side.
    - The output of the mult-headed attention networks are the passed to the feed-forward network.
    - In encoders side there are total of 6 (multi-headed attention + feedforward) blocks. So, the embedding vectors will be passed to all of the 6 blocks one by one and final contextualized vectors for each input tokens are generated in the outputs of encoder blocks. 
    - This final output vector is the deep representation of the structure and meaning of the input sequence.
    - This representation is inserted into the middle of the decoder to influence the decoder's self-attention mechanisms.

- `Transformers Decoder flow`
    - At decoder side, a start of sequence token is added to the input of the decoder.
    - This triggers the decoder to predict the next token, which it does based on the contextual understanding that it's being provided from the encoder.

    - Like Encoder, In Decoder there are also 6 blocks of mult-headed self attention plus feed forward blocks.
        - <img src='images/9.png' width='300'>  

        >>> **Source:** [The illustrated Transformer Blogs](https://jalammar.github.io/illustrated-transformer/)

    - Unlike Encoder, One block named  Encoder-Decoder attention is added in between mult-headed self attention and feed forward block to consider encoded input sequence information while translating the text. 
        - <img src='images/8.png' width='300'>  

        >>> **Source:** [The illustrated Transformer Blogs](https://jalammar.github.io/illustrated-transformer/)

    - After passing through all these decoder layers, the output vectors from final decoders layer is passed through the softmax layers, and we extract the tokens with highest probability, and hence we got our first translated tokens. 

    - In the next iteration previous input tokens along with predicted translated tokens is passed through the input of decoders, and we extract the next translated predicted tokens in the same manner.

    - This process is repeated until End of Sequence tokens is predicted by decoder.


### Modified Transformer
>> For translation tasks as seen before, you used both Encoder and Decoder part of the original transformer without any modification. Based on the use cases, we can modify original architecture and use either Encoder only models, Both Encoder-Decoder models, or Decoder only models.

<img src='images/11.png' width='500'>

1. **Encoder only blocks**
    - Encoder models use only the encoder of a Transformer model.
    - These, models are useful for Natural language understanding tasks, i.e. understanding the input sequences.
    - They are often called as `auto-encoding models`.
    - The pre-training part is carried out by masking or corrupting given sequences (via `[MASK]`) and asking the model to find or reconstruct the original sentences.
    - We can add additional layers on the top of pretrained Encoder only models to perform different tasks such as:
        - sentence classification
        - named entity recognition
        - extractive question answering

    - Example models:
        - [ALBERT](https://huggingface.co/docs/transformers/model_doc/albert)
        - [BERT](https://huggingface.co/docs/transformers/model_doc/bert)
        - [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)
        - [ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra)
        - [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)

2. **Encoder-Decoder models**
    - Also called as `sequence-to-sequence models`.

    - Uses both Encoder and Decoder part of original Transformer architecture.

    - These architectures are useful for tasks where input needs to sequence and outputs also needs to be sequence such as:    
        - Summarization
        - Translation
        - Generative question answering

    - Example models
        - [BART](https://huggingface.co/docs/transformers/model_doc/bart)
        - [mBART](https://huggingface.co/docs/transformers/model_doc/mbart)
        - [Marian](https://huggingface.co/docs/transformers/model_doc/marian)
        - [T5](https://huggingface.co/docs/transformers/model_doc/t5)


3. **Decoder only models**
    - Uses only the Decoder of original Transformer architecture.
    - These models are called `auto-regressive models.`
        - Auto-regressive models is a techniques to predict the next word in a sequence of words based on the words that have come before it.

    - The pretraining tasks of Decoder only models is predicting the next word in the sentence.

    - Example models:
        - [GPT-4](https://paperswithcode.com/method/gpt-4)
        - [GPT-3](https://paperswithcode.com/method/gpt-3)
        - [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
        - [GPT-1](https://huggingface.co/docs/transformers/model_doc/openai-gpt)


_**At this point, I want you to encourage to take look at [Jay Alammar "The illustrated transformer"](https://jalammar.github.io/illustrated-transformer/) for interactive visualizations.**_


## Prompting and Prompt Engineering
- The text you fed into the model is called `prompt`.
- The act of generating text is called `inference`.
- The output text is know as `completion`.
- The length of the longest sequence of tokens that a LLM can use to generate a completion token is known as context window of a LLM.
- Prompt Engineering  is the practice of crafting effective queries or inputs referred to as prompts to guide AI to deliver the most accurate and useful answer.
- For the good answer, we generally add examples inside the prompt that helps LLMs to generalize based on the provided examples. Hence, providing examples inside the context window or prompts is called as `incontext-learning`

- Instructor talks about 3 different scenarios:
    1. `Zero Shot Inference`
    ```
    > Here no example is provided in the prompt
    Classify this review:
    I loved this movie!
    Sentiment:
    ```

    2. `One Shot Inference`
    > Here one example is provided in the prompt.
    ```
    Classify this review:
    I loved this movie!
    Sentiment: Positive

    Classify this review:
    I don't like this chair.
    Sentiment:
    ```

    3. `Few Shot Inference`
    > Here few examples are provided in the prompt
    ```
    Classify this review:
    I loved this movie!
    Sentiment: Positive

    Classify this review:
    I don't like this chair.
    Sentiment:  Negative

    Classify this review:
    Who would use this product?
    Sentiment: 
    ```

**`As per the Instructor, Largest models are good at zero-shot inference with no examples, where as Smaller models can benefit from one-shot or few-shot inference. You need to go for fine-tuning if your model is not performing well even with 5 or 6 examples.`**


## Generative Configuration (Inference parameters)
>> Here we'll see some of the configuration parameters that we can adjust to make the Large Language Models perform better. These configurations parameters are different from training parameters and contribute to make inference result better, allowing users to control various aspects of the text generation process.

_**A full list for available Generative Configuration parameters can be found [here](https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig)**_

- **`Max new tokens`**
    - Limits the number of tokens that the model will generate.
    - Example:
        - max_new_token = 100. We are asking model to generate max of 100 tokens. Here, it is not necessary that model will always generate 100 tokens, because if model generate stop token in advance before reaching 100 tokens, the generation process stops. In this case generated tokens can be less than 100.

- **`Greedy Decoding`**
    - In final layer of LLMs architecture, there is a softmax layer, which outputs the probability distribution of all the words present in the Vocabulary.
    - Most LLMs by default will operate with greedy decoding.
        - Simplest form of decoding where model always choose the word with highest probability.
        - Suitable for short generations.
        - can include repeated words in the generations.
    - `Example:`
        ```
        cake: 0.20
        donut: 0.10
        banana: 0.02
        apple: 0.01
        ....  ......

        In Greedy Decoding, the models output the words with highest probability scores i.e. cake in this case.
        ```

- **`Random Sampling`**
    - Introduces variability by randomly selecting words based on their probability distribution.
    - This allows for more natural and diverse text generation.
    - However, it can also lead to outputs that may not make sense or wander off into unrelated topics.
    - `Example: `
        ```
        cake: 0.20
        donut: 0.10
        banana: 0.02
        apple: 0.01
        ....  ......

        In Random Sampling, the models output the words by sampling based on the probability scores.
        - here, chances of occuring cake is 20%
        - chances of occuring donut is 10%
        - chances of occuring banana is 2%
        - chances of occuring apple is 1% and so on.

        So, in this case, model could outputs apple as well although it has less probability score.
        ```
         
- **`Sample top K`** 
    - With "Sample Top k", you specify a value k, and the model randomly samples from the top k tokens with the highest probabilities.
    - `Example: `
    ```
    cake: 0.20
    donut: 0.10
    banana: 0.02
    apple: 0.01
    ....  ......

    Let, K = 2

    Top K words: cake(0.20), and donut(0.10)

    Random Sample between cake(0.20), and donut(0.10) with,
    
        - 20% of chances of selecting cake
        - 10% of chances of selecting donut 
    ```

- **`Sample top P`**
    - With "Sample Top p", you specify a probability threshold p, and the model only considers tokens whose cumulative probability mass is less than or equal to p.
    - select an output using the random-weighted strategy with the top-ranked consecutive results by probability and with a cumulative probability <= p.

    - `Example: `
    ```
    cake: 0.20
    donut: 0.10
    banana: 0.02
    apple: 0.01
    ....  ......

    if p = 0.30

    Now, The options are words, cake (0.20) and donut (0.10) since (0.20 + 0.10 <= 0.30)

    Next, The model choose next words via random sampling between cake (0.20) and donut (0.10) i.e.
        - 20% of chances of selecting cake
        - 10% of chances of selecting donut

    ```

- **`Temperature`**
    - Range: 0 to positive inifinity.
    - You can visit [this site](https://lukesalamone.github.io/posts/what-is-temperature/) to play with the temperature.
    - The softmax function with temperature (θ) is defined as:  

    - $$
       \sigma(z_i) = \frac{e^{z_i / \theta}}{\sum_{j=1}^{N} e^{z_j / \theta}}
      $$

        - where,  
            - $\sigma(z_i)$ represents the logits for the i-th element in the input vector z.
            - N, is the total number of elements in the input vector z

    - `High Temperature (e.g. T > 1.0)`
        - Makes output more random and diverse.
        
    - `Medium Temperature (e.g. T close to 1.0)`
        - Balances randomness and determinism.

    - `Low Tempearture (e.g. T < 1.0)`
        - output more focused and deterministic.

    - `Cooler Temperature vs Higher Temperature`
        - <img src='images/12.png' width='400'>
        - When Temperature is low, we can see less variability distribution with single peak at word **cake** i.e. less randomness
        - When Temperature is high, we can see more variable distribution i.e. Broader, flatter probability distribution meaning more randomness.

_**`Higher the temperature, Higher the randomness. Lower the temperature, Lower the randomness.`**_


## Generative AI project lifecycle
> In this section Instructor highlights the overall Generative AI project lifecycle starting from Project Scoping to Project Integration.

<img src='images/13.png' width='500'>

- **Scope(Define the use cases)**
    - LLMs can do a variety of tasks.
    - LLMs abilities depends on the size and architecture of the model. 
    - Big models are good at performing variety of tasks as they have seen more data and trained on big architecture with large compute resources. However small models are good at performing small or subset of tasks.
    - You need to decide what LLMs can be used in your applications use cases
        - Choose Large models to Perform many tasks i.e Essay Writing, Summarization, Translation, Information retrieval (NER), Invoke APIs and actions.
        - Choose Small models to perform single tasks i.e. Information retrieval (NER).   
    

- **Select LLM**
    - LLM can be used as: 
        - `Inference` via existing model
        - `FineTune` an existing model
        - `Pretrain` your own

    - In general you can start via model inference and test it via prompting, then you can move towards the steps of FineTuning on custom data, or even Pretraining your own LLMs from scratch.

    - Choice of LLMs depends on 
        - The variety of task you wish to perform. 
        - Available compute resources you have.
        - Licence (commercial, research, non-comercial)
        - [LLM LeaderBoard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)



- **Adapt and align model**
    - Once you have choosed your model, Next step is to assess its performance and carry out additional training if needed for your application.
    - Instructor suggests to start from Prompt engineering via in-context learning, switch to Finetuning if in-context learning doesn't works, finally you can shift towards additional fine-tuning technique called reinforcement learning with human feedback.
    - In order to evaluate the models (How well your model is performing?) you can rely on different metrics based on the specific tasks (Example: ROUGE score for summarization.)

 _**`Adapting and Evaluating process is Highly Iterative i.e. Start with sample prompt evaluate the result. Based on the evaluation result go back to prompt-tuning or even fine-tuning and again evaluate the model unless you reach to certain evaluation threshold.`**_

 - **Application Integration**
    - Two ways:
        1. Optimize and deploy model for inference
            - Directly host or deploy your model into your infrastructure for inference.
            - Here, you need to optimize your model via techniques such as quantization, so that you can make best use of your compute resources with best user experiences. 

        2. Augment model and build LLM-powered applications


## Introduction to AWS labs
> This course encourages hands-on learning via several lab exercises to solidify concepts. In first week you will do the `dialogue summarization task` using generative AI.  You will explore how the input text affects the output of the model, and perform prompt engineering to direct it towards the task you need.

- The lab environment called `Vocareum` provides access to Amazon SageMaker through an AWS account at no cost to the learners.
    - [Vocareum](https://www.vocareum.com/) is an online platform that provides cloud-based learning environments students and learners. It is commonly used in educational settings like online courses and workshops, to provide hands-on learning experiences without the need for learners to set up their own local environments. 

- Learners can access the labs in Vocareum, launch the AWS console, and open SageMaker Studio, a jupyter-based IDE for running notebooks.

- Learners are guided step-by-step on how to access the labs, open the terminal, and copy the necessary code from a public S3 bucket.

- Learners are asked to perform Dailogue Summarization via FLAN-T5 LLM from Hugging Face. 
    1. Dialogue Summarization without prompt engineering.
    2. Dialogue Summarization with prompt engineering.

- LLM: [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)
- Tokenizer: [Tokenizer parameter details](https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/auto#transformers.AutoTokenizer)
- Dataset: [knkarthick/dialogsum](https://huggingface.co/datasets/knkarthick/dialogsum)


_**`Note: If you are auditing the course, then you will not be able to access the lab environments.`**_


## Lab Assignments
```python
print("hello")

```


## LLM Pre-training Large Language Models
> Here, Instructor highlights about different LLMs model architecture along with their pretraining objectives.

- Once you define your project scope, Next step is to select the Large Language Model, For this you may have 2 options i.e. to work with existing foundation model via prompting or train your own from scratch.
- It is recommended to start with foundation model and the move towards the training part.
- AI community named `Hugging Face` offers numerous [open source models](https://huggingface.co/models) with model cards, containing vital information about the best use cases, training details, and known limitation of each model. 

<img src='images/14.png' width='400'>

- As per the instructor, The initial training process for LLMs is referred to as pre-training. 
- In a pre-training phase, models learns from vast amounts (gigabytes, terabytes, petabytes) of unstructured textual data.
- This vast amount of data is ofter created from many sources such as scraping off the internet and corpora of texts that have been collected specifically for training large language models.
- This pre-training steps doesn't require a labelled data, since the training process is carried out in a `self-supervised` manner i.e. predicting next token based on the previous token.
- Pretraining helps the model with a deep statistical representation of language, enabling it to understand and capture complex patterns and structures present in the text.
- Pretraining requires large amount of computational power i.e. GPUs.
- You need to ensure quality of scrapped data before feeding it to the model during training i.e. address bias, and remove other harmful content.

- **Comparative Study of Different Model Architecture**

| **Model Type**       | **Pre-training Objective**                | **Context Handling**           | **Use Cases**                                          | **Examples**                                          |
|---------------------|----------------------------------------|-----------------------------|-------------------------------------------------------|------------------------------------------------------|
| Autoencoding Models | Masked Language Modeling               | Bi-directional               | Sentence Classification, Named Entity Recognition, Word Classification | BERT, RoBERTa                                          |
| Autoregressive Models | Causal Language Modeling               | Uni-directional             | Text Generation, Zero-shot Inference                    | GPT, GPT-2, GPT-3                                      |
| Sequence-to-Sequence Models | Span Corruption and Reconstruction    | Bi-directional (Encoder)    | Translation, Summarization, Question-Answering         | T5 (Text-to-Text Transfer Transformer), BART         |

- `Pre-training Objective`: 
    - Highlights primary training objective for each model during the pre-training phase.
- `Context Handling:`
    - Autoencoding models capture bi-directional context, allowing them to understand the full context of a token. Given a word at position <t> it considers all the past words and future words in a sequence.
    - In contrast, [autoregressive models](https://en.wikipedia.org/wiki/Autoregressive_model) have uni-directional context i.e. words from the past, since it is predicting future words.
    - Sequence to Sequence models use both encode and decoder part, giving them bidirectional context during training.

- <img src='images/15.png' width='500'>
    
_**`As per the researcher, Larger the model size the wide range of task it can performed with no or less in-context learning.`**_


## Computational Challenges of training LLMs
- One of the major issues when you try to train LLMs is `OutOfMemoryError: CUDA out of memory`.
- This error may occurs when trying to train your models or just loading your models on Nvidia GPUs.
- CUDA stands for Compute Unified Device Architecture. It is collection of libraries and tools developed for Nvidias GPU.
- Libraries such as PyTorch and Tensorflow use CUDA to boost performance on matrix multiplication and other operations common to deep learning.
- Out of Memory issues arises because of huge size of LLMs and require tons of memory to store and train all of their parameters.
- **Calculation of Approximate GPU RAM needed to store 1B parameters**

```
- 1 parameter = 4 bytes (32-bit float) --> way computer used to represent real numbers  

- 1B parameters = 4 * 10**9 bytes = 4GB (GPU RAM)

```

- **Calculation of Approximate GPU RAM needed to train LLMs**
```
Model Parameters (Weights) -->  4 bytes per parameters

Adam Optimizers (2 states)  --> +8 bytes per parameters

Gradients                   --> +4 bytes per parameters

Activations and             --> +8 bytes per parameters (high-end estimates)
temp memory (variable size)

TOTAL                ---------> = 4 bytes per parameter + 20 extra bytes per parameters (APPROX 80GB)
```

_**`80GB is the memory capacity of a single Nvidia 100 GPU, popular for machine leraning tasks in the cloud.`**`_


- **Quantization:**
    - It is the technique used to reduce the memory.
    - `IDEA:` _Reduce the memory required to store the weights of your model by reducing their precision from 32-bit floating point numbers to 16-bit floating point numbers or 8-bit integer numbers_
    - This technique reduces model accuracy slightly but it is acceptable since it reduces the model size tremendously  with minimal performance loss.

    - `FP32 and FP16`
        - FP32: full precision 32 bit. By default computer system use this to represent numbers.
        - FP16: full precision 16 bit
        

        - <img src='images/16.png' width='450'>

    - `BFLOAT16`
        - stands for Brain Floating Point Format.
        - Developed by Google Brain.
        - Captures the dynamic range of FP32 using only 16 bits.
        - It maintains the full exponent range of FP32 but truncates the fraction to save memory.
        - Improves training stability and can enhance model performance, especially on newer GPUs like NVIDIA's A100.
        - It strikes a balance between FP16 and FP32, offering memory savings while still retaining good performance.
        - `Cons: ` Not suited for Integer Calculations (which is rare in deep learning).
        - <img src='images/17.png' width='450'>

    - `INT8`
        - Quantization of 32 bit floating point to 8 bit integer.
        - This quantization type represents model parameters using 8-bit integer numbers.
        - Since Integers have a smaller range compared to floating point numbers, INT8 quantization can result in significant memory savings.
            ```
            Example: Represent 3.141592 using INT8

            1. Convert 3.141592 to an integer i.e. 3 (truncate the decimal part)
            2. Represent 3 as an 8-bit integer:
                - In binary: 00000011 (3 in binary)

            So, in 8-bit representation, the value closest to 3.141592 is 00000011.

            - This leads to significant data loss, however results in memory savings.
            ```
- **Comparitive Study**

    | Quantization Type | Bits | Exponent Bits | Fraction Bits | Memory (Bytes) |
    |-------------------|------|---------------|---------------|----------------|
    | FP32              | 32   | 8             | 23            | 4              |
    | FP16 (Half)       | 16   | 5             | 10            | 2              |
    | BFLOAT16          | 16   | 8             | 7             | 2              |
    | INT8              | 8    | -             | 7             | 1              |


- **[Quantization] Approximate GPU RAM needed to store 1B parameters**
```
Given same Model size (1B parameters),

1. Full Precision Model: 4GB @ 32-bit full precision

2. 16-bit quantized Model: 2GB @ 16-bit half precision

3. 8-bit quantized Model: 1GB @ 8-bit precision
```

- **[Quantization] Approximate GPU RAM needed to train 1B parameters**

```
Given Same model size (1B parameters),

1. 80GB @ 32-bit full precision

2. 40GB @ 16-bit half precision

3. 20GB @ 8-bit precision

Note: 80GB is the maximum memory for the Nvidia A100 GPU, so to keep the model on a single GPU, you need to use 16-bit or 8-bit quantization.
```
-  **[No Quantization] Approximate GPU RAM needed to train Larger Models**

```
- 1B param model: 4G @ 32-bit precision

- 175B param model: 14,000GB @ 32-bit full precision

- 500B param model: 40,000GB @ 32-bit full precision


Note: Increase in model sizes, increases GPU RAM needed for training. So, from single GPU you need to shift towards distributed training using 100s of GPUs.
```
- **summary**
    - Quantization technique reduces required memory to store and train models.
    - Quantization projects original 32-bit floating point numbers into lower precision spaces like FP16, INT8.
    - Modern Deep Learning Framework supports `Quantization-aware training (QAT)` which learns the quantization scaling factors during training process.
    - BFloat16 is popular choice in deep learning due to its ability to maintain dynamic range of FP32 but reduces the memory footprint by half.
    - Many LLMs like `FLAN T5` have been pre-trained with BFLOAT16


## Efficient Multi-GPU Compute Strategies
> This section covers the general idea process of scaling model training efforts beyond single GPU.The focus is on efficiently distributing computation across multiple GPUs, even for small models, and addressing the challenges of memory constraints when dealing with larger models.

- May need to scale your model training efforts beyond single GPU.
- It is recommended to fit your model to multiple GPUs to speed up the training even if  its fit on a single GPUs.
- **Case: When model fits on a single GPU**
    - Distribute large datasets across multiple GPUs and process these batches of data in parallel.
    - A popular implementation of these model replication technique is Pytorch Distributed Data Parallel.
    - Distributed Data Parallel (DDP) copies model in each GPUs and send batches of data in each of the GPUs in parallel.
    - Each data is processed parallely, syncronization step combines the results of each GPU.
    - <img src='images/18.png' width='500'>
    - `Limitation:`
        - Need to keep full model copy and training parameters on each GPU.

- **Case2: When model doesn't fit into a single GPU**
    - You can use technique called `Model Sharding`
    - Sharding is a technique used to distribute and split data or components across multiple devices or nodes for efficient processing.
    - Popular implementation of model sharding is Pytorch `Fully Sharded Data Parallel (FSDP)`
    - FSDP is motivated by the `ZERO --> Zero data overlap between GPUs`.
    - Goal of Zero is to optimize memory by distributing or sharding model parameters, gradients, and optimizer states across GPUs with ZeRO data overlap.
    - Suitable when model doesn't fit onto a single GPU.
    - `ZeRO`
        >> Optimize memory by distributing (sharding) the model parameters, gradients, and optimizer states across GPUs.
        1. `Stage 1:`
            - shards or distributes only optimizer states, reducing memory by upto 4x.

        2. `Stage 2:`
            - shards optimizer plus gradients accross GPUs, reducing memory by up to 8x when combined with Stage 1.

        3. `Stage 3:`
            - Shards all components (optimizers + gradients + parameters) across GPUs.
            - Memory reduction is linear with a number of GPUs.

    - Unlike DDP, In FSDP, you distribute data accross multiple GPUs along the with the model parameters, gradients, optimizer states. This is achieved using strategies specified in [Zero](https://paperswithcode.com/method/zero)
    - <img src='images/20.png' width='450'>

    - **Notes:**
        - Helps to reduce overall GPU memory utilization.
        - Configure level of sharding via `sharding factor`  
            1. Full Replication (no sharding)
                - sharding_factor = 1 GPU
            2. Full Sharding
                - sharding_factor = max number of  available GPUs
                - Most memory savings but increase communication overhead required for synchronization between different GPUs.
            3. Hybrid Sharding
                - sharding_factor = in between 1 GPU and available GPUs.


## Scaling laws and compute-optimal models

- Pretraining Goals: maximize model performance i.e. minimizing loss when predicting tokens.

- Two choices to achieve pretraining goals:
    - Increase Dataset size
    - Increase number of parameters in the model

- In performing above tasks, we have constraint on compute budget (i.e. GPUs, training time)

- **Compute Budget for training LLMs**
    - `petaflop/s-day:`
        - number of floating point performed at rate of 1 petaFLOP per second for one day
        - 1petaFLOP/s = 1,000,000,000,000,000 (one quadrillion) floating point operations per second.
        - Floating point operations (FLOPs) are fundamental arithmetic operations (like addition, subtraction, multiplication, division) performed on floating-point numbers in a computer.
        - 1 petaflop/s for day is equivalent 8 NVIDIA V100 GPUs operating at full efficiency for one full day.
        - Also, 2 Nvidia A100 GPUs are equivalent to 8 Nvidia V100 GPUs (since A100 have a more powerful processor that can carry out more operations at once).
        - `petaflop/s per day:` per second day adds the element of time. It means that the petaFLOP value is being sustained over the course of a full day (24 hours)
    - `Number of petaflop/s-days to pretrain varous LLMs`
        - <img src='images/21.png' width='400'>
        - Above chart shows a comparison of the peta flops per second days required to pre-train different variant of BERT and ROBERTA (encoder only), T5 (encoder-decoder), and GPT-3 (decoder only).
        - Here,
            - T5 XL with 3B parameters required close to 100 petaFLOP per second days.
            - Large GPT-3 175B parameters model required approx 3,700 petaFLOP per second days.

    - `ChinChilla Paper`
        - In the paper [ChinChilla paper](https://arxiv.org/abs/2203.15556) published in 2022 a group of researchers carried out a detailed study of the performance of large language models of various sizes and quantities of training data. The goal was to find the optimal number of parameters and volume of training data for a given customer budget.
        - Chinchilla paper, hints that many of the 100 billion parameter LLMs like GPT-3 may actually be over parametrized (meaning they have more parameters than they need to achieve good understanding of language) and under-trained.
        - Chinchilla authors hypothesized that smaller models may be able to achieve the same performance as much large ones if they are trained on larger datasets.
        - `Compute-Optimal vs Non Compute-Optimal Model Comparison`
            - A compute optimal model is one that is designed and trained to achieve the best possible performance considering the available computational resources, such as GPU or other hardware, within a given compute budget.
            - A non compute-optimal model might not be designed or trained with careful consideration of the available computational resources i.e. it could be larger or more complex than necessary given the compute budget. 
            - <img src='images/22.png' width='450'>
            - Instructor highlights key takeways from Chinchilla paper i.e. 
                - _**`The optimal training dataset size for a given model is about 20 times larger than the number of parameters in the model.`**_
                - _**`Llama was trained on dataset that is nearly close to chichilla recommended number i.e. 20 times larger than number of parameters.`**_
                - _**`Compute optimal chichilla model outperforms non-compute models such as GPT-3 on a large range of downstream tasks.`**_
            - This means, for a 70B parameter model, the ideal training dataset contains 1.4 trillion tokens or 20 times the number of parameters.
            - From the above table, we can see last 3 models were trained on datasets that were smaller than the chinchilla optimal size (These larger models may be under-trained due to lower training size). 
                - You can see compute-optimal # of tokens columns to see recommended numbers of training data size.
    - `Future Trends in Model Development considering Chinchilla paper findings`
        - Researchers and developers are shifting towards optimizing model design rather than just increasing model size.
        - <img src='images/23.png' width='450'>
        - As seen BloombergGPT with model size 50B was trained following exactly the recommendation from Chinchilla paper.



## Additional Readings:
>> Includes additional resources suggested by instructor.

1. **Transformer Architecture:**
    - [Attention is all you need](https://arxiv.org/pdf/1706.03762)
    - [BLOOM: BigScience 176B Model](https://arxiv.org/abs/2211.05100)
    - [Vector Space Models](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/home/week/3)


2. **Pre-training and Scaling Laws:**
    - [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

3. **Model architectures and pre-training objectives:**
    - [What Language Model Archiectures and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/pdf/2204.05832.pdf)
    - [HuggingFace Tasks](https://huggingface.co/tasks)
    - [HuggingFace Model Hub](https://huggingface.co/models)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971.pdf) 
 
4. **Scaling laws and compute-optimal models**
    - [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
    - [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)
    - [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/pdf/2303.17564.pdf)




## References
- https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
- https://github.com/google-research/FLAN/tree/main/flan/v2
- https://huggingface.co/docs/transformers/model_doc/flan-t5
- https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/auto#transformers.AutoTokenizer
- https://www.vocareum.com/
- https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/text_generation#transformers.GenerationConfig
