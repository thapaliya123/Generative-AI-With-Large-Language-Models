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


3. Decoder only models
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
