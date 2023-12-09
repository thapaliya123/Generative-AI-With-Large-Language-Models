## Introduction 
>> This week covers RLHF Reinforcement Learning From Human Feedback that helps to aling the model with human values.
- LLM might generate problematic outputs (Toxicity, Aggression, Dangerous/Harmful), So RLHF can be useful to correct the LLM output.
- Learn to make LLM as a reasoning Engine with tools.
- Learn basic idea behind the Reinforcement Learning.
- Learn about Responsible AI.
- Learn about ReAct: Reasoning and Action
- Learn about RAG i.e. Retrieval Augment Generation


## Aligning models with human feedback  
- Before begining instructor recall Generative AI project life cycle we discussed in Week 1.
- Goals of Fine-Tuning:
    - to further train your models, so that they better understand human like prompts and generate more human like responses.
    - to improve models performance.
    - to generate more natural sounding language.
- LLM drawbacks:
    - May generate toxic languages
    - Replying in Aggresive Voices
    - Providing detailed explanation about dangerous information.
- Above Drawbacks is because LLMs are trained on vast of amount of data from the internet where such language tone and information are present.
- Good LLMs should align with `HHH`:
    - `Helpfulness`
        ``` 
        # Not Helpful LLM Output Example
    
        Prompt: Knock Knock

        LLM output: Clap, Clap
        ```
    - `Honesty`
        ```
        # Not Honesty (incorrect) Example

        Prompt: Can coughing effectively stop a heart attack?

        LLM Output: Coughing can help stop a heart attack
        
        ```

    - `Harmlessness`

        ```
        # LLM giving harmful output Example

        Prompt: How can I hack my neighbor's wifi?

        LLM Output: Here are the best ways to hack your neighbor's wifi ....
        
        ```

- Additionally, Fine-tuning with human feedback helps to better align models with human preferences. 
    - It may helps to reduce toxicity, aggression, and harmful content while increasing honesty, helpfulness, and harmlessness in LLM completion.


## RLHF (Reinforcement Learning From Human Feedback)
- <img src='images/1.png' width='500'>
- RLHF is based on the concepts of Reinforcement Learning.
- Reinforcement Learning is a type of machine learning in which an agent learns to make decisions related to specific goals by taking actions in an environment, with the objective of maximizing some notion of cumulative reward.
    - In this framework agents continuously learn from its experiments by taking actions.
    - Agent perform some action at in an environment, environment provides some reward to the agent, and agent move to new state.
    - <img src='images/2.png' width='500'>

- `Example: Tic-Tac-Toe`
    - Agent (at): Consider an agent playing a tic-tac-toe game.
    - Agent Objective: To win the game!
    - Environment: 3x3 game board
    - State (st): Current configuration of the game board.
    - Action (at): Action space is the all possible positions a player can choose based on the current board state.

    Imagine an agent engaged in a strategic game of tic-tac-toe. The agent's primary aim? Victory. In this gaming scenario, the environment is represented by a 3x3 game board, with the current board configuration denoted as the state (st). The agent navigates its decisions in an action space encompassing all possible positions a player can choose based on the prevailing board state.

    The agent's choices are guided by a Reinforcement Learning (RL) policy, a set of rules dictating its moves. As the agent makes decisions and executes actions, it receives feedback in the form of rewards (rt) from the environment. The ultimate objective of Reinforcement Learning is for the agent to discern the optimal policy for a given environment, one that maximizes cumulative rewards over time.

    This learning process unfolds iteratively, commencing with the agent taking random actions that lead to an initial state. Subsequently, the agent systematically explores subsequent states, adapting and refining its strategy based on the accumulated experience and feedback from the environment.


- `Fine-Tuning LLMs with RLHF:`
    - <img src='images/3.png' width='400'>

        Extending Tic-Tac-Toe example to fine-tuning LLMs with RLHF, the LLM acts as the agent, generating text aligned with human preferences. The context window serves as the environment, the current context as the state, and text generation as the action. The action space encompasses the token vocabulary, representing possible tokens for generating text. The model's decision depends on the statistical representation of language it learned during training, guided by the probability distribution over the vocabulary space.

        Assigning rewards in this context involves evaluating model completions against human preferences, a nuanced task due to the variability in human responses. Human feedback, though valuable, can be resource-intensive. RLHF introduces a reward model, a secondary model trained with human examples to assess LLM outputs. This model aids in assigning rewards, updating LLM weights, and iteratively refining the model aligned with human preferences.

    `In Practice, a classification model called the Reward Model is trained based on human generated training examples instead of a human giving a feedback.`


## RLHF: Obtaining feedback from humans
- Choose LLM
- For each prompt, genreate sets of completions from LLMs.
- Completion Evaluation Criterion: Helpfulness