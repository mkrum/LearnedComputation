# LearnedComputation

Transfomers are a neural network architecture that has reached state of the art
across many tasks. Initially used for natural language processing, these models
have slowly bled into the surrounding fields: from reinforcement learning to
computer vision. They provide an efficient an scalable way for modeling
sequence-to-sequence problems, which enables a flexible framework that can be
applied to many different problems.

The paper that we chose, Leaning Advanced Computation From Example, use these
networks in order to learn how to perform different types of computation. This
paper demonstrates the flexibility of these networks. They we able to develop a
simple grammar that converted their mathematical problems into a simple,
consistent syntax. They were then able to use this syntax to generate
datasets of (problem, solution) pairings and were able to learn the
mapping between these two using a vanilla transformer. 

Using these models typically rely on what is known as a tokenizer. Our raw input
will typically be some text-based expression. We need to convert this text into
a list of integers within some known dictionary. For example, we could convert
raw text into a tokens representing the index of each symbol within some list. A
sentence like:
```
the dog ran
```
Would then be mapped into the sequence of integers:
```
20,8,5,27,4,15,7,27,18,1,14
```
These would then be passed into an embedding layer. This layer converts the raw
integer into a unique dense representation. These representations are then
treated as parameters, meaning that the representation for the letter 'a' will
start out random and become something that captures the semantics of 'a'.

In practice, representations are almost never done at the character level.
Learning the right level of abstraction is an active field of research in NLP,
with most methods using some kind of cluster to try to find the most efficient
tokens to split up text. There is a balance between width, the size of the
dictinoary, and depth, the required number of tokens to specify some text.
Increase the number of tokens will usually decrease the total number needed to
express text, but at the cost of increasing the total number of tokens that need
to be considered at any one time.

To represent numbers, the paper takes a similar approach. They convert each
number into a tokenized representation based on their raw symbols. For example,
the number 256 will become
```
[INT+, 2, 5, 6]
```
However, the model will have no idea about any of the relationships between `2`
and `5` and `6`. It will treat like the raw symbols from the language case and
have to learn how to map from `256` the symbol to `256` the concept. 

For this project, we wanted to experiment with different ways we can represent
numbers for deep learning models. On its face, it seems very inefficient to
represent numbers with their raw symbolic values. We already have dense
representations that we can use for numbers from their underlying representation
on the hardware. 
