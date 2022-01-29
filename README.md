# Improving Systematic Generalization Through Modularity and Augmentation

---

This repository contains the code and additional documentation for the anonymous CogSci 2022 submission "Improving Systematic Generalization Through Modularity and Augmentation".

Contents:

- The full domain-specific language (DSL) that was used for structured data augmentation, including the meta-grammar, and the L-system programs for each adverb in gSCAN.
- The full model architecture for both the baseline and the modular model.
- The code for all the experiments and the structured data augmentation.
- Scripts to run the experiments in the paper.
- The actual predictions made by the models in all experiments are available once this repository is de-anonymized (too large files to upload).

---

## Structured Data Augmentation

### The meta-grammar
In the file `Code/GroundedScan/dsl.py` all the code related to the domain-specific language can be found.
Specifically, find the meta-grammar in the class `MetaGrammar`, where all the rewrite
rules used to generate new adverbs can be found in the function `add_rules()`. 

<details>
<summary>Click for full meta-grammar.</summary>
<br>

```python3
ACTION -> Tl
ACTION -> Tr
ACTION -> Stay
Walk -> ACTION Walk
Walk -> Walk ACTION
{ACTION}ACTION ACTION{ACTION} -> Tl Tl
{ACTION}ACTION ACTION{ACTION} -> Tr Tr
Push -> ACTION Push
Push -> Push ACTION
Pull -> ACTION Pull
Pull -> Pull ACTION
East -> ACTION East
East -> East ACTION
East -> North East South
East -> South East North
North -> ACTION North
North -> North ACTION
North -> East North West
North -> West North East
South -> ACTION South
South -> South ACTION
South -> East South West
South -> West South East
West -> ACTION West
West -> West ACTION
West -> North West South
West -> South West North
East South -> South East
East North -> North East
West South -> South West
West North -> North West
South East -> East South
North East -> East North
South West -> West South
North West -> West North
```

</details>

### The program for each gSCAN adverb

By selecting a subset of rules from the meta-grammar, each adverb program can be constructed.
Note that in the meta-grammar you can have a rules like `Pull -> ACTION Pull` and
`ACTION -> Tl`. Fully uppercase symbols need to be replaced by another symbol for the adverb program
to be finished. 

Additionally, while zigzagging type adverbs are applied recursively to a sequence instead of in parallel.

<details>
<summary>Cautiously</summary>
<br>

```Python3

Pull -> Tl Tr Tr Tl Pull
Push -> Tl Tr Tr Tl Push
Walk -> Tl Tr Tr Tl Walk
```
</details>

<details>
<summary>Hesitantly</summary>
<br>

```Python3

Pull -> Pull Stay
Push -> Push Stay
Walk -> Walk Stay
```
</details>

<details>
<summary>While Spinning</summary>
<br>

```Python3

Push -> Tl Tl Tl Tl Push
Pull -> Tl Tl Tl Tl Pull
West -> Tl Tl Tl Tl West
South -> Tl Tl Tl Tl South
North -> Tl Tl Tl Tl North
East -> Tl Tl Tl Tl East
```
</details>

<details>
<summary>While Zigzagging</summary>
<br>

```Python3

West South -> South West
West North -> North West
East North -> North East
East South -> South East
```
</details>


### Sampling new adverbs

New adverbs can be sampled from the meta-grammar by selecting rules from the meta-grammar
and iteratively replacing all right-hand symbols that aren't action primitives by
other rules with that symbol on the LHS. Alternatively, all possible adverb programs can be generated
and one can sample from this. For a function that does the latter see `AdverbWorld.generate_all_adverbs()`
in `Code/GroundedScan/dsl.py`

## 
