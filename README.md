# Improving Systematic Generalization Through Modularity and Augmentation

---

This is the code and additional documentation for the anonymous CogSci 2022 submission "Improving Systematic Generalization Through Modularity and Augmentation".

The repository contains:

- The full domain-specific language (DSL) that was used for structured data augmentation, including the meta-grammar, and the L-system programs for each adverb in gSCAN.
- The full model architecture for both the baseline and the modular model.
- The code for all the experiments and the structured data augmentation.
- Scripts to run the experiments in the paper.
- The actual predictions made by the models in all experiments.

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

Pull -> Tl Tr Tr Tl Pull
Push -> Tl Tr Tr Tl Push
Walk -> Tl Tr Tr Tl Walk
```
</details>

<details>
<summary>While Spinning</summary>
<br>

```Python3

Pull -> Tl Tr Tr Tl Pull
Push -> Tl Tr Tr Tl Push
Walk -> Tl Tr Tr Tl Walk
```
</details>

<details>
<summary>While Zigzagging</summary>
<br>

```Python3

Pull -> Tl Tr Tr Tl Pull
Push -> Tl Tr Tr Tl Push
Walk -> Tl Tr Tr Tl Walk
```
</details>


### Sampling new adverbs

