# Methodology

## Starting evidence and marginal support labels

The method starts with the first five passages from the fixed fusion ranking described in Section 2. For a question \(q\), let \(S\) denote this evidence set. A replacement removes one selected passage \(o\) and inserts an unselected candidate \(d\), producing

\[
S'=(S\setminus\{o\})\cup\{d\}.
\]

HotpotQA's annotated supporting sentences are grouped by their document title. A group is covered when the selected passages contain every annotated supporting sentence for that title. Let \(F(S,q)\) be the number of groups fully covered by \(S\). The training label for a replacement is

\[
y(o,d\mid S,q)=F(S',q)-F(S,q).
\]

The label rewards newly completed groups and subtracts groups lost through removal. A passage can therefore be useful to insert but unsuitable as a replacement if it displaces another necessary passage. Positive, zero and negative labels indicate a net increase, no net change and a net decrease in complete support groups, respectively.

Labels are computed from the passage text available to the reader, using normalized titles and supporting sentence matching. They measure benchmark evidence coverage. Answer correctness and citation quality are evaluated separately. Reference answers and support annotations are used for labeling and evaluation, and are excluded from the model's prediction inputs.

## Candidate replacements and evidence constraints

Five selected passages and 45 unselected candidates give 225 possible replacements. To reduce scoring work, we construct a smaller candidate list for each outgoing passage using four rankings:

1. Relevance to the question, measured by the fixed relevance model.
2. Connections to retained passages, measured by occurrences of one passage's title in another passage's text.
3. The same title connections with other unselected candidates.
4. A reproducible ordering computed from the question identifier and replacement, to include candidates outside the first three rankings.

We take up to four new feasible replacements from each ranking for each outgoing slot, removing duplicates. This produces at most 80 candidate replacements per question. Candidate construction does not consult reference answers or supporting fact annotations.

Every replacement must preserve five distinct passages and keep the other four passages in their original positions. Its evidence length cannot exceed the initial fusion set under either of the two token counting models used during preparation. The shared maximum is 1,536 tokens, including titles and passage identifiers. The incoming passage occupies the outgoing passage's position. No further replacements are made after the first decision.

## Model inputs and training

Each candidate replacement is represented by 43 numerical features describing:

| Information | Examples of features |
| :--- | :--- |
| Incoming and outgoing passages | Relevance scores, retrieval positions, overlap with the question and passage lengths |
| Change caused by replacement | Differences in relevance and evidence length |
| Retained evidence | Relevance summaries and similarities between each candidate and the retained passages |
| Connections between passages | Title occurrences linking passages and changes in those connections |

A small neural network maps these 43 features through a hidden layer of 32 units to one predicted utility value. It has 1,441 trainable parameters. Feature values are rescaled using statistics from the training examples only.

The 512 training questions produce **33,673 labeled candidate replacements**. Since useful replacements are less common than neutral or harmful ones, we sample up to four examples per question from each available category: gains without lost support, replacements that lose support, and neutral changes. The final fitting sample contains **4,159 examples**. The 128 validation questions supply 8,392 candidate replacements.

Training minimizes the average squared difference between predicted utility and the label in Section 4.1. The implementation uses NumPy, 60 training rounds, batches of 32 examples and Adam weight updates with a learning rate of 0.001. Adam adjusts the model weights using the observed prediction errors. The random setting is fixed at 17, and we retain the weights with the lowest validation prediction error. No reader generations are required to construct these training labels or fit the selector.

For comparison, the incoming passage control uses the same architecture, labels and fitting examples, but only its 11 incoming passage features are active. This tests whether information about the outgoing passage and retained evidence improves selection.

## Selection rule and worked example

The trained model scores every feasible candidate replacement. We apply the replacement with the highest score only when its predicted utility exceeds **0.15**. Otherwise, the original fusion set is retained. The threshold is chosen on validation questions to maximize the number of evidence chains completed minus the number broken, with ties favouring fewer replacements. The incoming passage control independently selects a threshold of 0.30. Both thresholds are fixed before final evaluation.

```text
S = first five passages from fusion
E = candidate replacements satisfying the evidence constraints
if E is empty:
    return S
e = replacement in E with the highest predicted utility
if predicted utility of e > 0.15:
    return S with e applied
return S
```

Consider an illustrative question: **Where was the director of Film A born?** Passage D1 identifies the director, while another candidate, D6, gives that person's birthplace. The initial set includes D1 and a redundant passage D2, but lacks D6. It covers one of the two supporting document groups.

| Replacement | Covered groups afterward | Training label |
| :--- | ---: | ---: |
| Replace redundant D2 with birthplace passage D6 | 2 | +1 |
| Replace director passage D1 with birthplace passage D6 | 1 | 0 |
| Replace director passage D1 with an irrelevant passage | 0 | Negative one |

The first replacement completes the reasoning chain. The second gains birthplace evidence but loses the link identifying the director, so its net support gain is zero. This example illustrates the label definition; actual selection uses the model's predicted scores. The selector preserves the initial set whenever none of those scores clears the threshold.
