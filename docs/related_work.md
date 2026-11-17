# Related work

## Utility based reranking

**eRAG** evaluates each retrieved document by generating an answer from that document alone and scoring the answer against a reference. These document scores are then used to evaluate retrieval rankings. Our training target instead measures the change in complete support after replacing a passage within an existing set. The contribution of a passage depends on what remains available after the replacement. [eRAG](https://arxiv.org/html/2404.13781)

**InfoGain RAG** measures how much a document changes the reader's confidence in the correct answer compared with receiving the question alone. It uses these measurements to train a reranker that distinguishes helpful and harmful documents. Our main selector learns from changes in annotated evidence support, and its features describe the incoming passage, outgoing passage and retained set. This also separates label construction from the reader's confidence in an answer. [InfoGain RAG](https://aclanthology.org/2025.emnlp-main.365/)

**RRPO** trains a reranker using the quality of the answer generated after each selection step as its reward. The reward therefore reflects the documents selected so far. Its passage scorer takes the question and an individual document as inputs; selection probabilities are updated as documents are removed from the candidate list. Our model explicitly scores a replacement using information about the outgoing passage and retained evidence. Its labels count support gained and lost, and training directly predicts that numerical change. [RRPO](https://arxiv.org/html/2604.02091)

**DynamicRAG** learns the order and number of passages to select. After learning from example selections, it learns to prefer complete selections whose generated answers receive better scores. Its training also updates the generator. Our method keeps the readers fixed, retains five passages and learns a single replacement from a fusion starting set. It uses benchmark support labels for the main selector rather than answer scores from complete selection sequences. [DynamicRAG](https://proceedings.neurips.cc/paper_files/paper/2025/file/f69d2246a85e28c3e63dee3daf8d4af8-Paper-Conference.pdf)

## Collective evidence selection

**SetR** identifies the information needed to answer a question and selects passages that collectively satisfy those requirements. It trains a language model from a stronger model's reasoning and passage selections. This directly addresses the need to assess passages together. Our implementation uses numerical features and a small neural network to repair an existing set, with supervision derived from measured support changes. [SetR](https://aclanthology.org/2025.acl-long.861/)

**UtilitySelection** trains a smaller language model to reproduce a teacher's generated answer and passage selection. During selection, it processes passages in successive windows and carries selected passages into later windows. It therefore already considers previously selected evidence. Our model predicts a replacement's net support change directly, preserves five passage positions and makes at most one edit. [UtilitySelection](https://arxiv.org/html/2507.19102v1)

**LMVU**, an October 2026 preprint, is the closest comparison to our benchmark supervised formulation. It learns which candidate adds more uncovered viewpoints to the current set, where a viewpoint represents a distinct answer or position. Its small linear model uses observable features and adds passages sequentially from an empty context under a token budget. The viewpoint annotations supply training labels and are unavailable at inference. Our method similarly separates benchmark supervision from prediction inputs, but learns the net completion and loss of supporting document groups during replacement. It also evaluates generated answers and their cited support. [LMVU](https://arxiv.org/html/2610.29234)

## Differences in our formulation and implementation

The defining decision in our method is: **given four retained passages, should the fifth passage be replaced by this candidate?** The incoming passage's usefulness and the consequences of removing the outgoing passage are learned together.

The closest comparison with LMVU can be summarized as follows:

| Element | LMVU | Our method |
| :--- | :--- | :--- |
| Starting evidence | Empty context | Five passages from fusion |
| Decision | Add a candidate to the growing set | Replace one selected passage or retain the set |
| Training target | Preferences based on additional viewpoint coverage | Numerical change in fully supported document groups, including losses |
| Learned scorer | Linear model of candidate and selected context features | Small neural network of incoming, outgoing and retained evidence features |
| Main evaluation | Viewpoint coverage within a token budget | Complete evidence chains, answer correctness and cited support |

This formulation preserves a strong retrieval result while testing whether a targeted exchange improves its support. The incoming passage control isolates the value of outgoing and retained evidence features. Forced replacement and passage order experiments examine the decision rule and sensitivity to presentation.

These are comparisons of methods and training signals. The performance comparisons in Section 6 use baselines evaluated on the same question sets and candidate pools as our selector; published scores from different experimental settings are not combined with those results.
