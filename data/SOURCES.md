# Sources and attribution

HotpotQA was introduced by Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov and Christopher D. Manning in 2018. Its questions and supporting facts, and the processed Wikipedia corpus, are distributed under [CC BY SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). See the [official HotpotQA release](https://hotpotqa.github.io/) and [BEIR HotpotQA corpus](https://huggingface.co/datasets/BeIR/hotpotqa).

The cached HotpotQA inputs select questions from the public development collection and passages from the BEIR corpus. Passage prefixes, rendering, evidence labels and numerical retrieval features were prepared for the experiments. These transformations do not replace the upstream attribution or license for the question and passage text.

QAMPARI questions and answer references originate from the [official QAMPARI project](https://github.com/samsam3232/qampari). The transfer experiment uses the passage retrieval release and scoring implementation from [ALCE](https://github.com/princeton-nlp/ALCE). QAMPARI question records, retrieved Wikipedia passages and ALCE code have their respective upstream terms; a code license should not be interpreted as replacing the passage sources' terms. The prepared release retains the original question identifiers and cited document text.

Tokenizer files come from the pinned Qwen2.5, Qwen3 and Phi 4 model repositories listed in `manifests/sources.json`. Their original model repository terms and attribution apply. Only tokenizer JSON files are included; model weights are not redistributed here.

The source manifest records the exact public file URLs, model revisions and checksums used in preparation. These data assets contain benchmark material and derived features, not private annotation notes or credentials.
