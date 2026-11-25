# CodeBrain

CodeBrain 是 ICLR 2026 发表的两阶段 EEG 基础模型。本机的 `upstream/` 保存作者代码，`checkpoints/` 保存两阶段的官方预训练权重。代码与权重均标注 Apache-2.0。[论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/86d1ab582afb247ccaa84bec4a7e24f7-Abstract-Conference.html) · [源码版本](https://github.com/jingyingma01/CodeBrain/tree/22d350caf68246d2fda4f630ef837420db3fb130) · [权重版本](https://huggingface.co/YjMajy/CodeBrain/tree/bef08d2fdb1759685371cc635aad21ce59163689)

| 权重文件 | 论文中的用途 | 官方 SHA-256 |
| --- | --- | --- |
| `checkpoints/CodeBrain_Tokenizer.pth` | 第一阶段，TFDual-Tokenizer 时频双码本分词器 | `e9560b670d64ea4712fd99a48dc2131326b919744c7d3eb504cf57b1ef3af999` |
| `checkpoints/CodeBrain.pth` | 第二阶段，EEGSSM 编码器 | `d9714b8732c9883a04d022ee66254cd578ae1fa27f5458e6ab7f1aa96e9a7352` |

`upstream/Checkpoints/` 下的同名符号链接指向本机权重，符合[作者的目录约定](https://github.com/jingyingma01/CodeBrain/blob/22d350caf68246d2fda4f630ef837420db3fb130/Checkpoints/README.md)。
