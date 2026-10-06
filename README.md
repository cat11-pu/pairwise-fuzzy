# fuzzy

一小套模糊匹配打分内核：把文本切成 n-gram 集合，用 Jaccard 与 Dice 系数打底，
再按共同前缀给奖励、按长度差扣分，得出一个 0 到 1 之间的相似度；候选表在此之上
提供按阈值过滤、按分数降序排列的 Top-K 查询。内核只用 Python 标准库，结果只取决于
文本与加入顺序，不做任何 I/O，也不依赖随机数。

## 目录

- fuzzy/core.py 打分内核（normalize、ngrams、jaccard、dice、score、FuzzyMatcher）
- tests/test_core.py 行为测试

## 运行测试

在项目根目录（本文件所在目录）执行：

    python3 -m unittest discover -s tests -v

全部用例通过时进程退出码为 0。若系统里没有 `python3`，把命令里的 `python3` 换成
`python` 即可（Windows 上常见）。
