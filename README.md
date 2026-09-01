# Cognitive Shorts · Batch Prediction

这是 [SinglePrediction](https://github.com/PSCRedefine/SinglePrediction) 的下一站。上一个项目回答的是「**这一个人看这一个视频，会不会互动？**」；这个项目回答的是「**这一百条记录，一次全给我，而且中间有几条是脏数据也不许崩。**」

听起来只是把 for 循环搬到服务端，其实不是。批量场景会把单条场景里所有能被忽略的问题——延迟、内存、部分失败、概率是否可信、阈值定在哪——统统放大一百倍。这个仓库把这些问题一个一个拆开、量化、并给出取舍理由。

> **可点击的演示**：`dist/batch_console.html` 是一个不需要后端、双击就能打开的单文件页面。里面的每一个数字都是仓库里那个真实模型离线算出来的，包括 What-if 滑块（脚本预先在网格上跑过一遍，页面只负责查表）。

![批量控制台](image/console_batch.png)

---

## 目录

- [写给完全没接触过模型的你](#写给完全没接触过模型的你)
- [数据从哪来，标签怎么定](#数据从哪来标签怎么定)
- [完整链路](#完整链路)
- [核心设计：三段式批量引擎](#核心设计三段式批量引擎)
- [模型选型：为什么最后是最简单的那个](#模型选型为什么最后是最简单的那个)
- [训练过程长什么样](#训练过程长什么样)
- [阈值 0.5 是个默认值，不是一个答案](#阈值-05-是个默认值不是一个答案)
- [排序质量：批量真正被拿来干什么](#排序质量批量真正被拿来干什么)
- [真实数据的坑：我们踩到的五个](#真实数据的坑我们踩到的五个)
- [扩展性与部署](#扩展性与部署)
- [目录说明](#目录说明)
- [从零运行](#从零运行)
- [验收路线](#验收路线)
- [已知局限与下一步](#已知局限与下一步)
- [参考资料](#参考资料)

---

## 写给完全没接触过模型的你

如果你没做过机器学习，下面五段话足够看懂这个仓库在干什么。

**一、什么叫"预测"。** 我们手里有 50 万条历史记录，每条记录写着「某人看了某视频多久，之后有没有点赞/分享/评论/关注/重播」。所谓训练一个模型，就是让程序从这 50 万条里自己总结出规律；所谓预测，就是给它一条它没见过的新记录，让它输出一个 0 到 1 之间的数：**这次观看发生主动互动的概率**。

**二、什么叫"特征"。** 模型不认识"用户"和"视频"，它只认识数字和类别。所以我们要把「用户 user_000001」翻译成「35 岁、女、美国、账号存在 262 天、历史点赞 266 次……」，把「视频 video_0000001」翻译成「时长 55 秒、时尚类、完播率 0.673、趋势分 0.955……」。这些翻译出来的列就叫特征。本项目一共 32 个特征。

**三、什么叫"训练/验证/测试"。** 把数据切成三份：模型在第一份上**学习**；我们在第二份上**挑模型、调概率**；最后在第三份上**打分汇报**。第三份必须从头到尾没被碰过——否则你汇报的不是模型有多好，而是你试了多少次。

**四、单条预测和批量预测差在哪。** 单条预测是"一次一个人点一下按钮"。批量预测是"运营导入一个 CSV，一次要 100 条结果"，或者"每天凌晨给 1000 万个用户各算 20 个候选视频的分数"。差别有三个：
- **速度**：单条 10 毫秒无所谓；一亿条就是 11 天。
- **容错**：单条失败就报错给用户看；批量里第 37 行格式错了，不能把另外 99 行一起弄没。
- **概率的可信度**：单条只看一个数字，凑合就行；批量要**求平均、要卡阈值、要按分数排序**，这时候"模型说 0.3 到底是不是真的 30%"就变成一个必须回答的问题。

**五、这个仓库最想让你带走的一句话。** 模型选型不是"谁的准确率高谁赢"。这里四个模型的准确率在统计意义上分不出高下（下面有证据），所以真正决定上线谁的，是体积、延迟和可维护性。

---

## 数据从哪来，标签怎么定

### 三张原始表

| 文件 | 行数 | 是什么 |
|---|---|---|
| `data/users.csv` | 25,000 | 用户画像与历史统计（年龄、国家、是否付费、历史观看/点赞总量……） |
| `data/videos.csv` | 35,000 | 视频内容与累计统计（时长、分类、完播率、互动率、趋势分……） |
| `data/interactions.csv` | 150,000 | 观看流水，也是标签的来源 |

### 标签：`target_engaged`

```
target_engaged = liked OR shared OR commented OR followed_creator OR replayed
```

一次观看只要发生过任意一种**主动**行为，就记为 1。当前数据的正例率是 **24.1%**——也就是说，四次观看里大约一次会产生互动。这个"不平衡"的事实会一路影响后面所有的指标选择。

### 为什么不能拿 `engagement_score` 当特征

原始数据里有一列 `engagement_score`，看起来是个现成的好特征。它是**这次互动发生之后**才算得出来的分数。把它放进特征，等于考试时把答案印在题干里：训练集上准确率能飙到 0.99，上线后全线崩溃。因为线上做预测的那一刻，这一列根本还不存在。

这个坑有个正式名字叫**数据泄漏（label leakage）**，是新手项目最常见的死法。所以 `features.py` 里有一行明确的黑名单：

```python
LEAKY_COLUMNS = ["liked", "shared", "commented", "followed_creator", "replayed", "engagement_score"]
```

线上真正能拿到的输入只有四个：`user_id`、`video_id`、`watch_time`、`hour_of_day`。剩下 28 个特征由服务端拿 id 去特征库里查出来补齐——这也是为什么在线和离线必须共用同一份特征代码。

### 关于本仓库里的 `interactions.csv`

课程原始的 `interactions.csv` 有 139 MB，超过 GitHub 单文件 100 MB 的上限，无法入库。没有它，任何人 clone 下来都跑不通。

所以 `scripts/generate_sample_data.py` 会**合成**一份同样结构的流水。它是诚实的虚构：列名、类型、连接键都是真的，行为是我手写的一个生成模型。这个生成模型刻意包含了三种非线性——

- **阈值效应**：看完 85% 以上，互动意愿会跳一个台阶（"我把它看完了，所以我点了个赞"）；
- **交互效应**：喜欢的分类 × 看过一半、付费用户 × 完播、晚间 × 喜欢的分类；
- **饱和效应**：视频太旧之后，再高的趋势分也不再有帮助。

**为什么要刻意加这些？** 因为如果生成器是纯线性的，线性模型必然赢，而这个"赢"只反映我怎么造的数据，不反映真实世界。加上非线性，树模型才有东西可学，这场比较才有意义。

> ⚠️ **必读的诚实声明**：README 里所有指标都是在这份合成数据上测出来的。换成课程真实的 `interactions.csv` 后，**必须重跑一遍**，模型排名很可能会变。方法、代码、图表全部不用改，只有数字会变。

---

## 完整链路

```text
users.csv (25k)  videos.csv (35k)  interactions.csv (150k)
        └────────────┬────────────────────┘
                     ▼
        prepare_data.py  分块流式 join + Pandera 校验
                     ▼
        processed_interactions.csv   150k 行 × 32 特征 + 1 标签
                     ▼
        train.py   按 session 分组切分 → 4 个候选模型
                   → 配对 bootstrap 判定统计打平
                   → 成本择优 → 概率校准（如有收益）
                     ▼
        best_model.joblib + model_metadata.json + training_report.json + 8 张图
                     ▼
        api.py     POST /predict        单条
                   POST /predict/batch  ≤100 条，逐行容错
                     ▼
        app.py     Streamlit：CSV 上传 / 手动录入 / 结果看板 / 下载
```

---

## 核心设计：三段式批量引擎

### 需求里藏着一个矛盾

需求文档 §4.4 写着：*"遍历请求列表时，需对单条数据进行 try-except 包裹，单条失败不应导致整个批次崩溃。"*

照字面写出来是这样：

```python
for row in rows:                     # ❌ 看起来很对
    try:
        results.append(model.predict_proba(build_features(row)))
    except Exception as e:
        results.append({"error": str(e)})
```

它满足了容错，但把性能毁了。scikit-learn 的 `ColumnTransformer` 每次调用都有一笔**固定开销**（校验、拼装、one-hot 编码的调度），这笔钱和行数几乎无关。调 100 次就付 100 遍。

实测（`scripts/benchmark_batch.py`，同一批 100 行数据，同一个模型）：

| 批大小 | 逐行推理 | 一次向量化推理 | 差距 |
|---:|---:|---:|---:|
| 1 | 11.8 ms | 11.1 ms | 1.1× |
| 10 | 108 ms | 10.2 ms | **10.6×** |
| 50 | 606 ms | 11.3 ms | **53.6×** |
| 100 | 1226 ms | 12.7 ms | **96.4×** |

![批量推理基准](image/benchmark_batch.png)

一秒二对零点零一二秒。这不是微优化，这是"这个接口能不能上线"的区别。

### 解法：把容错和推理分到不同的阶段

关键观察是：**真实世界里几乎所有的行级错误，都发生在推理之前**——id 不存在、`watch_time` 是个字符串、`hour_of_day` 写成了 99。这些检查非常便宜。而真正贵的是模型推理，它反而几乎不会因为单行数据出错。

于是把一次批量拆成三段（代码见 `src/batch_prediction/batch.py`）：

| 阶段 | 做什么 | 容错粒度 | 成本 |
|---|---|---|---|
| **第一段** | 格式校验 + 范围校验 + 特征库查表 | 逐行，用向量化布尔掩码实现 | 便宜，随便隔离 |
| **第二段** | 把活下来的行拼成一个 DataFrame，调**一次** `predict_proba` | 整批 | 贵，只付一次 |
| **第三段** | 万一第二段仍然抛异常，才退回逐行模式，并把 `degraded=true` 写进响应 | 逐行 | 慢，但不是 500 |

一句话：**容错发生在便宜的地方，向量化发生在贵的地方。** 两个需求不再互相打架。

第三段不是形式主义。真的会有那种"某一行的类别值让底层库炸了"的情况；这时候整批返回 500 是最糟的结果，因为调用方不知道该重试哪一行。退化成慢速模式、把坏行标出来、其余照常返回，才是可运维的行为。

### 顺带一提：HTTP 状态码怎么定

| 情况 | 状态码 | 理由 |
|---|---|---|
| 100 行里有 37 行 id 不存在 | **200** | 请求本身是成功的。失败的是行，不是请求。每行自带 `error` 字段 |
| 提交了 101 行 | **422** | 超出契约，在 Pydantic 校验阶段就被挡掉，一行活都没干 |
| 模型还没训练 / 加载失败 | **503** | 服务不可用，且 `/health` 会同时告诉你原因 |
| 单条 `/predict` 遇到未知 id | **404** | 单条接口没有"部分成功"，找不到就是找不到 |

批量接口和单条接口对同一个错误给出不同的状态码，这是刻意的：单条的调用方要的是"成功或失败"，批量的调用方要的是"哪些成功了"。

---

## 模型选型：为什么最后是最简单的那个

完整推演见 **[docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md)**。这里给结论和证据。

### 四个候选，各自的作用

| 模型 | 为什么让它进场 |
|---|---|
| **Logistic Regression** | 线性基线。它的作用不是赢，是给出"这个问题有多难"的地板值。如果复杂模型只比它好一点点，说明**特征没做够**，而不是模型不够强 |
| **Random Forest** | 袋装树，表格数据的"安全牌"，几乎不用调参。代价是体积和推理速度——这两项在批量场景里会被放大 100 倍 |
| **Gradient Boosting** | sklearn 原生提升树。用来验证"提升树这一族是否真的比袋装树强"，把功劳归给算法族而不是某个实现 |
| **LightGBM** | 工业界表格数据的默认选择，直方图分箱 + 叶子优先生长，原生支持早停。预期的赢家，但必须用同一套指标赢下来 |

### 结果：四个模型在统计上分不出高下

![候选模型对照](image/model_comparison.png)

| 模型 | 验证 ROC-AUC | 95% 区间 | Brier | 每 100 行延迟 | 模型体积 | 训练耗时 |
|---|---:|---|---:|---:|---:|---:|
| logistic_regression | **0.6932** | [0.6852, 0.7011] | 0.1652 | **9.1 ms** | **0.011 MB** | 2.7 s |
| gradient_boosting | 0.6929 | [0.6848, 0.7008] | 0.1651 | 9.6 ms | 0.21 MB | 9.3 s |
| lightgbm | 0.6920 | [0.6844, 0.7004] | 0.1652 | 11.0 ms | 1.35 MB | 5.8 s |
| random_forest | 0.6913 | [0.6832, 0.6997] | 0.1663 | 76.0 ms | 26.85 MB | 51.7 s |

最高分和最低分只差 **0.0019**。这个差距有意义吗？没有。

判断方法是**配对 bootstrap**：把验证集有放回地重采样 300 次，每一次都让四个模型在**同一份重采样**上重新算 AUC（配对，让共同的噪声抵消），然后看"与第一名的差距"的分布。结果——

```
random_forest      与第一名差距 95% 区间 [-0.0002, +0.0040]   包含 0 → 打平
gradient_boosting  与第一名差距 95% 区间 [-0.0018, +0.0028]   包含 0 → 打平
lightgbm           与第一名差距 95% 区间 [-0.0010, +0.0036]   包含 0 → 打平
```

区间都跨过 0，意味着**换一份验证集，排名很可能就反过来**。所谓"LightGBM 比逻辑回归高 0.001"，是抽样噪声，不是本事。

### 所以选型规则长这样（写死在 `train.py` 里）

```
① 成本一票否决：延迟 > 250 ms/100行 或 体积 > 200 MB → 出局，多准都没用
② 统计打平判定：配对 bootstrap，区间含 0 的都算和第一名平手
③ 打平集合里挑最便宜的：先比延迟，再比体积
```

按这个规则，冠军是 **logistic_regression**：它比最大的候选**小 2441 倍**、每 100 行**快 8.4 倍**，而准确率一分没输。

这条规则不是我发明的，它是 CART 里"one standard error rule"的思路（Breiman et al., 1984）：在统计上分不出差别的候选里，选最简单的那个。区别只是我把"简单"的定义从"树更浅"换成了"上线更便宜"。

### 为什么这个结论值得写下来，而不是让人尴尬

一个学生项目里最常见的场景是：跑完四个模型，LightGBM 高 0.003，于是宣布"LightGBM 最优"，然后把一个 1.3 MB 的模型和一堆依赖带上线。

**这不是选型，这是看排行榜。** 正确的问法是三个：

1. 这个差距超出噪声了吗？（这里：没有）
2. 复杂模型带来的额外成本是多少？（这里：体积 ×2441，延迟 ×8.4，依赖多一个 C++ 编译的库）
3. 如果分不出高下，我更愿意在凌晨三点排查哪一个？（一个 11 KB 的线性方程，还是一片森林）

顺带还有一个信息：**四个模型打平这件事本身就是结论**——它说明瓶颈在特征，不在模型。想再往上走，该做的是加特征（用户最近 7 天行为、视频 embedding、用户-分类亲和度），而不是换更大的模型。

### 关于 `class_weight="balanced"`：一个刻意没用的选项

正例只有 24%，很多教程会建议加 `class_weight="balanced"`。这里**四个模型全都没加**，原因是：

- 它对 ROC-AUC 几乎没有帮助（AUC 只看排序）；
- 它会把所有预测概率整体抬高，**破坏校准**——模型说 0.5 的那批人，实际只有 0.3 会互动；
- 而批量场景恰恰最依赖校准：要算平均概率、要卡阈值、要按分数排序。

正确的做法是：**让模型输出诚实的概率，然后单独选一个阈值**。这两件事分开做，才能各自解释、各自调整。详见下面的阈值一节。

### 概率校准：做了，但没用上

![校准曲线](image/calibration.png)

我们仍然跑了一遍 isotonic 校准（在验证集上拟合"模型分数 → 实际频率"的单调映射），结果是：

| 指标 | 原始 | 校准后 | 变化 |
|---|---:|---:|---|
| Brier（越低越好） | 0.16542 | 0.16549 | +0.0% |
| ECE（越低越好） | 0.01016 | 0.00834 | −17.8% |
| Log loss | 0.5070 | 0.5081 | +0.2% |

判定规则是"**Brier 必须变好才采用**"——Brier 和 log loss 是严格适当评分规则（proper scoring rule），而 ECE 单独一个是可以被"骗"的（把所有预测都压到基准率附近，ECE 会很好看，但模型变得毫无用处）。这里 Brier 没有改善，所以**保留未校准的模型**，少一层封装、少一份序列化风险。

看左边那张可靠性曲线就明白了：原始模型的点本来就贴着对角线——逻辑回归优化的就是 log loss，它天生输出校准良好的概率。**校准这一步在这里是个 no-op，但它必须跑，因为在换了模型或换了数据之后，它经常不是。**

---

## 训练过程长什么样

### 每加一棵树，模型学到了什么

![训练曲线](image/training_curves.png)

这张图是理解"训练"两个字最直观的方式。横轴是已经加进去的树的数量，蓝线是模型在**见过的数据**上的表现，橙线是在**没见过的数据**上的表现。

- 前 20 棵树：两条线一起猛涨，模型在学真正的规律；
- 20–85 棵：橙线慢慢躺平，蓝线还在爬——模型开始背答案了；
- 85 棵之后：橙线连续 50 轮没有改善，**早停**触发，训练停止。

蓝线和橙线之间的缝隙，就是**过拟合**的可视化。这条缝隙永远存在，问题只是"允许它有多大"。

### 再多给点数据还有用吗

![学习曲线](image/learning_curve.png)

| 训练行数 | 9,600 | 24,000 | 48,000 | 72,000 | 96,000 |
|---|---:|---:|---:|---:|---:|
| 验证 ROC-AUC | 0.6866 | 0.6907 | 0.6928 | 0.6929 | 0.6932 |

从 4.8 万到 9.6 万，数据翻了一倍，AUC 只涨了 0.0004。曲线已经躺平。

**这个结论很值钱**：它意味着"再去标 100 万条数据"是最贵、也最没用的改进方向。下一个增量必须来自**新特征**或**新问题定义**，不是更多的同类数据。很多团队在这里烧掉半年预算，而一张图就能提前告诉他们。

### 模型到底在看什么

![特征重要性](image/feature_importance.png)

`watch_ratio`（看完了百分之多少）一个特征就占了 10.7% 的权重，是第二名的三倍。这符合直觉——**看完一个视频，是最强的兴趣信号**。

看得懂特征重要性，才有可能发现"它在用一个不该用的字段"。如果哪天 `engagement_score` 出现在这张图的第一名，那就是泄漏又漏进来了。

### ROC 与 PR 曲线

![ROC 与 PR 曲线](image/roc_pr_curves.png)

四条线几乎完全重叠——这是"统计打平"的另一种看法。

顺带解释一下为什么两张图都要画：**ROC 曲线在类别不平衡时会过于乐观**。24% 的正例率下，一个把所有人都判为"不会互动"的废物模型，准确率就有 76%。PR 曲线（以及 PR-AUC ≈ 0.44，对比基准率 0.24）才诚实地反映"在你真正关心的那一小撮人身上，模型有多准"（Saito & Rehmsmeier, 2015）。

---

## 阈值 0.5 是个默认值，不是一个答案

![阈值权衡](image/threshold_tradeoff.png)

模型输出的是 0–1 的概率。要变成"会/不会"的判断，必须挑一个阈值。绝大多数教程直接用 0.5——那只在正负样本各占一半时才自然。

这里正例只有 24%。用 0.5 会发生什么：

| 阈值 | Precision | Recall | F1 | 被判为"会互动"的比例 |
|---|---:|---:|---:|---:|
| 0.50（默认） | 0.609 | 0.153 | **0.244** | 6.1% |
| 0.24（F1 最优） | ~0.38 | ~0.61 | **0.468** | ~38% |

**同一个模型，只是换了一个数字，F1 差 92%。** 0.5 的时候模型极度保守：它只敢标出 6% 的人，标出来的六成是对的，但漏掉了 85% 真正会互动的用户。

更重要的是：**这个数字根本不该由算法工程师定**。

- 如果下游是"给这些人推送一条通知"——推送很便宜，漏掉才可惜 → 阈值调低，要 recall；
- 如果下游是"给这些人发一张 10 元优惠券"——发错要花钱 → 阈值调高，要 precision；
- 如果下游只是"把候选视频排个序"——那**根本不需要阈值**，直接用概率排序。

所以本项目的做法是：模型只负责输出诚实的概率，`model_metadata.json` 里附带一个 F1 最优阈值 **0.24** 作为参考，`/predict/batch` 的响应**不做二值化决策**（`predicted_engaged` 只是按 0.5 给的一个便利字段，调用方可以无视）。**阈值属于业务决策，不属于模型。**

---

## 排序质量：批量真正被拿来干什么

一个批次在现实中通常长这样：*"给用户 A 的 20 个候选视频打分。"* 它是一个**候选集**，不是 20 条互不相干的记录。

这时候全局 AUC 就不够用了：AUC 衡量的是"随便抽一个正例和一个负例，模型能不能把正例排前面"，跨用户混着算。而产品真正在乎的是"**在这一个用户的列表里，最该看的那条有没有被排到第一**"。

所以额外算了三个组内指标（测试集里 4,516 个至少含一个正例的候选集）：

| 指标 | 数值 | 人话 |
|---|---:|---|
| **Precision@1** | 0.564 | 排在第一位的视频，有 56.4% 真的被互动了（随机排是 ~30%） |
| **NDCG@3** | 0.682 | 前三位的排序质量，1.0 是完美 |
| **MRR** | 0.732 | 第一个正例平均出现在第 1.37 位 |

![分位提升](image/lift_deciles.png)

还有一张运营会直接看的图：把所有预测按分数从高到低排，切成十份。

- **最高的 10%**：实际互动率是整体平均的 **2.25 倍**，这一档就装下了全部互动用户的 **22.5%**；
- **最高的 30%**：装下了 **51.5%** 的互动用户。

翻译成业务语言：**如果预算只够触达 30% 的用户，用这个模型挑，能覆盖一半以上真正会互动的人。** 这比"AUC 0.69"更容易让人点头。

---

## 真实数据的坑：我们踩到的五个

这一节是这个项目最值钱的部分——不是因为技术多难，而是因为这些坑没有一个会在单元测试里报错。

### 坑一：每次请求都在复制整张特征表

第一版压测，端到端稳定在 **400 ms**，而且**不管批大小是 1 还是 100 都一样**。这个"常数"是最大的线索：耗时和请求内容无关，那它一定和请求**之外**的东西有关。

`cProfile` 指向了这一行：

```python
valid.merge(self.users.drop(columns=["user_id"]), ...)   # ❌
```

`.drop()` 会**复制**整张 25,000 行的用户表——每次请求都复制一遍。修复方式是在构造 `FeatureStore` 的时候算一次，存起来。

### 坑二：`Series.isin` 在 Arrow 字符串上退化成 Python 循环

改完还剩 390 ms。继续 profile，凶手是：

```python
user_id.isin(self.users.index)   # ❌ 190 ms
```

pandas 新版默认的字符串类型是 Arrow 后端的。在这个类型上做 `isin`，pandas 内部会退回**逐元素的 Python 循环**，而且是对着 25,000 行的索引循环。换成一个普通的 `set`：

```python
known_user = np.fromiter((v in self.user_ids for v in user_id), dtype=bool, count=len(user_id))
```

复杂度从 O(特征库大小) 变成 O(批大小)。

**两处修完的效果：端到端 400 ms → 47 ms，吞吐 250 → 2,120 rows/s，模型一行没改。** 教训很直白：**性能问题经常不在模型里**，而在你觉得"这行代码人畜无害"的地方。而且只有真的去压测才会发现——两处 bug 都不影响正确性，任何单元测试都是绿的。

### 坑三：训练时是 `bool`，上线时是字符串

`is_premium` 这一列，从 CSV 读进来是 Python 的 `bool`，从 JSON 请求进来是字符串 `"true"`。one-hot 编码之后，前者产生的列名是 `is_premium_True`，后者是 `is_premium_true`——**两个不同的特征**。模型不会报错，只会悄悄地把这个特征当成"从没见过的值"，然后给出一个错误的预测。

这叫**训练/服务偏斜（train-serve skew）**，是生产 ML 系统最经典的故障（Sculley et al., 2015）。解法是在管道**内部**统一转成字符串：

```python
("as_text", FunctionTransformer(to_text, feature_names_out="one-to-one")),
```

只要它在 `Pipeline` 里，训练和推理走的就一定是同一段代码。

### 坑四：随机切分会泄漏

数据里一个 session 是"一个用户的 5 个候选视频"。如果按行随机切分，同一个 session 的第 1 条会进训练集，第 2 条会进测试集。这两条共享同一份用户快照和同一个时刻——模型等于**背下了这个 session**，测试分数会虚高。

所以用 `GroupShuffleSplit` 按 `session_id` 切，保证一整个候选集只出现在一侧。真实数据里没有 `session_id` 的话，代码会退而按 `user_id` 分组，并把用的是哪种策略写进 `model_metadata.json`（当前是 `group:session_id`）。

> 顺带一提：真正上生产时，还应该再加一层**按时间切分**——用 7 月的数据训练，用 8 月的数据验证。因为线上永远是"用过去预测未来"，而随机切分假装你能看到未来。这一步本项目没做，原因和限制写在[已知局限](#已知局限与下一步)里。

### 坑五：线上一定会遇到训练时没见过的类别

新上线一个视频分类、新支持一个国家，`OneHotEncoder` 遇到没见过的值默认会**抛异常**。在批量场景里，这意味着一条新视频能让 100 条记录一起失败。

```python
OneHotEncoder(handle_unknown="ignore", min_frequency=2)
```

`handle_unknown="ignore"` 让未知类别编码成全 0（"我不认识它"），而不是崩溃。`min_frequency=2` 把只出现过一次的取值合并成一个"稀有"桶，避免高基数列（比如 country）把矩阵撑爆。

---

## 扩展性与部署

两份独立文档：

- **[docs/SCALING.md](docs/SCALING.md)** — 从 100 行到 1 亿行的路线：为什么单批上限锁死在 100，什么时候该从同步接口换成异步任务，什么时候该把在线服务换成离线 Spark/Ray 作业，内存怎么算，特征库怎么从 CSV 换成 Redis/特征平台。
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** — Docker 镜像分层、worker 数怎么定、健康检查与就绪探针、模型工件为什么不打进镜像、灰度与回滚、上线后要监控什么（尤其是**数据漂移**和**校准漂移**）。

先给最关键的两个数字：

- **单批上限 100 不是性能限制，是保护机制。** 100 行的 P95 延迟约 47 ms，还有大量余量。上限的作用是让"一个失控的调用方"无法把服务拖垮。要处理 100 万行的客户端不需要更大的上限，它需要 `iter_chunks(rows, 100)` 和一个循环。
- **内存按 worker 线性增长。** 特征库常驻约 40 MB，每个 uvicorn worker 各持一份。4 worker ≈ 160 MB + 模型 11 KB。这就是为什么"模型体积"在选型时是一票否决项而不是加分项。

---

## 目录说明

```text
app.py                              Streamlit 前端（Batch / Single / Model Info 三页）
src/batch_prediction/
  config.py                        路径、批量上限、置信度分界，全部可用环境变量覆盖
  features.py                      唯一的特征构造入口：build_many 向量化，build_one 是它的单行包装
  batch.py                         三段式批量引擎（本项目的核心）
  schemas.py                       Pydantic 契约，批量上限在这里被强制
  api.py                           FastAPI：/predict /predict/batch /health /model/info
  prepare_data.py                  分块流式的离线特征表生成
  train.py                         四模型比较 + 统计打平判定 + 成本择优 + 校准
  metrics.py                       判别力 / 校准 / 候选集排序 三类指标
  plots.py                         8 张训练与选型图表
scripts/
  generate_sample_data.py          合成 interactions.csv（含非线性与噪声）
  make_sample_batch.py             生成"id 真实存在"的示例批次
  prepare_data.py / train_models.py  命令入口
  benchmark_batch.py               向量化 vs 逐行的压测
  build_demo_payload.py            为 HTML 控制台预计算真实预测
  build_console.py                 把数据内联进单文件 HTML
web/console_template.html          可点击控制台的模板
dist/batch_console.html            ⭐ 双击即可打开的成品页面
tests/                             32 个测试：特征、批量容错、HTTP 行为、指标正确性
docs/                              选型推演、架构决策记录、API、扩展、部署
image/                             README 里的所有图（由 train.py 生成）
reports/                           training_report.json / benchmark.json（不入库）
data/ models/                      数据与模型工件
```

---

## 从零运行

需要 Python 3.10–3.13。

```bash
python3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

然后一条命令跑完整条链路：

```bash
make all        # = 生成数据 → 特征表 → 训练选型出图 → 跑测试
```

或者拆开来看每一步：

```bash
python scripts/generate_sample_data.py       # 若你有真实的 interactions.csv，跳过这步
python -m batch_prediction.prepare_data      # → data/processed_interactions.csv
python -m batch_prediction.train             # → models/ + reports/ + image/
pytest -q                                    # 32 passed
python scripts/benchmark_batch.py            # 压测，约 1 分钟
```

快速冒烟（约 20 秒，会覆盖本地模型工件，最终结果请用完整数据重跑）：

```bash
python scripts/generate_sample_data.py --groups 1500
python -m batch_prediction.prepare_data
python -m batch_prediction.train --max-rows 8000
```

启动服务（两个终端，同一个虚拟环境）：

```bash
make api        # uvicorn，http://127.0.0.1:8000/docs
make ui         # streamlit，http://localhost:8501
```

或者用 Docker：

```bash
make train && docker compose up      # API :8000 + UI :8501
```

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{"requests":[
        {"user_id":"user_000001","video_id":"video_0000001","watch_time":45,"hour_of_day":21},
        {"user_id":"user_999999","video_id":"video_0000001","watch_time":45}
      ]}'
```

第二行故意用了一个不存在的 id，你会看到 HTTP 200，第一行有概率，第二行有 `error`。

---

## 验收路线

1. `python -m batch_prediction.prepare_data` — 输出行数与输入一致，正例率约 24%。
2. `python -m batch_prediction.train` — 四个模型都有指标，打印出"统计打平"的集合和最终冠军，`image/` 下生成 8 张图。
3. `pytest -q` — 32 项全绿（特征契约、逐行容错、退化路径、HTTP 状态码、指标正确性）。
4. `GET /health` — `status` 为 `healthy`，并回报特征库里索引了多少用户和视频。
5. 上传 `data/batch_requests_sample.csv` — 24 行全部成功，直方图和 CSV 下载可用。
6. 上传 `data/batch_requests.csv`（课程原始文件）— **HTTP 200，20 行全部带 `error`**，这是容错路径的验收点。
7. 提交 101 行 — 返回 422，且服务端没有开始任何计算。
8. `python scripts/benchmark_batch.py` — 100 行批次的向量化加速比应在 50× 以上。
9. 打开 `dist/batch_console.html` — 五个标签页都能点，What-if 滑块实时响应。

---

## 已知局限与下一步

诚实地列出这个项目**没做**的事，比列做了什么更有意义。

| 局限 | 影响 | 下一步该怎么做 |
|---|---|---|
| 指标基于合成数据 | 模型排名可能与真实数据不同 | 拿到真实 `interactions.csv` 后重跑 `make all`，把 README 的数字整体替换 |
| 按 session 切分，没有按时间切分 | 高估了模型在"预测未来"上的能力 | 改用时间切分：前 80% 时间训练，后 20% 验证 |
| 特征库是内存里的 CSV 快照 | 用户画像不会实时更新；重启才刷新 | 换成 Redis / 特征平台，并给特征加 TTL 与版本号 |
| 没有在线监控 | 数据漂移和校准漂移只能靠人发现 | 采样落盘预测 + 回填真实标签，日更 PSI 与 Brier |
| 没有 A/B 实验框架 | 无法证明模型真的改善了业务指标 | 离线指标只是必要条件，上线要看留存/时长 |
| 单机单进程 | 吞吐上限约 2,100 rows/s | 见 [docs/SCALING.md](docs/SCALING.md)：多 worker → 异步任务 → 离线批处理 |
| 只有一个全局模型 | 冷启动用户和头部用户共用一套参数 | 分群模型，或加入用户侧 embedding |

---

## 参考资料

方法上的每一个选择，都能追溯到一份公开材料：

**关于概率校准**
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities With Supervised Learning*. ICML. — 不同模型族的原生校准差异；逻辑回归天生校准良好，而袋装/提升树通常不是。
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks*. ICML. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) — ECE 的定义与分箱做法。
- scikit-learn, *Probability calibration*. https://scikit-learn.org/stable/modules/calibration.html — isotonic 与 sigmoid 的实现与适用条件。

**关于在不平衡数据上选指标**
- Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE 10(3). — 为什么本项目 ROC 和 PR 两张都画。
- Hanley, J. & McNeil, B. (1982). *The Meaning and Use of the Area Under a ROC Curve*. Radiology. — AUC 的概率解释。

**关于"两个模型是否真的有差别"**
- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. — 配对 bootstrap 的做法。
- DeLong, E. et al. (1988). *Comparing the Areas under Two or More Correlated ROC Curves*. Biometrics. — AUC 差异检验的经典解析解；本项目用 bootstrap 是因为它同样适用于 PR-AUC 等没有解析解的指标。
- Breiman, L. et al. (1984). *Classification and Regression Trees*. — "one standard error rule"：在统计上打平的候选里选最简单的。

**关于生产 ML 系统**
- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. — 训练/服务偏斜、粘合代码、配置债务。
- Zinkevich, M. *Rules of Machine Learning: Best Practices for ML Engineering*. https://developers.google.com/machine-learning/guides/rules-of-ml — 尤其是"先上线一个简单模型"和"把指标和阈值分开"。
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS. — 直方图分箱与叶子优先生长。

**工具文档**
- FastAPI https://fastapi.tiangolo.com/ · Pydantic v2 https://docs.pydantic.dev/ · Streamlit https://docs.streamlit.io/ · pandas https://pandas.pydata.org/docs/ · Pandera https://pandera.readthedocs.io/

**数据说明**：本仓库的 `users.csv` / `videos.csv` 来自课程提供的 Cognitive Shorts 数据集；`interactions.csv` 由 `scripts/generate_sample_data.py` 合成，原因见[上文](#关于本仓库里的-interactionscsv)。

---

## 上一个项目

- 单条预测：https://github.com/PSCRedefine/SinglePrediction
