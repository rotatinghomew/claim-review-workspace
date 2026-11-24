# 分类与采用工作台运行手册（claimdesk）

本目录是在原始资料（`data/`、`materials/`，只读）之上交付的可训练、可比较、可选择、
可应用的包装声明分类能力。所有写入状态集中在 `local-state/`（已在 `.gitignore`）。

> 本系统只做**声明类型判读**：未添加某种糖 / 总糖含量声称 / 无关口味宣传 / 无法判断。
> 输出不是法规符合性结论；一个营养表上的“0”不会被当作无糖证据，缺项也不会被插补成 0。

## 1. 模块地图

| 模块 | 职责 |
| --- | --- |
| `config.py` | 路径、抽核概率/份额、随机种子；支持测试状态隔离 |
| `store.py` | 原始 JSONL 只读加载、本地状态读写 |
| `labels.py` | 训练真值与线上采用的**两套资格规则**、时点快照、适用性明细 |
| `features.py` | 文字 char n-gram + 显式短语标志 + 营养数值（每份折算每百克、缺糖独立标志） |
| `models.py` | 两个候选：`lr-text`（纯文字）、`lr-rich`（文字+配料+结构化），一对多多标签 |
| `evaluate.py` | 家族隔离切分、按类/按渠道可靠性、容限阈值扫描、抽核 IPW 错误率 |
| `policy.py` | release/review/hold 三类决定、容限选阈、周复核队列与随机抽核落盘 |
| `lifecycle.py` | 提交定稿、撤销传播、共享意见、发布快照/回退、模型差异、历史核对 |
| `ingest.py` | 新记录导入校验、发布/拒判依据查询 |
| `pipeline.py` | 训练→评估→应用→三方核对的串联入口 |
| `api.py` / `cli.py` | FastAPI 路由与命令行，共用同一服务层 |

## 2. 标准操作流程（CLI）

```bash
# 0) 安装与自检
python3 -m pip install -r requirements.lock.txt
python3 -m unittest discover -s tests -v

# 1) 导入一批新包装记录（外来文件单独留档，不覆盖原始表）
python3 -m claimdesk import --file /path/to/new.jsonl --batch 2026-W48-incoming

# 2) 训练两个候选模型（只使用合格正式定稿；临时/撤销/未来意见自动排除）
python3 -m claimdesk train --candidate lr-rich --tag v1
python3 -m claimdesk train --candidate lr-text  --tag v1

# 3) 家族隔离评估（同家族留版本 + 整族留出），按类/渠道输出并按容限选阈
python3 -m claimdesk evaluate --model <model_version>

# 4) 并排比较候选模型
python3 -m claimdesk compare --models <version_a> <version_b>

# 5) 对全目录生成 release/review/hold 决定（含抽核 IPW 误差估计）
python3 -m claimdesk apply --model <model_version>

# 6) 查看某规格发布或拒判的具体依据
python3 -m claimdesk explain --model <model_version> --product p4

# 7) 生成周复核队列（疑点 + 随机抽核，按编辑权限/分钟分配，抽核选择方式落盘）
python3 -m claimdesk queue --model <model_version> --week 2026-W48

# 8) 编辑提交意见（无定稿权限只能 temporary；跨印刷组借用被拒绝）
python3 -m claimdesk submit --product p4 --editor editor0 \
  --classes 无法判断 --valid-specs p4 --status final \
  --selected-by suspect_queue

# 9) 撤销定稿：传播“失去证据范围”，有独立定稿的规格保留
python3 -m claimdesk withdraw --review l46

# 10) 发布前比较候选与现行模型的栏目增删（销量按规格去重）
python3 -m claimdesk diff --current <current_version> --candidate <candidate_version>

# 11) 确认发布（先写快照，可整批回退）与回退
python3 -m claimdesk publish --model <model_version>
python3 -m claimdesk rollback --snapshot <snapshot_id>

# 12) 模型输出 / 人审状态 / 栏目采用三方核对，并核对历史 publication_records
python3 -m claimdesk reconcile --model <model_version>
```

HTTP 入口：`python3 -m uvicorn claimdesk.api:app`，路由见 `/docs`。

## 3. 评价纪律（为什么这样切分与汇报）

- **家族隔离**：`unseen_families` 把整族放在训练侧之外；`temporal_later_versions`
  在同家族内用低版本训练、高版本评估。同一家族的换码版本不会跨训练/验证两边。
- **未标注不是负样本**：评估指标只在当时有合格定稿的记录上计算；无意见规模以
  `n_eval_unlabeled` 单独报告。渠道 `ch3` 没有自身人审时明确标记
  `no_eligible_labels`，不静默给分。
- **按声明类与渠道汇报**：每类给出误放率、漏收率、自动处理覆盖、计数；
  总体精确率/召回只作补充，不能替代分口径数字。
- **容限选阈**：逐类在 `cost_policies.maximum_error` 内选覆盖最大的阈值；
  无可行阈值时该类标记 `feasible=false` 并禁止自动发布。
- **抽核估计**：只有 `random_audit` 且带 `selection_probability` 的定稿参与
  Horvitz–Thompson 逆概率加权；`suspect_queue` 是定向样本，不代表全目录；
  未收到抽核意见的发布不计为判对。

## 4. 资格规则对照（训练 vs 线上）

| 条件 | 训练真值 | 线上栏目采用 |
| --- | --- | --- |
| 临时意见 temporary | 排除 | 不支撑发布 |
| 被撤销定稿（当前时点） | 排除 | 已发布项标 evidence_lost 并停用 |
| 被撤销定稿（撤销前的历史评估） | 按时点可知纳入 | 历史模型表现按当时定稿解释 |
| 不在 valid_specs 的规格 | 排除 | 排除 |
| 印刷组不一致 | 排除 | 排除 |
| 规格已被同家族新版本取代（分版） | 文字标签仍可训练 | 不可继续借用旧结论 |
| 模型自身输出 | 不是真值，不能自我举证 | 必须另有正式定稿支撑才可发布 |

## 5. 状态文件与交叉核对

```
local-state/
  models/<version>/{pipeline.joblib, metadata.json}   # 模型与训练资格审计
  evaluations/<version>.json                          # 完整评估与阈值扫描
  editorial/decisions_<version>.jsonl                 # 逐 规格×类 决定与依据
  editorial/apply_<version>.json                      # 决定汇总与 IPW 估计
  editorial/submissions.jsonl                         # 编辑新提交/撤销覆盖记录
  queues/queue_<week>.json、assignments_<week>.jsonl  # 队列与权限分配
  audit/selections.jsonl                              # 抽核实际选中方式与概率
  adoptions/adoptions.jsonl                          # 采用状态（active/evidence_lost/rolled_back）
  snapshots/<snapshot>.json                           # 发布前快照，支持回退
  imports/<batch>/                                    # 外来文件原样留档与暂存
```

`reconcile` 同时核对：采用项是否都有模型输出、active 采用是否仍被模型放行、
active 采用的证据是否仍是合格定稿、每个产品是否都有模型输出。

## 6. 合成数据上的已知现象（运行时可见，不是缺陷）

- 72 条 `sugar_g=null` 的记录在“总糖含量声称/无法判断”两类上一律 `hold`。
- 历史 `publication_records` 中 5 条引用临时意见、2 条引用已撤销意见、
  30 条落在已分版规格上，`reconcile` 会逐条给出发现。
- 渠道 `ch3` 没有自身人审；其少量标签来自跨家族共享意见，评估视图分别标注。
