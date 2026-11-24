# 食品声明自动分类的人审学习与栏目采用

本项目资料使用本地合成记录，不含真实账号或个人资料。现有工具读取业务资料并检查交接格式，不对资料作业务判决。

运行环境为 Python 3.11。`requirements.txt` 固定直接依赖版本，`requirements.lock.txt` 固定完整依赖。建立虚拟环境后使用 `python3 -m pip install -r requirements.lock.txt` 安装；离线运行时使用已准备的依赖环境。

在项目根目录执行 `python3 -m materials` 查看资料清单，执行 `python3 -m unittest discover -s tests -v` 校验字段类型、编号和跨表引用。`python3 -m compileall -q materials tests claimdesk` 检查现有 Python 入口。上述检查只证明资料可读取，不代表业务应用的结果。

## 声明分类与栏目采用工作台（claimdesk）

可训练、比较、选择和应用的包装声明分类能力位于 `claimdesk`，覆盖四个声明类型
（未添加某种糖、总糖含量声称、无关口味宣传、无法判断），支持一条包装多类声明，
并把自动发布、排队复核、暂缺资料作为不同决定。训练资格与线上采用资格分别检查，
临时意见与被撤销意见不进真值；评估按家族隔离并按声明类与渠道汇报误放、漏收与覆盖。
编辑操作、模型输出、人审状态与栏目采用状态在 `local-state` 分表保存并可交叉核对。
完整操作步骤见 [工作台运行手册](docs/claimdesk-runbook.md)，常用入口：

```bash
python3 -m claimdesk train --candidate lr-rich --tag v1
python3 -m claimdesk evaluate --model <model_version>
python3 -m claimdesk apply --model <model_version>
python3 -m claimdesk queue --model <model_version> --week 2026-W48
python3 -m claimdesk reconcile --model <model_version>
python3 -m uvicorn claimdesk.api:app   # HTTP 入口，路由见 /docs
```

数据字典见 [资料协议](docs/data-contract.md)，机器可读定义为 `data/schema.json`。每张表对应同名 UTF-8 JSONL 文件，每行一条原始记录。时点使用含时区的 ISO 8601，日期采用 YYYY-MM-DD。结束边界不包含边界时点；空值表示协议说明的未知或不适用，不作默认数值。

资料读取入口位于 `materials`，交接格式检查位于 `tests`。本地运行状态可以写入 `local-state`；原始资料保持可追踪。现有资料不包含模型参数、业务算法或预先计算的分析结论。
