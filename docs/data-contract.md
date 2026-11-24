# 业务资料协议

数据仅用于解释包装声明，不是法规符合性结论。声明类型可以同时出现；无法判断不是无糖。印刷复用组只说明文字来源，编辑定稿是否可复用取决于valid_specs及有效期。临时意见和被撤销意见不是正式定稿。

所有数字、身份和业务来件均为合成。记录的稳定编号只在对应表内唯一，跨表关联以明确字段为准。布尔值为原来源明确回报，空字符串引用表示没有已知对应。金额如有均用整数分，比例为0至1，其他单位在字段中注明。

## products

包装与市场原始记录。

| 字段 | 格式 | 含义 |
| --- | --- | --- |
| id | string | 记录的稳定编号 |
| family | string | 稳定商品家族 |
| version | number | 家族内包装版本 |
| valid_from | string | 包装开始适用日期 |
| channel | string | 销售渠道 |
| text | string | 包装原文 |
| ingredients | string | 配料原文 |
| basis | string | 营养表的基准 |
| basis_g | number | 对应基准克数 |
| sugar_g | nullable_number | 该基准下总糖；null表示缺项 |
| units_sold | number | 对应窗口的实际销量 |
| known_at | string | 收到包装资料时间 |
| print_group | string | 同一包装印刷文字的复用组，仍须按意见实际适用规格核实 |

## reviews

编辑意见原件，可带有偏选择与撤销。

| 字段 | 格式 | 含义 |
| --- | --- | --- |
| id | string | 记录的稳定编号 |
| product_id | string | 被判读包装 |
| reviewer | string | 编辑身份键 |
| classes | array | 一个或多个声明类型 |
| status | string | temporary为临时意见，final为定稿 |
| selected_by | string | random_audit为概率抽核，suspect_queue为定向疑点 |
| selection_probability | nullable_number | 有效抽核选中概率，定向记录为空 |
| valid_specs | array | 意见明确支持的规格版本 |
| known_at | string | 意见收到时间 |
| withdrawn_at | nullable_string | 该意见被撤销的时点 |

引用关系：product_id 对应 products.id。

## editors

编辑的业务权限与时间。

| 字段 | 格式 | 含义 |
| --- | --- | --- |
| id | string | 记录的稳定编号 |
| classes | array | 可定稿的声明类型 |
| weekly_minutes | number | 每周可用分钟 |
| can_finalize | boolean | 能否给出正式定稿 |

## publication_records

栏目已有的采用关系。

| 字段 | 格式 | 含义 |
| --- | --- | --- |
| id | string | 记录的稳定编号 |
| product_id | string | 采用的规格版本 |
| review_id | string | 当时引用意见 |
| column | string | 栏目名称 |
| adopted_at | string | 采用时点 |
| model_version | string | 提出采用的原模型名称；不包含模型实现 |

引用关系：product_id 对应 products.id；review_id 对应 reviews.id。

## cost_policies

业务方提交的处理代价及容限。

| 字段 | 格式 | 含义 |
| --- | --- | --- |
| id | string | 记录的稳定编号 |
| class_name | string | 声明类型 |
| false_release_cost | number | 误放相对业务代价 |
| miss_cost | number | 漏收相对代价 |
| maximum_error | number | 自动采用的错误比例容限 |
| review_minutes | number | 常规复核所需分钟 |
