# P10 私有领域 Adapter 配置目录

这里仅保存不含密钥的模板。真实 manifest、数据集和 Adapter 应放在 D 盘独立私有目录，并
通过 `MINIGPT_DOMAIN_ROOT` 指向；不要把私有数据、访问 Token 或 Adapter 权重提交到公开仓库。

领域能力默认关闭。若未来启用成人领域，必须使用 `restricted`、`minimum_age=18`、同意证明
和全员成年人证明，并在训练前运行 `scripts/validate_domain_dataset.py`。涉及未成年人、强迫、
非自愿或来源/授权不明的数据会被门禁拒绝。

启用后的 Adapter 只允许请求级使用：在 `/chat` 或 `/agent/chat` 的 JSON 中同时传入
`domain_id` 和 `domain_access_token`。服务会在单个串行事务内激活、推理并恢复默认模型；
持久化 `/activate` 与 `/deactivate` 接口已禁用。
