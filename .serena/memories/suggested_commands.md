# 常用命令（Windows PowerShell）

- 查看目录：`Get-ChildItem`
- 递归查找文件：`Get-ChildItem -Recurse -Filter <pattern>`
- 文本搜索：`Get-ChildItem -Recurse | Select-String -Pattern "<text>"`
- Git 状态：`git status`
- Git 差异：`git diff`

# 项目命令

- 未在 README/TODO 中发现测试、格式化或运行入口的明确命令。
- 入口脚本似乎为 `evaluate.py`（需确认具体参数与运行方式）。
