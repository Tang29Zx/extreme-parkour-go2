# AGENTS.md

## 项目说明

- 项目：Extreme Parkour Go2
- 技术栈：Python 3.8、Isaac Gym、legged_gym、rsl_rl、Docker
- 目标：在 Unitree Go2 上训练、回放和部署极限跑酷策略。

## 开发约定

- 沟通和验证记录使用中文；代码、标识符和代码注释使用英文。
- 修改保持最小并与已有 `go2` 任务隔离。
- 不覆盖 checkpoint、日志或用户已有改动。
- 不修改宿主机 CUDA、驱动或 Python 环境。
- 不读取或记录 secret、私钥、token 和 `.env` 内容。

## 常用验证

- Python 语法：`python -m py_compile <files>`
- 聚焦测试：`python -m unittest <test-module>`
- Docker CPU 回放从 `/home/tang/parkour` 使用
  `docker/extreme_go2/run_gui.sh`。

## Git

- 修改前后检查 `git status --short` 和聚焦 diff。
- 不自动提交或推送。
- 不使用 `git reset --hard`、`git clean` 或强制推送。

