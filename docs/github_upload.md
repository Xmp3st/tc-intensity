# 上传到 GitHub 的仓库管理方案

本文件说明如何把这个「北太平洋热带气旋强度识别框架」纳入 Git 管理并推送至 GitHub。
所有命令均在 WSL 内、项目根目录 `tc_intensity/` 下执行。

---

## 1. 仓库里有啥 / 没啥

| 入库 | 不入（.gitignore） | 原因 |
|---|---|---|
| 全部 `.py` 源码、`configs/*.yaml`、`README.md`、`requirements.txt` | `outputs/`、`*.ckpt`、`*.log` | 权重/日志体积大，且可由代码+数据复现 |
| `tests/` 冒烟测试 | `data/synthetic`、`data/raw`、`data/processed`、`*.png/jpg` | 图片数据通常很大，单独管理 |
| `.gitattributes`（统一 LF 换行） | | |

> 若确实要版本化图片/权重，请用 **Git LFS**（`git lfs track "*.png"` 等），勿直接入库大文件。

---

## 2. 初始化本地仓库并规划提交

采用 **Conventional Commits** 风格，按功能拆分提交，便于 review 与回滚。
> 本仓库已完成 `git init` 与首轮提交；下面是从零开始的参考步骤（已初始化可跳过 `git init`）。

```bash
cd /home/yzm/tf/tc_intensity      # 进入项目根（换成你的项目目录）
git init                           # 仅首次需要
git add .gitignore .gitattributes requirements.txt README.md
git commit -m "chore: 初始化项目骨架与依赖说明"

git add configs tcintens/utils/config.py
git commit -m "feat(config): 增加 YAML 层级配置与 --set 命令行覆盖"

git add tcintens/data
git commit -m "feat(data): 增加 CSV/文件夹双模式数据集与可配置增强"

git add tcintens/models
git commit -m "feat(models): 增加可插拔 torchvision 骨干与强度预测头"

git add tcintens/engine tcintens/utils/{logger,metrics,misc}.py
git commit -m "feat(engine): 训练/评估/推理引擎（早停、Top-K 权重、AMP）"

git add scripts tests
git commit -m "feat(scripts): 合成数据生成与真实数据整理工具 + 冒烟测试"

git add docs
git commit -m "docs: 补充 README 使用说明与 GitHub 上传方案"
```

---

## 3. 在 GitHub 上建远端仓库

### 方式 A：网页（推荐，最直观）
1. 登录 GitHub → **New repository**
2. 仓库名建议：`tc-intensity`（或 `nwp-typhoon-intensity`）
3. **不要**勾选 "Add a README" / ".gitignore"（本地已有，避免冲突）
4. 创建后复制 HTTPS/SSH 地址，例如 `git@github.com:<you>/tc-intensity.git`

### 方式 B：GitHub CLI（若已 `gh auth login`）
```bash
gh repo create tc-intensity --private --description "NP tropical cyclone intensity estimation (PyTorch)" --source . --remote origin --push
```
> 该命令会一次性完成「建仓 + 关联 + 首次推送」。

---

## 4. 关联远端并推送

```bash
git remote add origin git@github.com:<you>/tc-intensity.git   # 或用 https
git branch -M main
git push -u origin main
```

后续日常提交：
```bash
git push
```

---

## 5. 分支 / 标签 / 发布建议

- **长期分支策略**：`main` 保持稳定；新功能开 `feature/xxx` 分支，PR 合并回 `main`。
- **打标签发布**：当框架可用时打一个版本标签
  ```bash
  git tag -a v0.1.0 -m "首个可用版本：分类/回归双任务 + 合成数据冒烟测试"
  git push origin v0.1.0
  ```
- **GitHub Releases**：可在 Release 里附「合成数据说明」与示例权重（小文件），大权重仍建议用 LFS 或外部云存储。

---

## 6. 可选：CI 自动跑冒烟测试

在项目根加 `.github/workflows/smoke.yml`，每次 push 用 CPU 跑 `tests/test_pipeline.py`，
确保重构不破坏端到端流程（GitHub Actions 默认有 Python，但需 `pip install -r requirements.txt`）。
