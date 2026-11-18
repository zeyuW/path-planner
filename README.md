# Path Planner

从 STEP 零件自动完成 **选面 → 视点生成 → 路径规划**，输出相机扫描视点与访问顺序，供机械臂或扫描系统使用。

- 输入：一个 STEP 文件（板类表面）
- 输出：提取的待测表面、相机视点、扫描路径（含世界坐标）

适用于可投影为单个连续区域的板类表面；孔壁、侧壁或折叠/复杂零件需另行指定目标区域。

> 想了解选面、视点、路径三阶段的算法原理，请阅读 [原理说明](docs/原理说明.md)。

## 环境要求

- Python 3.11+
- uv（依赖管理）

## 本地运行

结果写入 `out/`。

```powershell
uv sync                 # 首次运行安装依赖
uv run python main.py --step data\input\Part.STEP   # 运行完整流程（指定 STEP 文件）
```

常用参数：

```powershell
uv run python main.py --step D:\parts\PartA.step   # 指定 STEP 文件
uv run python main.py --config D:\config.yaml      # 指定配置文件
uv run python main.py --extract-only               # 只提取表面并生成预览
```

不指定 `--step` 时（如双击运行），程序会提示你输入 STEP 文件路径。算法参数集中在 `config.yaml`。

## 自动化脚本（scripts/）

| 脚本 | 作用 |
|------|------|
| `.\scripts\dev.bat` | 本地开发验证：同步依赖 → 单元测试 → 跑完整流程 |
| `.\scripts\prod.bat` | 打包生成 exe（保留之前的 out 结果） |

## 打包发布

```powershell
.\scripts\prod.bat   # 打包生成 exe
```

打包产物在 `dist\PathPlanner\`，内含 `PathPlanner.exe` + `config.yaml`。之后手动运行 exe：

```powershell
dist\PathPlanner\PathPlanner.exe                            # 双击也行，会提示输入 STEP 路径
dist\PathPlanner\PathPlanner.exe --step D:\parts\PartA.step   # 命令行指定 STEP
```

发布时把 `dist\PathPlanner\` 这个目录整体交给用户即可；`data\input\` 和 `out\` 会在首次运行时自动创建。重新打包时，之前的 `out\` 结果会被保留。

> 提示：OCP、matplotlib、scikit-learn 体积较大，exe 较大、打包耗时较长。

## 产出物

每次运行生成一个目录 `out/<零件名>_<时间戳>/`，零件名取 STEP 文件名，不同零件互不覆盖：

> **产物位置取决于运行方式**：
>
> - 本地运行源代码 → 项目根目录下的 `out/`
> - 打包后运行 exe → exe 所在目录（`dist\PathPlanner\`）下的 `out/`

```
out/Part_20261117_174008/
├── extraction/    # 选面结果
├── viewpoints/    # 视点结果
├── path/          # 路径结果
└── run.log        # 运行日志
```

| 文件 | 说明 |
|------|------|
| `path/Path_base_world.txt` | **最终路径（世界坐标）**，机械臂直接使用 |
| `path/Path_base_pannel.txt` | 局部坐标路径（视点访问顺序） |
| `path/Path_visualization.png` | 路径预览图 |
| `viewpoints/VP.txt` | 视点列表（局部坐标） |
| `viewpoints/VP_visualization.obj` | 视点位置与朝向的 3D 可视化 |
| `viewpoints/planning_report.json` | 覆盖率、停止原因等 |
| `extraction/target_surface.step` | 提取的 CAD 面，可导入 CAD 核对 |
| `extraction/target_surface_preview.png` | 选面二维预览 |
| `extraction/selection.json` | 选面报告（含局部↔世界坐标变换矩阵） |

## 测试

```powershell
uv run python -m unittest discover -s tests -v
```

## 目录结构

- `main.py`：入口，串联选面 → 视点 → 路径
- `src/vp_planner/`：表面提取与视点规划
- `src/path_planner/`：路径规划（蚁群算法）
- `scripts/`：自动化脚本（dev / prod）
- `data/input/`：输入 STEP 文件
- `config.yaml`：算法与可视化参数
