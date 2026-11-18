"""读取配置，依次执行 STEP 选面、视点生成和路径排序。

每个零件的输入 STEP 通过 --step 指定，结果写入 out/<零件名>_<时间戳>/ 下的
extraction、viewpoints、path 三个目录，运行日志为该目录下的 run.log。

打包为 exe 后，config.yaml、data/input/、out/ 与 exe 放在同一目录；
开发时则相对于项目根目录。"""

from pathlib import Path
import argparse
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

import yaml

from path_planner.planner import plan_path
from vp_planner import prepare_surface, ViewpointPlanner
from vp_planner.file_io import export_surface, export_planning_result
from vp_planner.visualization import preview_surface, export_viewpoint_visualization


def _app_dir() -> Path:
    """程序根目录：打包为 exe 后取 exe 所在目录，开发时取项目目录。"""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


PROJECT_DIR = _app_dir()
DATA_DIR = PROJECT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_ROOT = PROJECT_DIR / "out"


class _Tee:
    """把一个输出流同时写到控制台和日志文件。"""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def load_config(config_path: Path = PROJECT_DIR / "config.yaml") -> dict:
    """读取算法参数配置。输入和输出路径由程序目录结构统一管理。"""
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式无效: {config_path}")
    return config


def _run(config: dict, args, step_file: Path, run_dir: Path) -> int:
    extraction_dir = run_dir / "extraction"
    viewpoint_dir = run_dir / "viewpoints"
    path_dir = run_dir / "path"

    try:
        surface = prepare_surface(step_file, config['step_preprocessing'])
        export_surface(surface, extraction_dir=extraction_dir)
        preview_surface(surface, extraction_dir / 'target_surface_preview.png')
    except Exception as exc:
        print(f"STEP 预处理失败: {exc}")
        return 1

    if args.extract_only:
        print('表面提取完成，未运行视点规划。')
        return 0

    try:
        result = ViewpointPlanner(config['viewpoint']).plan(surface)
        export_planning_result(result, viewpoint_dir)
        export_viewpoint_visualization(surface, result.viewpoints, viewpoint_dir / 'VP_visualization.obj')
        print(f'覆盖率 {result.coverage_rate:.2%}，停止原因 {result.stop_reason}')
        if not result.viewpoints:
            print('没有生成有效视点')
            return 1
        viewpoint_file = viewpoint_dir / 'VP.txt'
    except Exception as exc:
        print(f'视点规划失败: {exc}')
        return 1

    print("\n" + "=" * 60)
    print("视点规划完成！")
    print(f"局部坐标系视点: {viewpoint_file}")
    print(f"视点预览: {viewpoint_dir / 'VP_visualization.obj'}")
    try:
        if not plan_path(config['path_planning'], viewpoint_file, surface.local_to_world, path_dir):
            return 1
    except Exception as exc:
        print(f"路径规划失败: {exc}")
        return 1
    print(f"STEP 坐标系路径: {path_dir / 'Path_base_world.txt'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='STEP 表面提取、视点生成和路径排序')
    parser.add_argument('--step', type=Path, default=None,
                        help='输入 STEP 文件路径（未指定时交互式输入）')
    parser.add_argument('--config', type=Path, default=None,
                        help='配置文件路径（默认 exe/项目同目录的 config.yaml）')
    parser.add_argument('--extract-only', action='store_true', help='只提取目标表面并输出预览')
    args = parser.parse_args()

    # 确保输入目录存在，用户拿到 exe 后直接往 data/input 放 STEP 文件即可。
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    step_file = args.step
    if step_file is None:
        # 双击运行时交互式输入路径，不再默认读取 data/input/Part.STEP。
        try:
            answer = input("请输入 STEP 文件路径（直接回车退出）: ").strip().strip('"')
        except EOFError:
            answer = ''
        if answer:
            step_file = Path(answer)
        else:
            return 2

    if not step_file.is_file():
        print(f"缺少 STEP 输入文件: {step_file}")
        return 2

    config = load_config(args.config or (PROJECT_DIR / "config.yaml"))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = OUTPUT_ROOT / f"{step_file.stem}_{timestamp}"
    for directory in (run_dir, run_dir / "extraction", run_dir / "viewpoints", run_dir / "path"):
        directory.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "run.log"
    with log_file.open("w", encoding="utf-8") as stream:
        stream.write(f"Path Planner run {datetime.now().isoformat(timespec='seconds')}\n")
        stream.write(f"输入 STEP: {step_file}\n")
        tee_out = _Tee(sys.stdout, stream)
        tee_err = _Tee(sys.stderr, stream)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            return _run(config, args, step_file, run_dir)


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:
        if getattr(sys, 'frozen', False):
            print(f"\n程序运行出错: {exc}")
            code = 1
        else:
            raise

    if getattr(sys, 'frozen', False) and sys.stdin is not None and sys.stdin.isatty():
        # 双击 exe 运行时暂停，避免窗口一闪而过看不到结果。
        try:
            input("\n程序已结束，按回车键退出...")
        except (EOFError, OSError):
            pass

    raise SystemExit(code)
