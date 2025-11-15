# app_tui.py (终极解决方案 v3: 健壮的日志轮询)
# -*- coding: utf-8 -*-

import sys
import os
import time
import subprocess
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Checkbox, DirectoryTree, Footer, Header, Log, RadioSet, Static

# --- 定义工作目录 ---
UPLOADS_DIR = Path("uploads")

class AutoReviewerApp(App):
    """一个用于自动化论文审稿的终端图形界面应用"""

    CSS_PATH = "tui.css"
    BINDINGS = [("ctrl+c", "quit", "退出应用")]

    class LogMessage(Message):
        def __init__(self, line: str) -> None:
            self.line = line
            super().__init__()

    class ProcessDone(Message):
        pass

    def __init__(self):
        super().__init__()
        self.selected_pdf_path: Path | None = None
        self.selected_tier: str = "CCF-B"
        self.force_rerun: bool = False
        UPLOADS_DIR.mkdir(exist_ok=True)

    def compose(self) -> ComposeResult:
        yield Header(name="🎓 Auto-Reviewer 终端控制台")
        with Horizontal(id="main-container"):
            with Vertical(id="control-panel"):
                yield Static("1. 从下方选择一个PDF文件:", classes="label")
                yield DirectoryTree(UPLOADS_DIR, id="file-tree")
                yield Static(id="selected-file-label")
                yield Static("\n2. 选择目标会议等级:", classes="label")
                yield RadioSet("CCF-A", "CCF-B", "CCF-C", id="tier-radioset")
                yield Checkbox("强制重新运行所有步骤", id="force-checkbox")
                yield Button("开始审稿", variant="primary", id="start-button")
            yield Log(id="log-view", auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#tier-radioset")._nodes[1].value = True
        self.query_one(Log).write_line("--- 欢迎使用Auto-Reviewer ---")
        self.query_one(Log).write_line(f"请将PDF文件放入 '{UPLOADS_DIR}' 文件夹, 然后在此处选择。")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        if str(event.path).lower().endswith(".pdf"):
            self.selected_pdf_path = event.path
            self.query_one("#selected-file-label").update(f"已选择: [bold green]{event.path.name}[/]")
        else:
            self.selected_pdf_path = None
            self.query_one("#selected-file-label").update("[bold red]请选择一个.pdf文件[/]")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self.selected_tier = event.pressed.label.plain

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        self.force_rerun = event.value

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-button":
            if not self.selected_pdf_path:
                self.query_one(Log).write_line("[bold red]错误: 请先在左侧文件列表中选择一个PDF文件！[/]")
                return
            event.button.disabled = True
            log = self.query_one(Log)
            log.clear()
            log.write_line("🚀 审稿流水线已启动...")
            log.write_line("="*60)
            self.run_worker(self.run_and_tail_process, thread=True)

    def run_and_tail_process(self) -> None:
        """
        启动 main.py 并将其输出重定向到日志文件,
        然后监视这个日志文件直到进程结束。
        """
        paper_name = self.selected_pdf_path.stem
        log_file_path = Path(f"{paper_name}_review_session.log")

        command = [
            sys.executable, "-u", "main.py",
            "--pdf", str(self.selected_pdf_path),
            "--tier", self.selected_tier,
        ]
        if self.force_rerun:
            command.append("--force")

        process = None  # 先声明 process 变量
        try:
            with open(log_file_path, "wb") as log_file:
                process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)

            # 调用全新的、健壮的日志读取函数
            self.tail_log_file_robustly(log_file_path, process)

        except Exception as e:
            self.post_message(self.LogMessage(f"\n❌ 启动进程时发生致命错误: {e}\n"))
        finally:
            self.post_message(self.ProcessDone())

    def tail_log_file_robustly(self, log_path: Path, process: subprocess.Popen):
        """
        健壮的日志读取逻辑，消除竞态条件。
        """
        # 等待文件肯定被创建
        while not log_path.exists():
            time.sleep(0.1)

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    # 尝试读取一行
                    line = f.readline()
                    
                    if line:
                        # 如果读到了内容，就发送给UI，然后继续下一轮循环
                        self.post_message(self.LogMessage(line.strip()))
                        continue
                    
                    # 如果没读到内容，我们再检查进程是否已经结束
                    if process.poll() is not None:
                        # 进程已结束，并且文件也没新内容了，可以跳出循环
                        break
                    
                    # 进程还在运行，只是暂时没新日志，那就等一小会儿
                    time.sleep(0.1)

        except Exception as e:
            self.post_message(self.LogMessage(f"\n❌ 监视日志文件时出错: {e}\n"))

    def on_log_message(self, message: LogMessage) -> None:
        self.query_one(Log).write_line(message.line)

    def on_process_done(self) -> None:
        self.query_one(Log).write_line("="*60)
        self.query_one(Log).write_line("🎉🎉🎉 流水线执行完毕！🎉🎉🎉")
        # 任务结束后，可以选择性地删除临时日志文件
        paper_name = self.selected_pdf_path.stem
        log_file_to_remove = Path(f"{paper_name}_review_session.log")
        if log_file_to_remove.exists():
            # log_file_to_remove.unlink() # 如果您想自动删除，取消这行注释
            self.query_one(Log).write_line(f"ℹ️  本次运行的详细日志已保存到: {log_file_to_remove}")

        self.query_one("#start-button").disabled = False

if __name__ == "__main__":
    app = AutoReviewerApp()
    app.run()