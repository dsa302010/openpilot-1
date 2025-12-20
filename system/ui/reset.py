#!/usr/bin/env python3
import os
import sys
import threading
import time
from enum import IntEnum

import pyray as rl

from openpilot.system.hardware import PC
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.label import gui_label, gui_text_box

NVME = "/dev/nvme0n1"
USERDATA = "/dev/disk/by-partlabel/userdata"
TIMEOUT = 3*60


class ResetMode(IntEnum):
  USER_RESET = 0  # user initiated a factory reset from openpilot
  RECOVER = 1     # userdata is corrupt for some reason, give a chance to recover
  FORMAT = 2      # finish up a factory reset from a tool that doesn't flash an empty partition to userdata


class ResetState(IntEnum):
  NONE = 0
  CONFIRM = 1
  RESETTING = 2
  FAILED = 3


class Reset(Widget):
  def __init__(self, mode):
    super().__init__()
    self._mode = mode
    self._previous_reset_state = None
    self._reset_state = ResetState.NONE
    self._cancel_button = Button("關閉", self._cancel_callback)
    self._confirm_button = Button("確認", self._confirm, button_style=ButtonStyle.PRIMARY)
    self._reboot_button = Button("重啟", lambda: os.system("sudo reboot"))
    self._render_status = True

  def _cancel_callback(self):
    self._render_status = False

  def _do_erase(self):
    if PC:
      return

    # Best effort to wipe NVME
    os.system(f"sudo umount {NVME}")
    os.system(f"yes | sudo mkfs.ext4 {NVME}")

    # Removing data and formatting
    rm = os.system("sudo rm -rf /data/*")
    os.system(f"sudo umount {USERDATA}")
    fmt = os.system(f"yes | sudo mkfs.ext4 {USERDATA}")

    if rm == 0 or fmt == 0:
      os.system("sudo reboot")
    else:
      self._reset_state = ResetState.FAILED

  def start_reset(self):
    self._reset_state = ResetState.RESETTING
    threading.Timer(0.1, self._do_erase).start()

  def _update_state(self):
    if self._reset_state != self._previous_reset_state:
      self._previous_reset_state = self._reset_state
      self._timeout_st = time.monotonic()
    elif self._reset_state != ResetState.RESETTING and (time.monotonic() - self._timeout_st) > TIMEOUT:
      exit(0)

  def _render(self, rect: rl.Rectangle):
    label_rect = rl.Rectangle(rect.x + 140, rect.y, rect.width - 280, 100)
    gui_label(label_rect, "System Reset", 100, font_weight=FontWeight.BOLD)

    text_rect = rl.Rectangle(rect.x + 140, rect.y + 140, rect.width - 280, rect.height - 90 - 100)
    gui_text_box(text_rect, self._get_body_text(), 90)

    button_height = 160
    button_spacing = 50
    button_top = rect.y + rect.height - button_height
    button_width = (rect.width - button_spacing) / 2.0

    if self._reset_state != ResetState.RESETTING:
      if self._mode == ResetMode.RECOVER:
        self._reboot_button.render(rl.Rectangle(rect.x, button_top, button_width, button_height))
      elif self._mode == ResetMode.USER_RESET:
        self._cancel_button.render(rl.Rectangle(rect.x, button_top, button_width, button_height))

      if self._reset_state != ResetState.FAILED:
        self._confirm_button.render(rl.Rectangle(rect.x + button_width + 50, button_top, button_width, button_height))
      else:
        self._reboot_button.render(rl.Rectangle(rect.x, button_top, rect.width, button_height))

    return self._render_status

  def _confirm(self):
    if self._reset_state == ResetState.CONFIRM:
      self.start_reset()
    else:
      self._reset_state = ResetState.CONFIRM

  def _get_body_text(self):
    if self._reset_state == ResetState.CONFIRM:
      return "您確定要重設裝置嗎？"
    if self._reset_state == ResetState.RESETTING:
      return 正在重設裝置...這可能需要一分鐘的時間。"
    if self._reset_state == ResetState.FAILED:
      return "重設失敗。請重新開機後再試。"
    if self._mode == ResetMode.RECOVER:
      return "無法掛載資料磁區。磁區可能已損壞。請按『確認』以清除並重設您的裝置。"
    return "系統重設已觸發。請按『確認』以清除所有內容與設定。按『取消』以繼續開機。"


def main():
  mode = ResetMode.USER_RESET
  if len(sys.argv) > 1:
    if sys.argv[1] == '--recover':
      mode = ResetMode.RECOVER
    elif sys.argv[1] == "--format":
      mode = ResetMode.FORMAT

  gui_app.init_window("System Reset", 20)
  reset = Reset(mode)

  if mode == ResetMode.FORMAT:
    reset.start_reset()

  for _ in gui_app.render():
    if not reset.render(rl.Rectangle(45, 200, gui_app.width - 90, gui_app.height - 245)):
      break


if __name__ == "__main__":
  main()
