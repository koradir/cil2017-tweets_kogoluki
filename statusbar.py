# -*- coding: utf-8 -*-
"""
creates a console statusbar
"""

def status_update(current, top, label="Progress"):
    workdone = current/top
    print("\r{0:s}: [{1:30s}] {2:.1f}%".format(label,'#' * int(workdone * 30), workdone*100), end="", flush=True)
    if workdone == 1:
        print()