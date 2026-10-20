"""一次性校验：板端 main_tracking 二进制里，旧 OSD 的“两空格”字面量应已消失，
   新口径的“一空格”字面量应存在。
   做法：把校验脚本 SFTP 到板端 /tmp 执行，避免多层引号转义。
"""
import paramiko

HOST, USER, PWD = "192.168.1.104", "topeet", "topeet"
BIN = "/home/topeet/code/gatekeeper_V4/build/main_tracking"
REMOTE_PY = "/tmp/_chk_disp_literal.py"

REMOTE_SRC = f'''
data = open({BIN!r}, "rb").read()
tests = [
    ("旧 OSD \\"  OUT:\\"   (两空格, 期望 0)", b"  OUT:"),
    ("旧 OSD \\"  RETRO:\\" (两空格, 期望 0)", b"  RETRO:"),
    ("新 \\"OUT:\\"          (一空格, 期望 >0)", b"OUT:"),
    ("新 \\" RETRO:\\"       (一空格, 期望 >0)", b" RETRO:"),
    ("新 \\"IN:\\"                       (期望 >0)", b"IN:"),
]
for name, pat in tests:
    print("%-34s -> %d 次" % (name, data.count(pat)))
'''

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username=USER, password=PWD, timeout=15)

sftp = cli.open_sftp()
with sftp.open(REMOTE_PY, "w") as f:
    f.write(REMOTE_SRC)
sftp.close()

_, out, err = cli.exec_command(f"bash -lc \"python3 {REMOTE_PY}; rm -f {REMOTE_PY}\"", timeout=30)
print(out.read().decode("utf-8", "replace"), end="")
e = err.read().decode("utf-8", "replace")
if e.strip():
    print("STDERR:", e)
cli.close()
