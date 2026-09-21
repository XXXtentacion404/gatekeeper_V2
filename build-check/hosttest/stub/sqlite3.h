// 宿主侧「语法检查专用」桩头文件（见 HCNetSDK.h 说明）。
#ifndef HOSTTEST_STUB_SQLITE3_H
#define HOSTTEST_STUB_SQLITE3_H

#define SQLITE_OPEN_READWRITE 0x00000002
#define SQLITE_OPEN_CREATE    0x00000004
#define SQLITE_OPEN_FULLMUTEX 0x00010000
#define SQLITE_OK 0

typedef struct sqlite3 sqlite3;
typedef int (*sqlite3_callback)(void*, int, char**, char**);

inline int sqlite3_open_v2(const char*, sqlite3** ppDb, int, const char*) {
    if (ppDb) *ppDb = nullptr;
    return SQLITE_OK;
}
inline int sqlite3_exec(sqlite3*, const char*, sqlite3_callback, void*, char**) {
    return SQLITE_OK;
}
inline int sqlite3_close(sqlite3*) { return SQLITE_OK; }

#endif
