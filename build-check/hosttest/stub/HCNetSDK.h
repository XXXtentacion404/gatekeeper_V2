// ============================================================================
// 宿主侧「语法检查专用」桩头文件（不属于产品代码，仅用于在 x86 主机上对
// main_tracking.cpp 做 -fsyntax-only 检查；板端编译使用真实的海康 SDK）。
// 只补齐 main_tracking.cpp 实际用到的类型/函数；FRAME_INFO、T_YV12、
// STREAME_REALTIME 等由仓库内 include/LinuxPlayM4.h 提供。
// ============================================================================
#ifndef HOSTTEST_STUB_HCNETSDK_H
#define HOSTTEST_STUB_HCNETSDK_H

#include <cstdint>

typedef int32_t LONG;
typedef uint32_t DWORD;
typedef uint16_t WORD;
typedef uint8_t BYTE;
typedef int BOOL;
typedef void* HWND;

#ifndef CALLBACK
#define CALLBACK
#endif

#define NET_DVR_STREAMDATA 1

typedef struct {
    char sDeviceAddress[128];
    WORD wPort;
    char sUserName[64];
    char sPassword[64];
} NET_DVR_USER_LOGIN_INFO;

typedef struct {
    DWORD dwDeviceType;
} NET_DVR_DEVICEINFO_V40;

typedef struct {
    LONG lChannel;
    DWORD dwStreamType;
    DWORD dwLinkMode;
    DWORD bBlocked;
} NET_DVR_PREVIEWINFO;

typedef void (CALLBACK* REALDATACALLBACK)(LONG lRealHandle, DWORD dwDataType,
                                          BYTE* pBuffer, DWORD dwBufSize, void* pUser);

inline BOOL NET_DVR_Init() { return 1; }
inline BOOL NET_DVR_Cleanup() { return 1; }
inline BOOL NET_DVR_SetConnectTime(DWORD, DWORD) { return 1; }
inline LONG NET_DVR_Login_V40(NET_DVR_USER_LOGIN_INFO*, NET_DVR_DEVICEINFO_V40*) { return 1; }
inline BOOL NET_DVR_Logout(LONG) { return 1; }
inline LONG NET_DVR_RealPlay_V40(LONG, NET_DVR_PREVIEWINFO*, REALDATACALLBACK, void*) { return 1; }
inline BOOL NET_DVR_StopRealPlay(LONG) { return 1; }

#endif
