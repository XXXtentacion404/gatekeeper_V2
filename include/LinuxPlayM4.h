#ifndef __LINUX_PLAYM4_H__
#define __LINUX_PLAYM4_H__

/* 海康播放库Linux版头文件 - 简化版 */
#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned int PLAYM4_HWND;
typedef void * PLAYM4_HWNDEX;
typedef void * PLAYM4_HDC;

#define PLAYM4_API 
#define __stdcall
#ifndef CALLBACK
#define CALLBACK
#endif

#define STREAME_REALTIME 0
#define STREAME_FILE     1

#define T_AUDIO16	101
#define T_AUDIO8	100
#define T_UYVY		1
#define T_YV12		3
#define T_RGB32		7

typedef struct tagSystemTime {
    short wYear, wMonth, wDayOfWeek, wDay;
    short wHour, wMinute, wSecond, wMilliseconds;
} SYSTEMTIME;

typedef struct {
    int nWidth, nHeight, nStamp, nType, nFrameRate;
    unsigned int dwFrameNum;
} FRAME_INFO;

/* API 声明 */
int  PlayM4_GetPort(int* nPort);
int  PlayM4_FreePort(int nPort);
int  PlayM4_SetStreamOpenMode(int nPort, unsigned int nMode);
int  PlayM4_OpenStream(int nPort, unsigned char* pFileHeadBuf, unsigned int nSize, unsigned int nBufPoolSize);
int  PlayM4_CloseStream(int nPort);
int  PlayM4_InputData(int nPort, unsigned char* pBuf, unsigned int nSize);
int  PlayM4_Play(int nPort, PLAYM4_HWND hWnd);
int  PlayM4_Stop(int nPort);
int  PlayM4_SetDecCallBackMend(int nPort, void (CALLBACK* DecCBFun)(int nPort, char* pBuf, int nSize, FRAME_INFO* pFrameInfo, void* nUser, int nReserved2), void* nUser);
int  PlayM4_SetDecCallBackExMend(int nPort, void (CALLBACK* DecCBFun)(int nPort, char* pBuf, int nSize, FRAME_INFO* pFrameInfo, void* nUser, int nReserved2), char* pDest, int nDestSize, void* nUser);
int  PlayM4_GetLastError(int nPort);
unsigned int PlayM4_GetSdkVersion();

#ifdef __cplusplus
}
#endif

#endif // __LINUX_PLAYM4_H__
