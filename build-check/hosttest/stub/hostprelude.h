// 宿主侧语法检查前置声明：MinGW 在 -std 严格模式下不声明 POSIX setenv()，
// 板端(Linux glibc)由 <stdlib.h> 提供，故此处仅为通过宿主语法检查。
#pragma once
#include <cstdlib>
extern "C" int setenv(const char* name, const char* value, int overwrite);
