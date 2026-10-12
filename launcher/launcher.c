/* CrispFrame 启动器：薄窗口壳，定位同目录 runtime\pythonw.exe 运行 app\gui.py。
 *
 * 构建发布版（无控制台）：
 *   vcvars64.bat && cl /O2 /W3 /DUNICODE /D_UNICODE launcher.c ^
 *       /link /SUBSYSTEM:WINDOWS /OUT:CrispFrame.exe
 * 构建调试版（带控制台，看 stderr）：把 /SUBSYSTEM:WINDOWS 换成 :CONSOLE，
 * 并将 wWinMain 改为 wmain（本文件底部有等价入口）。
 */
#include <windows.h>
#include <shellapi.h>
#include <stdio.h>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "shell32.lib")

static void fail(HWND parent, const wchar_t *msg) {
    MessageBoxW(parent, msg, L"CrispFrame", MB_ICONERROR);
}

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE prev, PWSTR cmdline, int show) {
    (void)inst; (void)prev; (void)cmdline; (void)show;

    wchar_t root[MAX_PATH];
    if (!GetModuleFileNameW(NULL, root, MAX_PATH)) { fail(NULL, L"GetModuleFileName 失败"); return 1; }
    wchar_t *slash = wcsrchr(root, L'\\');
    if (slash) *slash = 0;                                   /* root = exe 所在目录 */

    wchar_t py[MAX_PATH], script[MAX_PATH];
    _snwprintf(py, MAX_PATH, L"%s\\runtime\\pythonw.exe", root);
    _snwprintf(script, MAX_PATH, L"%s\\runtime\\app\\gui.py", root);

    if (GetFileAttributesW(py) == INVALID_FILE_ATTRIBUTES) {
        fail(NULL, L"未找到 runtime\\pythonw.exe —— 请确认解压完整（crash 排查见使用说明）");
        return 2;
    }
    if (GetFileAttributesW(script) == INVALID_FILE_ATTRIBUTES) {
        fail(NULL, L"未找到 runtime\\app\\gui.py —— 请确认解压完整");
        return 2;
    }

    wchar_t args[32768];
    _snwprintf(args, 32768, L"\"%s\" \"%s\"", py, script);

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    wchar_t cwd[MAX_PATH];
    _snwprintf(cwd, MAX_PATH, L"%s", root);
    if (!CreateProcessW(NULL, args, NULL, NULL, FALSE,
                        CREATE_UNICODE_ENVIRONMENT, NULL, cwd, &si, &pi)) {
        wchar_t buf[512];
        _snwprintf(buf, 512, L"启动 Python 运行时失败（错误码 %lu）", GetLastError());
        fail(NULL, buf);
        return 3;
    }
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    return (int)code;
}
