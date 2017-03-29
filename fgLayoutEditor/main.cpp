// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayoutEditor.h"
#include <string>
#include "bss-util/bss_defines.h"

#ifdef BSS_PLATFORM_WIN32 //Windows function
#include "bss-util/bss_win32_includes.h"

int Listdir(const char* cdir, bool(*fn)(const char*), char flags, const char* filter = "*")
{
  WIN32_FIND_DATAW ffd;
  HANDLE hdir = INVALID_HANDLE_VALUE;

  std::wstring dir;
  dir.reserve(fgUTF8toUTF16(cdir, -1, 0, 0));
  dir.resize(fgUTF8toUTF16(cdir, -1, const_cast<wchar_t*>(dir.data()), dir.capacity()));

  if(dir[dir.length() - 1] == '/') const_cast<wchar_t*>(dir.data())[dir.length() - 1] = '\\';
  if(dir[dir.length() - 1] != '\\') dir += '\\';
  hdir = FindFirstFileW(dir + filter, &ffd);

  while(hdir != 0 && hdir != INVALID_HANDLE_VALUE)
  {
    if(WCSICMP(ffd.cFileName, L".") != 0 && WCSICMP(ffd.cFileName, L"..") != 0)
    {
      std::wstring file = dir + ffd.cFileName;
      std::string fldir;
      fldir.reserve(fgUTF16toUTF8(file.c_str(), -1, 0, 0));
      fldir.resize(fgUTF16toUTF8(file.c_str(), -1, const_cast<char*>(fldir.data()), fldir.capacity()));

      if(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
      {
        if(flags & 2) if(!(*fn)(fldir.c_str())) break;
        if(flags & 1) Listdir(fldir.c_str(), fn, flags);
      }
      else if(!(*fn)(fldir.c_str()))
        break;
    }
    if(FindNextFileW(hdir, &ffd) <= 0) break; //either we're done or it failed
  }

  if(hdir != 0)
    FindClose(hdir);
  return 0;
}
#else //Linux function
int Listdir(const wchar_t* cdir, bool(*fn)(const wchar_t*), char flags)
{
  DIR* srcdir = opendir(path);

  if(!srcdir)
    return -1;

  struct stat st;
  struct dirent* dent;
  cStr dir(path);
  if(dir[dir.length() - 1] == '\\') dir.UnsafeString()[dir.length() - 1] = '/';
  if(dir[dir.length() - 1] != '/') dir += '/';

  while((dent = readdir(srcdir)) != 0)
  {
    if(strcmp(dent->d_name, ".") == 0 || strcmp(dent->d_name, "..") == 0)
      continue;
    if(fstatat(dirfd(srcdir), dent->d_name, &st, 0) == 0)
    {
      cStr sdir(dir + dent->d_name);
      if(S_ISDIR(st.st_mode))
      {
        if(flags & 2) if(fn(sdir.c_str())) break;
        if(flags & 1) ListDir(sdir.c_str(), fn, flags);
      }
      else if(fn((dir + dent->d_name).c_str()))
        break;
    }
  }
  closedir(srcdir);
  return 0;
}
#endif

bool attemptbackend(const char* path)
{
  return !fgLoadBackend(path);
}

int main(int argc, char** argv)
{
  std::string rootpath;
#ifdef BSS_PLATFORM_WIN32
  std::wstring commands;
  commands.reserve(MAX_PATH);
  GetModuleFileNameW(0, const_cast<wchar_t*>(commands.data()), MAX_PATH);
  const_cast<wchar_t*>(commands.data())[wcsrchr(commands.c_str(), '\\') - commands.c_str() + 1] = '\0';
  rootpath.reserve(fgUTF16toUTF8(commands.c_str(), -1, 0, 0));
  rootpath.resize(fgUTF16toUTF8(commands.c_str(), -1, const_cast<char*>(rootpath.data()), rootpath.capacity()));
#else
  rootpath = ".";
#endif

  Listdir(rootpath.c_str(), attemptbackend, 0, "*.dll");

  if(!fgSingleton())
    return 1;

  fgLayoutEditor editor;
  fgLayoutEditor_Init(&editor);
  while(fgSingleton()->backend.fgProcessMessages());
  fgLayoutEditor_Destroy(&editor);

  fgUnloadBackend();
}

struct HINSTANCE__;

// WinMain function, simply a catcher that calls the main function
int __stdcall WinMain(struct HINSTANCE__* hInstance, struct HINSTANCE__* hPrevInstance, char* lpCmdLine, int nShowCmd)
{
  main(0, (char**)hInstance);
}