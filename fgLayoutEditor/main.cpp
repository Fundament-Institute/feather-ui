// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgLayoutEditor.h"
#include <string>
#include <fstream>
#include "bss-util/defines.h"

#ifdef BSS_PLATFORM_WIN32 //Windows function
#include "bss-util/win32_includes.h"
#include <Shlwapi.h>

template<class R, class T, size_t(*F)(const T* BSS_RESTRICT, ptrdiff_t, R* BSS_RESTRICT, size_t)>
inline std::basic_string<R> ConvUTF(const T* s)
{
  std::basic_string<R> wstr;
  wstr.resize(F(s, -1, 0, 0));
  wstr.resize(F(s, -1, const_cast<R*>(wstr.data()), wstr.capacity()));
  return wstr;
}

int Listdir(const char* cdir, bool(*fn)(const char*), char flags, const char* filter = "*")
{
  WIN32_FIND_DATAW ffd;
  HANDLE hdir = INVALID_HANDLE_VALUE;

  std::wstring dir;
  dir.resize(MAX_PATH);
  std::wstring wfilter = ConvUTF<wchar_t, char, fgUTF8toUTF16>(filter);

  PathCanonicalizeW(const_cast<wchar_t*>(dir.data()), ConvUTF<wchar_t, char, fgUTF8toUTF16>(cdir).c_str());
  dir.resize(wcslen(dir.data()));

  if(dir[dir.length() - 1] == '/') const_cast<wchar_t*>(dir.data())[dir.length() - 1] = '\\';
  if(dir[dir.length() - 1] != '\\') dir += '\\';
  hdir = FindFirstFileW((dir + wfilter).c_str(), &ffd);

  while(hdir != 0 && hdir != INVALID_HANDLE_VALUE)
  {
    if(WCSICMP(ffd.cFileName, L".") != 0 && WCSICMP(ffd.cFileName, L"..") != 0)
    {
      std::wstring file = dir + ffd.cFileName;
      std::string fldir = ConvUTF<char, wchar_t, fgUTF16toUTF8>(file.c_str());

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

void LoadEmbeddedResource(int id, int type, DWORD& size, const char*& data)
{
  HMODULE handle = ::GetModuleHandle(NULL);
  HRSRC rc = ::FindResource(handle, MAKEINTRESOURCE(id), MAKEINTRESOURCE(type));
  HGLOBAL rcData = ::LoadResource(handle, rc);
  size = ::SizeofResource(handle, rc);
  data = static_cast<const char*>(::LockResource(rcData));
}

#else //Linux function
int Listdir(const wchar_t* cdir, bool(*fn)(const wchar_t*), char flags)
{
  DIR* srcdir = opendir(path);

  if(!srcdir)
    return -1;

  struct stat st;
  struct dirent* dent;
  Str dir(path);
  if(dir[dir.length() - 1] == '\\') dir.UnsafeString()[dir.length() - 1] = '/';
  if(dir[dir.length() - 1] != '/') dir += '/';

  while((dent = readdir(srcdir)) != 0)
  {
    if(strcmp(dent->d_name, ".") == 0 || strcmp(dent->d_name, "..") == 0)
      continue;
    if(fstatat(dirfd(srcdir), dent->d_name, &st, 0) == 0)
    {
      Str sdir(dir + dent->d_name);
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
  rootpath = ConvUTF<char, wchar_t, fgUTF16toUTF8>(commands.c_str());
#else
  rootpath = ".";
#endif

#ifdef BSS_DEBUG
  Listdir(rootpath.c_str(), attemptbackend, 0, "*_d.dll");
#else
  Listdir(rootpath.c_str(), attemptbackend, 0, "*.dll");
#endif

  if(!fgSingleton())
    return 1;

  fgRegisterFunction("menu_file", fgLayoutEditor::MenuFile);
  fgRegisterFunction("menu_recent", fgLayoutEditor::MenuRecent);
  fgRegisterFunction("menu_edit", fgLayoutEditor::MenuEdit);
  fgRegisterFunction("menu_view", fgLayoutEditor::MenuView);
  fgRegisterFunction("menu_help", fgLayoutEditor::MenuHelp);

  EditorSettings settings = {
    false,
    false,
    true,
    true,
    false,
    true,
    false
  };

  std::ifstream settingsfile("fgLayoutEditor.toml", std::ios_base::binary|std::ios_base::in);

  if(settingsfile.good())
  {
    bss::Serializer<bss::TOMLEngine> serializer;
    serializer.Parse(settings, settingsfile, "editor");
  }
  else
  {
    std::ofstream settingsout("fgLayoutEditor.toml", std::ios_base::binary | std::ios_base::out);
    bss::Serializer<bss::TOMLEngine> serializer;
    serializer.Serialize(settings, settingsout, "editor");
  }

  fgLayout layout;
  fgLayout_Init(&layout);
#ifdef BSS_PLATFORM_WIN32
  const char* data;
  DWORD sz;
  LoadEmbeddedResource(201, 256, sz, data);
  fgLayout_LoadXML(&layout, data, sz);
#else
  fgLayout_LoadFileXML(&layout, "../media/editor/editor.xml");
#endif
  {
    fgLayoutEditor editor(&layout, settings);
    editor.OpenLayout(&layout);
    while(fgSingleton()->backend.fgProcessMessages());
  }
  fgLayout_Destroy(&layout);
  fgUnloadBackend();
}

struct HINSTANCE__;

// WinMain function, simply a catcher that calls the main function
int __stdcall WinMain(struct HINSTANCE__* hInstance, struct HINSTANCE__* hPrevInstance, char* lpCmdLine, int nShowCmd)
{
  main(0, (char**)hInstance);
}