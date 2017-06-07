@ECHO OFF
md "..\Packages\feathergui"

RMDIR /S /Q "./Debug/"
RMDIR /S /Q "./Release/"
RMDIR /S /Q "./x64/"
RMDIR /S /Q "./Win32/"
RMDIR /S /Q "./bin/obj/"
RMDIR /S /Q "./bin32/obj/"
RMDIR /S /Q "./bindings/fgDotNet/x64/"
RMDIR /S /Q "./bindings/fgDotNet/Win32/"
RMDIR /S /Q "./examples/c/x64/"
RMDIR /S /Q "./examples/c/Win32/"
RMDIR /S /Q "./examples/cs/obj/"
RMDIR /S /Q "./examples/x64/"
RMDIR /S /Q "./examples/Win32/"
RMDIR /S /Q "./feathergui/x64/"
RMDIR /S /Q "./feathergui/Win32/"
RMDIR /S /Q "./fgDirect2D/x64/"
RMDIR /S /Q "./fgDirect2D/Win32/"
RMDIR /S /Q "./fgLayoutEditor/x64/"
RMDIR /S /Q "./fgLayoutEditor/Win32/"
RMDIR /S /Q "./fgNative/x64/"
RMDIR /S /Q "./fgNative/Win32/"
RMDIR /S /Q "./fgTest/x64/"
RMDIR /S /Q "./fgTest/Win32/"
DEL /F /S /Q *.out.xml
DEL /F /S /Q *.log *.exp *.idb *.ilk *.iobj *.ipdb *.metagen
DEL /F /Q .\bin\fgTest*.pdb
DEL /F /Q .\bin32\fgTest*.pdb
DEL /F /Q .\bin\fgExample*.pdb
DEL /F /Q .\bin32\fgExample*.pdb
XCOPY *.* "..\Packages\feathergui" /S /C /I /R /Y