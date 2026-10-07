' Ejecuta actualizar.bat sin abrir ventana (lo usa el Programador de tareas).
Dim sh, carpeta
Set sh = CreateObject("WScript.Shell")
carpeta = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
WScript.Quit sh.Run("cmd /c """ & carpeta & "\actualizar.bat""", 0, True)
