!macro NSIS_HOOK_PREINSTALL
  ; Tauri launches the updater before its process exits. Give Windows time to release the exe.
  ${If} $UpdateMode = 1
  ${AndIf} ${FileExists} "$INSTDIR\${MAINBINARYNAME}.exe"
    Push $R0
    StrCpy $R0 40
    ${Do}
      ; Probe write access with OPEN_EXISTING and all sharing modes. The file is never written.
      System::Call 'kernel32::CreateFileW(w "$INSTDIR\${MAINBINARYNAME}.exe", i 0x40000000, i 7, p 0, i 3, i 0x80, p 0) p .r0 ?e'
      Pop $1
      ${If} $0 P<> -1
        System::Call 'kernel32::CloseHandle(p r0)'
        ${ExitDo}
      ${EndIf}
      ; Only sharing and lock violations indicate that waiting can help.
      ${If} $1 <> 32
      ${AndIf} $1 <> 33
        ${ExitDo}
      ${EndIf}
      ${If} $R0 = 0
        Pop $R0
        Abort "Raptor LLM is still open. Close it, then try the update again."
      ${EndIf}
      Sleep 250
      IntOp $R0 $R0 - 1
    ${Loop}
    Pop $R0
  ${EndIf}
!macroend
