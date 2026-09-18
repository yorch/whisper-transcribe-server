; Inno Setup script for the transcription server launcher.
;
;   iscc packaging\transcribe-server.iss
;
; Build dist\TranscriptionServer first (packaging\build-windows.ps1), which is
; what this packages. Admin is required only to add the firewall rule.

#define AppName "Transcription Server"
#define AppVersion "1.0.0"
#define AppPublisher "yorch"
#define AppExe "Transcription Server.exe"
#define AppURL "https://github.com/yorch/whisper-transcribe-server"
#define ServerPort "8765"

[Setup]
AppId={{8E2C1F44-6B1D-4C7A-9F31-7A5E2D9C4B10}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=TranscriptionServer-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The firewall rule needs elevation; the app itself runs unelevated.
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "startup"; Description: "Start the server when I sign in"; GroupDescription: "Startup:"; Flags: unchecked
Name: "firewall"; Description: "Allow other devices on my network to reach port {#ServerPort} (adds a firewall rule)"; GroupDescription: "Network:"; Flags: checkedonce

[Files]
; The whole PyInstaller output, including the vendored uv.exe and ffmpeg.exe.
Source: "..\dist\TranscriptionServer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon
; A shortcut in Startup rather than a service: the tray app owns the process,
; so it can show the token and stop cleanly. A service would have no console
; and would need its own credential story.
Name: "{userstartup}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: startup

[Run]
; Scoped to the private profile: the server is for a home/office LAN, and
; exposing it on Public networks is not what an unchecked-by-default install
; should do.
Filename: "netsh"; Parameters: "advfirewall firewall add rule name=""{#AppName}"" dir=in action=allow protocol=TCP localport={#ServerPort} profile=private"; Flags: runhidden; Tasks: firewall; StatusMsg: "Adding the firewall rule..."
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName} now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "netsh"; Parameters: "advfirewall firewall delete rule name=""{#AppName}"""; Flags: runhidden

[UninstallDelete]
; The launcher's own state (token, log) and the server's work dir are left
; behind on purpose: they hold the audit trail and any saved audio. Deleting a
; user's transcripts during an uninstall would be a nasty surprise.
Type: filesandordirs; Name: "{app}"
