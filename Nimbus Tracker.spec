# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('template.blend', '.'), ('aero_bg.svg', '.'), ('nimbus_bg.jpg', '.'), ('nagai_bg.svg', '.'), ('nimbus.ico', '.'), ('yolo11n-seg.pt', '.'), ('yolo11x-seg.pt', '.'), ('sam2.1_s.pt', '.'), ('auto_track_stage1.py', '.'), ('auto_track_stage2.py', '.'), ('apply_track_stage3.py', '.'), ('render_stage4.py', '.'), ('preview_track.py', '.'), ('export_setup.py', '.'), ('blender_setup.py', '.'), ('place_static.py', '.'), ('export_camera.py', '.'), ('static', 'static')]
binaries = []
hiddenimports = ['ui', 'auto_track', 'split_shots', 'segment_people', 'flow_solve',
                 'webview', 'webview.platforms.edgechromium',
                 'webview.platforms.winforms',
                 'clr', 'clr_loader', 'pythonnet']
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# pywebview (native window) + pythonnet/clr — the import lives inside a
# function in ui.py so PyInstaller can't see it statically; collect it all
# explicitly. If the WebView2 runtime is missing on the target machine the
# app still falls back to the browser, but this makes the native window work.
for _pkg in ('webview', 'clr_loader', 'pythonnet'):
    try:
        _r = collect_all(_pkg)
        datas += _r[0]; binaries += _r[1]; hiddenimports += _r[2]
    except Exception as _e:
        print('spec: could not collect', _pkg, _e)


a = Analysis(
    ['aerotrack_main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Nimbus Tracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['nimbus.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Nimbus Tracker',
)
