# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/geniusai_server.py'],
    pathex=['src'],
    binaries=[],
    datas=datas,
    hiddenimports=[ 'chromadb.telemetry.product.posthog', 'chromadb', 'chromadb.api.rust' ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'jupyter', 'notebook', 'pandas', 'scipy'],
    win_no_prefer_redirects=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='lrgenius-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='lrgenius-server',
)
