"""Match Qt platform plugins to the Qt library already loaded by Conda/OpenCV."""
from pathlib import Path


def loaded_plugin_directory(maps=None):
    """A Conda Qt and a PySide wheel can have incompatible private Qt symbols."""
    if maps is None:
        try:
            maps = Path('/proc/self/maps').read_text()
        except OSError:
            return None
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        library = Path(fields[5])
        if library.name.startswith('libQt6Core.so'):
            plugins = library.parent/'qt6/plugins'
            if (plugins/'platforms/libqxcb.so').is_file():
                return plugins
    return None


def configure_plugins():
    # Run after PySide imports and before QApplication. No package installation,
    # global environment changes or changes to the camera process are needed.
    plugins = loaded_plugin_directory()
    if plugins:
        import os
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.setLibraryPaths([str(plugins)])
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(plugins/'platforms')
