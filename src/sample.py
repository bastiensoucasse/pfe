'''
To run this sample script, start Slicer with the following arguments.
    --python-script "path/to/sample.py".

For instance, from the project root folder:
    - On Apple macOS: open Slicer.app --args --python-script "$(pwd)/src/sample.py".
    - On a Linux distribution: .Slicer --python-script "$(pwd)/src/sample.py".

A typical Slicer window will open with the main drop-down list containing a Hello World action.
Clicking this action will open a message box saying “Hello, World!”.
'''

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    slicer: Any = None
    qt: Any = None


class HelloWorldExtension(qt.QObject):
    '''
    Hello World sample extension which opens a simple Hello World sample dialog.
    '''

    def __init__(self, parent=None):
        qt.QObject.__init__(self, parent)

    def __del__(self):
        pass

    def on_hello_world(self):
        '''
        Opens a Hello World sample dialog.
        '''

        qt.QMessageBox.information(slicer.util.mainWindow(), 'Hello, World!', 'Hello, World!')


helloWorldExtension = HelloWorldExtension()

action = qt.QAction("Hello World", slicer.util.mainWindow())
action.connect('triggered()', helloWorldExtension.on_hello_world)

moduleSelector = slicer.util.mainWindow().moduleSelector()
moduleSelector.modulesMenu().addAction(action)
