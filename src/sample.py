'''
To run this sample script, start Slicer with the following arguments.
    --python-script "$(pwd)/src/sample.py".

For instance:
    - On Apple macOS: open /path/to/Slicer.app --args --python-script "$(pwd)/src/sample.py".
    - On a Linux distribution: ./path/to/Slicer --python-script "$(pwd)/src/sample.py".

A typical Slicer window will open with the main drop-down list containing a Hello World action.
Clicking this action will open a message box saying “Hello, World!”.
'''


class HelloWorldExtension(qt.QObject):
    def __init__(self, parent=None):
        qt.QObject.__init__(self, parent)

    def __del__(self):
        pass

    def onHelloWorld(self):
        qt.QMessageBox.information(slicer.util.mainWindow(), 'Hello, World!', 'Hello, World!')


helloWorldExtension = HelloWorldExtension()

action = qt.QAction("Hello World", slicer.util.mainWindow())
action.connect('triggered()', helloWorldExtension.onHelloWorld)

moduleSelector = slicer.util.mainWindow().moduleSelector()
moduleSelector.modulesMenu().addAction(action)
