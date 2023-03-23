#
# Name:        IVviewer
#
# Purpose:     3D Open Inventor file viewer
#
# Author:      Toby Borland 2023, tobyborland@hotmail.com
#
# Created:     08/02/2019, 2023
# Copyright:   (c) Toby Borland 2019 - 2023
# Licence:     This program is free software: you can redistribute it and/or modify
#              it under the terms of the GNU Lesser General Public License as published by
#              the Free Software Foundation, either version 2 of the License, or
#              (at your option) any later version.
#
#               This program is distributed in the hope that it will be useful,
#               but WITHOUT ANY WARRANTY; without even the implied warranty of
#               MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#               GNU General Public License for more details.
#
#               You should have received a copy of the GNU General Public License
#               along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
    This script loads a 3D viewer for Open Inventor format files, the purpose of the viewer is to examine and debug the
    process of generating feature points from CAD STEP library models using the FreeCAD geometry engine.
    The process of determining feature points is saved in the Open Inventor ASCII format, which allows this process to
    be monitored outside the native FreeCAD interface, a loaded *.iv file may be refreshed.
    For compatibility, the PySide2 library within FreeCAD 0.20 installation is used rather than a PyQt5 library
    installation
    This software is part of a research project determining automated means to enable CAD API feature mapping,
    further details at

    Borland, T.D., 2019. An automated method mapping parametric features between computer aided design software
    (Doctoral dissertation, Brunel University London).
    https://bura.brunel.ac.uk/bitstream/2438/19184/1/FulltextThesis.pdf

"""

# keywords: python, openGL, graphics, python3, FreeCAD, scenegraph, coin3D, viewer

__author__ = "Toby Borland <tobyborland@live.com>"

#!/usr/bin/env python

import argparse
import glob
import os
import platform
import sys
from collections import namedtuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from shapematch import ViewScene
from shapematch import checkFilePath

Point = namedtuple("Point", "x y z")

# eps machine precision constant
import numpy as np

eps = np.finfo(float).eps

if platform.system() == "Linux":  # tested Ubuntu 20.04.5 LTS w/ FreeCAD 0.20.2
    # Ubuntu snaps FreeCAD 0.20.2 installation does not load local Python site-packages or FreeCAD python objects
    FreeCADPATH = r"/usr/lib/freecad/bin"
    if glob.glob(FreeCADPATH):
        PythonPackagePATH = r"/usr/lib/freecad-python3/lib"
        sys.path.insert(0, FreeCADPATH)
        # sys.path.append(FreeCADPATH)
        sys.path.insert(1, PythonPackagePATH)
        # sys.path.append(PythonPackagePATH)
        os.system("export FONTCONFIG_PATH=/etc/fonts")
    else:
        print(
            "No instance of FreeCAD located in anticipated Linux /usr/lib/freecad directory"
        )
        print("Tested standard installation via '$ sudo apt-get install freecad'")
        print("AppImage squashfs package also tested, unpack and set pathnames")
        print(
            "Ubuntu snaps FreeCAD 0.20.2 installation does not load local Python site-packages or FreeCAD python objects"
        )
        print("See https://wiki.freecad.org/Installing_on_Linux")
        sys.exit(1)
        # FreeCADPATH = os.path.expanduser('~') + os.sep + 'Applications/squashfs-root/usr/bin'
        # PythonPackagePATH = r'~/Applications/squashfs-root/usr/lib/python3.1/site-packages'

elif platform.system() == "Windows":
    FreeCADPATH = glob.glob(r"C:\Program Files\FreeCAD*\bin\FreeCAD.exe")
    if len(FreeCADPATH) > 0:
        FreeCADPATH = os.path.split(FreeCADPATH[0])[0]
        PivyPATH = FreeCADPATH + r"\Lib\site-packages"
        sys.path.append(FreeCADPATH)
        os.add_dll_directory(FreeCADPATH)
        sys.path.append(PivyPATH)
        os.add_dll_directory(PivyPATH)
    else:
        print(
            "No instance of FreeCAD located in customary MS Windows C:\Program Files\ "
        )
        sys.exit(1)
else:
    print(
        "Note pathnames for FreeCAD installation under operating systems other than Microsoft Windows"
    )
    print("or Ubuntu/Debian Linux will have to be manually specified")

# use PySide2 rather than PyQt5 with FreeCAD 0.20
from PySide2 import QtCore, QtWidgets
from PySide2.QtWidgets import QLabel, QFileDialog
from pivy.quarter import QuarterWidget
import FreeCADGui as F_GUI

class MdiQuarterWidget(QuarterWidget):
    def __init__(self, parent, sharewidget):
        QuarterWidget.__init__(self, parent=parent, sharewidget=sharewidget)
        QuarterWidget.setBackgroundColor(self, [0.0, 0.66, 1.0, 1.0])
        # QuarterWidget.renderMode()
        self.render = QuarterWidget.getSoRenderManager(self)
        # self.render.setRenderMode(self.render.WIREFRAME)
        # self.render.setRenderMode(self.render.WIREFRAME_OVERLAY)

    def minimumSizeHint(self):
        return QtCore.QSize(640, 480)

    def loadFile(self, fileName):
        if os.path.exists(fileName):
            self.childScene = ViewScene()
            self.childScene.loadIV(fileName)
            self.setSceneGraph(self.childScene.sceneRoot)
            self.setWindowTitle(fileName)
            return True
        return False

    # def closeEvent(self, event):
    #     print ("User has clicked the red x on the main window")
    #     event.accept()


class MdiMainWindow(QtWidgets.QMainWindow):
    def __init__(self, qApp, title):
        QtWidgets.QMainWindow.__init__(self)
        self._firstwidget = None
        self._workspace = QtWidgets.QMdiArea()
        self.setCentralWidget(self._workspace)
        self.setAcceptDrops(True)
        self.setWindowTitle(title)

        filemenu = self.menuBar().addMenu("&File")
        fileopenaction = QtWidgets.QAction("&Open File", self)
        filemenu.addAction(fileopenaction)
        fileopenaction.triggered.connect(self.open)

        filerefreshaction = QtWidgets.QAction("&Refresh", self)
        filemenu.addAction(filerefreshaction)
        filerefreshaction.triggered.connect(self.refresh)

        windowmapper = QtCore.QSignalMapper(self)
        # self.connect(windowmapper, QtCore.SIGNAL("mapped(QWidget *)"), self._workspace.setActiveWindow)
        windowmapper.mapped[QtCore.QObject].connect(self._workspace.setActiveSubWindow)
        self.dirname = os.curdir

    def closeEvent(self, event):
        self._workspace.closeAllSubWindows()

    def open(self):
        # self.openPath(QFileDialog.getOpenFileName(self, "", self.dirname)[0])
        self.openPath(
            QFileDialog.getOpenFileName(
                self,
                "Open scene",
                "",
                "Image Files (*.iv)",
                None,
                QFileDialog.DontUseNativeDialog,
            )[0]
        )
        # QFileDialog.getOpenFileName(self,"Open scene", IVcacheDir, "Image Files (*.iv)")

    def refresh(self):
        currentScene = self._workspace.activeSubWindow()
        currentSceneFilePath = currentScene.widget().windowFilePath()
        if os.path.exists(currentSceneFilePath):
            # close & reopen window
            currentScene.close()
            self.openPath(currentSceneFilePath)

    def openPath(self, fileName):
        if os.path.exists(fileName):
            activeMDIchild = self.findMdiChild(fileName)
            if activeMDIchild:
                self._workspace.setActiveSubWindow(activeMDIchild)
                return
            child = self.createMdiChild(fileName)
            if child.loadFile(fileName):
                self.statusBar().showMessage("File loaded", 2000)
                child.setWindowFilePath(fileName)
                child.show()
            else:
                child.close()

    def findMdiChild(self, fileName):
        canonicalFilePath = QtCore.QFileInfo(fileName).canonicalFilePath()

        for window in self._workspace.subWindowList():
            if (
                QtCore.QFileInfo(window.widget().windowFilePath()).canonicalFilePath()
                == canonicalFilePath
            ):
                return window
        return None

    def viewCurrentModelScene(self, scene, title=""):
        child = self.createMdiChild(title)
        child.show()
        child.setSceneGraph(scene)

    def createMdiChild(self, title):
        widget = MdiQuarterWidget(
            self, self._firstwidget
        )  # include MDI parent to retain scope
        self._workspace.addSubWindow(
            widget
        )  # use PySide2 rather than PyQt5, otherwise breaks here.
        widget.setWindowTitle(title)
        # widget.setFixedWidth(300)
        # widget.showMaximized()
        return widget

    def messageBox(self, text, title):
        message = QLabel(self)
        self._workspace.addSubWindow(
            message
        )  # use PySide2 rather than PyQt5, otherwise breaks here.
        message.setWindowTitle(title)
        message.setText(text)
        # message.showMaximized()
        return message


def main():
    F_GUI.setupWithoutGUI()
    Qt_app = QtWidgets.QApplication(sys.argv)

    # viewerText = (
    #     "Tests to determine identification of similar models, "
    #     "along with rejection of dissimilar models, saved to CSV file.\n"
    #     "Three models are used,\n"
    #     "testModel_S: source model,\n"
    #     "testModel_P: positive src model, which is a transformed version of source model testModel_S,\n"
    #     "testModel_N: negative src model, which is an untransformed model that is not testModel_S\n"
    # )

    mouseText = (
        "Coin3D model viewer, use mouse scroll-wheel to zoom in and out of scene,\n"
        "Hold left-hand mouse button and move mouse to rotate scene viewpoint,\n"
        "Right-click mouse button menu for further options"
    )

    ViewParser_help = (
        "IVviewer will display an ASCII scenegraph file in the open inventor IV format"
    )
    ViewParser = argparse.ArgumentParser(prog="IVviewer", usage=ViewParser_help)

    ViewParser.add_argument(
        "-I",
        "--input_file_path",
        type=checkFilePath,
        # default='/tmp/non_existent_dir',
        nargs="?",
        help="open inventor file pathname",
    )

    # ViewParser.add_argument('--version', action='version', version='%(prog)s 0.2')
    ViewArgs = ViewParser.parse_args()

    MDIinstance = MdiMainWindow(Qt_app, "Standalone FreeCAD scenegraph viewer")
    # MDIinstance.messageBox(viewerText, "Introduction")
    MDIinstance.messageBox(mouseText, "Viewing controls")
    if ViewArgs.input_file_path:
        MDIinstance.openPath(ViewArgs.input_file_path)
    MDIinstance.show()
    sys.exit(Qt_app.exec_())


if __name__ == "__main__":
    main()
