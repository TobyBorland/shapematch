#
# Name:        shapematch.py
#
# Purpose:     This script and ancillary functions demonstrates geometric shape matching
#               using feature registration points.
#
# Author:      Toby Borland 2021-2023, tobyborland@live.com
#
# Created:     08/02/2019, 2022
# Copyright:   (c) Toby Borland 2019 - 2022
# Licence:     This program is free software: you can redistribute it and/or modify
#              it under the terms of the Lesser GNU General Public License as published by
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
    This software is part of a research project determining automated means to enable CAD API feature mapping.
    The code is presented as a single file rather than a library for ease of reproduction within a
    thesis appendix.

    Borland, T.D., 2019. An automated method mapping parametric features between computer aided design software
    (Doctoral dissertation, Brunel University London).
    https://bura.brunel.ac.uk/bitstream/2438/19184/1/FulltextThesis.pdf

    This script and ancillary functions demonstrates geometric shape matching using feature registration points.
    A number of ISO-10303 STEP geometry files are analysed to extract surface feature registration points,
    namely local surface features as measured from a shape model centroid. These features include local maxima,
    local minima, circular edge centre-points, circular groove centre-points and spherical sectors.
    These features are used to determine similarity against similar features extracted from other models.
    Model operations take place using a minimal set of CAD commands as follows:

        1. Geometry model file import
        2. Creation of a vector or line at a specified orientation and passing via a defined point
            (ProjectPointToSurface())
        3. Absolute Cartesian values of point at an intersection of a model surface with said vector/line.

    Further API functions are used for checking, (AddPoint(), ObjectColor()) model insertion,
    (LastCreatedObject(), RotateObject(), DeleteObject()) and surface checking,
    (IsPolysurfaceClosed(), IsSurfaceClosed(), IsSurface(), IsPolysurface())

    Two tests are carried out, GeometrySimilarityTestA() takes a random shape from the library,
    inserts it at one position and orientation, this model is then duplicated at a different position,
    orientation and scale. A second dissimilar model is imported.
    Discriminating feature points are generated for these 3 models and are used to determine similarity matches
    between the models that differ by affine transformation and the model pair that differs by geometry.
    These values are stores as a CSV file.

    Several iterations of the first test creates multiple feature point representations of the geometry models.
    In GeometrySimilarityTestB(), these representations are tested against each other to find measures of
    comparative similarity.

    Toby Borland 2019, tobyborland@hotmail.com
"""

__author__ = "Toby Borland <tobyborland@live.com>"

#!/usr/bin/env python

# from numpy import asarray, array, dot, mean, arctan2, cross, empty, sqrt, square
# from scipy.linalg import lstsq

import sys
import glob
import os
import math
import argparse
# from argparse import RawTextHelpFormatter
# from pathlib import Path

from datetime import datetime as dt
from operator import itemgetter
import csv  # consider pandas
import time  # assess how long it takes to generate word similarity matrix
import dill as pickle  # import pickle
from random import randint, random, sample, choice, uniform
from math import cos, sin, pi, sqrt
import cv2  # python 3.8 => pip install opencv-python
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance as scipyDist
import platform
from itertools import combinations

from collections import namedtuple  # PyCharm bug will flag this as type error

Point = namedtuple("Point", "x y z")

if platform.system() == "Linux":  # tested Ubuntu 20.04.5 LTS w/ FreeCAD 0.20.2
    # Ubuntu snaps FreeCAD 0.20.2 installation does not load local Python site-packages or FreeCAD python objects
    FreeCADPATH = r"/usr/lib/freecad/bin"
    if glob.glob(FreeCADPATH):
        PythonPackagePATH = r"/usr/lib/freecad-python3/lib"
        #PivyPackagePATH = r"/usr/lib/python3/dist-packages"
        sys.path.insert(0, FreeCADPATH)
        sys.path.insert(1, PythonPackagePATH)
        #sys.path.insert(2, PivyPackagePATH)
        # sys.path.append(FreeCADPATH)
        # sys.path.append(PythonPackagePATH)
        print(
            "FreeCAD located in " + FreeCADPATH
        )
        #print(sys.path[:3])
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
    # import oschmod # Windows file permissions
    # test an instance of FreeCAD exists within
    FreeCADPATH = glob.glob(r"C:\Program Files\FreeCAD*\bin\FreeCAD.exe")
    if len(FreeCADPATH) > 0:
        FreeCADPATH = os.path.split(FreeCADPATH[0])[0]
        # PythonPackagePATH = FreeCADPATH + r"\Lib\site-packages"
        sys.path.append(FreeCADPATH)
        os.add_dll_directory(FreeCADPATH)
        # sys.path.append(PythonPackagePATH)
        # os.add_dll_directory(PythonPackagePATH)
        print(
            "FreeCAD located in " + FreeCADPATH
        )
    else:
        print(
            r"No instance of FreeCAD located in customary MS Windows C:\Program Files\ "
        )
        sys.exit(1)
else:
    print(
        "Note pathnames for FreeCAD installation under operating systems other than Microsoft Windows"
    )
    print(" will have to be manually specified")

# The libraries required for Open Inventor visual display are installed within the FreeCAD installation
# (pivy, PtSide2, PyQT5), # separate installations on the customary Python paths don't play nicely with FreeCAD.
# use PySide2 rather than PyQt5 with FreeCAD 0.20
# from PySide2 import QtCore, QtGui, QtWidgets

# from PySide2.QtWidgets import QLabel, QFileDialog
# use the Pivy from FreeCAD installation

from pivy import coin # python3-pivy requires 0.6.5, not in PyPI, see Debian packages

# from pivy.quarter import QuarterWidget
import FreeCAD as F_app
import FreeCADGui as F_GUI

# from FreeCAD import Part
import Part as Part_app

# eps machine precision constant
eps = np.finfo(float).eps
phi = (1 + np.sqrt(5)) / 2
isVerbose = True  # reporting level
# create & view scene of source, positive & negative test models together.
# single instance per run of singleShapePRtest()
viewerFlag = True
# create & save individual scenes that includes higher detail of model feature detection
generativeSceneFlag = True

DesernoSphereCache = dict()
TammesSphereCache = dict()

# longer execution times use a finishing bell
if platform.system() == "Windows":
    try:
        import winsound

    except ImportError(winsound):

        def playsound(frequency, duration):
            # apt-get install beep
            os.system("beep -f %s -l %s" % (frequency, duration))

    else:

        def playsound(frequency, duration):
            winsound.Beep(frequency, duration)

elif platform.system() == "Linux":

    def playsound(frequency, duration):
        os.popen(
            "speaker-test -t sine -f %s -l 1 & sleep %s && kill -9 $!"
            % (frequency, duration)
        )
        # apt-get install beep
        # os.system("beep -f %s -l %s" % (frequency, duration))
        # beep now requires enabling under Ubuntu, "sudo modprobe pcspkr"
        # or comment out ‘blacklist pcspkr‘ in /etc/modprobe.d/blacklist


class ViewScene(object):
    """
    class for creating an open inventor *.iv scene viewable by pivy coin Quarter library or other viewer
    FreeCAD 0.20 does not yet read imported shapes
    """

    def __init__(self, name=""):
        self.name = name
        self.pointCoordsList = []
        self.shapeNameList = []
        self.shapeNodeList = []
        self.sceneRoot = coin.SoSeparator()

        self.modelDoc = F_app.newDocument("uniqueDocHandle", label="modelDocLabel")
        F_app.setActiveDocument("uniqueDocHandle")
        self.modelDoc.recompute()

        # point added to document root allows imported IV geometry to be read
        pointObject = self.modelDoc.addObject("Part::Vertex", "originPoint")
        rootNode = F_GUI.subgraphFromObject(self.modelDoc.Objects[0])
        self.sceneRoot.addChild(rootNode)
        self.shapeNodeList.append(rootNode)
        self.shapeNameList.append("originPoint")

        self.sceneCoords = coin.SoCoordinate3()
        self.sceneMarkers = coin.SoMarkerSet()
        self.sceneMarkers.markerIndex = coin.SoMarkerSet.DIAMOND_FILLED_5_5

        self.colorRed = coin.SoBaseColor()
        self.colorRed.rgb = (1.0, 0.0, 0.0)
        self.colorLime = coin.SoBaseColor()
        self.colorLime.rgb = (0.0, 1.0, 0.0)
        self.colorGreen = coin.SoBaseColor()
        self.colorGreen.rgb = (0.0, 0.5, 0.0)
        self.colorBlue = coin.SoBaseColor()
        self.colorBlue.rgb = (0.0, 0.0, 1.0)
        self.colorYellow = coin.SoBaseColor()
        self.colorYellow.rgb = (1.0, 1.0, 0.0)
        self.colorPurple = coin.SoBaseColor()
        self.colorPurple.rgb = (1.0, 0.0, 1.0)
        self.colorCyan = coin.SoBaseColor()
        self.colorCyan.rgb = (0.0, 1.0, 1.0)
        self.colorWhite = coin.SoBaseColor()
        self.colorWhite.rgb = (1.0, 1.0, 1.0)
        self.colorBlack = coin.SoBaseColor()
        self.colorBlack.rgb = (0.0, 0.0, 0.0)
        self.colorGrey = coin.SoBaseColor()
        self.colorGrey.rgb = (0.5, 0.5, 0.5)
        self.colorMaroon = coin.SoBaseColor()
        self.colorMaroon.rgb = (0.5, 0.0, 0.0)
        self.colorNavy = coin.SoBaseColor()
        self.colorNavy.rgb = (0.0, 0.0, 0.5)
        self.colorOlive = coin.SoBaseColor()
        self.colorOlive.rgb = (0.5, 0.5, 0.0)
        self.colorSilver = coin.SoBaseColor()
        self.colorSilver.rgb = (0.75, 0.75, 0.75)
        self.colorTeal = coin.SoBaseColor()
        self.colorTeal.rgb = (0.0, 0.5, 0.5)
        self.colorOrange = coin.SoBaseColor()
        self.colorOrange.rgb = (1.0, 0.5, 0.0)
        self.colorLightBlue = coin.SoBaseColor()
        self.colorLightBlue.rgb = (0.678, 0.847, 0.902)

        self.colorMap = {
            "red": self.colorRed,
            "green": self.colorGreen,
            "lime": self.colorLime,
            "blue": self.colorBlue,
            "yellow": self.colorYellow,
            "purple": self.colorPurple,
            "cyan": self.colorCyan,
            "white": self.colorWhite,
            "black": self.colorBlack,
            "grey": self.colorGrey,
            "maroon": self.colorMaroon,
            "navy": self.colorNavy,
            "olive": self.colorOlive,
            "silver": self.colorSilver,
            "teal": self.colorTeal,
            "orange": self.colorOrange,
            "lightblue": self.colorLightBlue,
        }

    def addShape(
        self,
        shapeObject,
        shapeName,
        ambColor="",
        diffColor="",
        specColor="",
        emColor="",
        transp=1.0,
    ):
        shapeViewObject = self.modelDoc.addObject("Part::FeaturePython", shapeName)
        self.modelDoc.recompute()
        shapeViewObject.Shape = shapeObject
        modelDocObjectListLast = len(self.modelDoc.Objects) - 1
        shapeNode = F_GUI.subgraphFromObject(
            self.modelDoc.Objects[modelDocObjectListLast]
        )

        # need to navigate through the various display structures in IV representation to change viewing values
        Render1material = shapeNode.getChild(2).getChild(1).getChild(2)  # points
        Render2material = shapeNode.getChild(2).getChild(1).getChild(2)  # wireframe
        Render3material = shapeNode.getChild(2).getChild(3).getChild(1)  # shaded?

        if ambColor:
            ac = self.colorMap[ambColor.lower()].rgb.values[0]
            Render1material.ambientColor.setValue(ac[0], ac[1], ac[2])
            Render2material.ambientColor.setValue(ac[0], ac[1], ac[2])

        if diffColor:
            dc = self.colorMap[diffColor.lower()].rgb.values[0]
            Render1material.diffuseColor.setValue(dc[0], dc[1], dc[2])
            Render2material.diffuseColor.setValue(dc[0], dc[1], dc[2])
            Render3material.diffuseColor.setValue(dc[0], dc[1], dc[2])

        if specColor:
            sc = self.colorMap[specColor.lower()].rgb.values[0]
            Render1material.specularColor.setValue(sc[0], sc[1], sc[2])
            Render2material.specularColor.setValue(sc[0], sc[1], sc[2])

        if emColor:
            ec = self.colorMap[emColor.lower()].rgb.values[0]
            Render1material.emissiveColor.setValue(ec[0], ec[1], ec[2])
            Render2material.emissiveColor.setValue(ec[0], ec[1], ec[2])

        Render1material.transparency.setValue(transp)  # FreeCADgui bug writing IV file?
        Render2material.transparency.setValue(transp)
        Render3material.transparency.setValue(transp)

        shapeSep = coin.SoSeparator()
        shapeSep.addChild(shapeNode)

        self.sceneRoot.addChild(shapeSep)  # only 1 object expected
        self.shapeNodeList.append(shapeNode)
        self.shapeNameList.append(shapeName)

    def addText(self, offset, text):
        self.labelFont = coin.SoFont()
        self.labelFont.name = "Ariel"
        self.labelFont.size.setValue(10)

        self.labelSep = coin.SoSeparator()
        self.labelTransform = coin.SoTransform()
        self.label = coin.SoText2()
        self.labelTransform.translation = offset

        if text.__class__.__name__ == "list":
            self.label.string.setValues(0, len(text), text)
        elif text.__class__.__name__ == "string":
            self.label.string.setValue(text)

        self.sceneRoot.addChild(self.labelSep)
        self.labelSep.addChild(self.labelFont)
        self.labelSep.addChild(self.labelTransform)
        self.labelSep.addChild(self.colorBlack)
        self.labelSep.addChild(self.label)

    def addLine(self, V1, V2, shapeName="Line", colour="black"):
        # add a line element to the document and set its points
        # (debugging use)

        if localDisp(V1, V2) < 1e-15:
            print("zero-length line warning")
            return
        line = Part_app.LineSegment()
        line.StartPoint = (V1.x, V1.y, V1.z)
        line.EndPoint = (V2.x, V2.y, V2.z)
        lineViewObject = self.modelDoc.addObject("Part::Feature", shapeName)
        lineViewObject.Shape = line.toShape()
        modelDocObjectListLast = len(self.modelDoc.Objects) - 1
        lineNode = F_GUI.subgraphFromObject(
            self.modelDoc.Objects[modelDocObjectListLast]
        )
        lineSep = coin.SoSeparator()
        self.sceneRoot.addChild(lineSep)
        lineSep.addChild(lineNode)
        lineSep.addChild(self.colorMap[colour.lower()])

    def addPoints(self, P, colour="black", marker=None):
        # possible to supply either FreeCAD Part::Shape Vertices or individual/list of cartesian tuple Point

        if P.__class__.__name__ == "Shape":
            # vertexFromObj = [i.Point for i in P.Vertexes]
            vertexFromObj = [[i.Point.x, i.Point.y, i.Point.z] for i in P.Vertexes]
        if P.__class__.__name__ == "Point":
            vertexFromObj = [
                [P.x, P.y, P.z],
            ]
        if P.__class__.__name__ == "list":
            if len(P) > 0:
                if P[0].__class__.__name__ == "Point":
                    # vertexFromObj = P
                    vertexFromObj = [[i.x, i.y, i.z] for i in P]
            else:
                print("viewscene.addPoints() zero points given")
                return

        novelSceneCoords = coin.SoCoordinate3()
        novelSceneCoords.point.setValues(0, len(vertexFromObj), vertexFromObj)

        pointSep = coin.SoSeparator()
        self.sceneRoot.addChild(pointSep)
        pointSep.addChild(novelSceneCoords)
        pointSep.addChild(self.colorMap[colour.lower()])

        if not marker:
            pointSep.addChild(self.sceneMarkers)
        else:
            sceneMarkers = coin.SoMarkerSet()
            if marker == "cross7":
                sceneMarkers.markerIndex = coin.SoMarkerSet.CROSS_7_7
            elif marker == "cross5":
                sceneMarkers.markerIndex = coin.SoMarkerSet.CROSS_5_5
            elif marker == "diamond7":
                sceneMarkers.markerIndex = coin.SoMarkerSet.DIAMOND_FILLED_7_7
            elif marker == "diamond5":
                sceneMarkers.markerIndex = coin.SoMarkerSet.DIAMOND_FILLED_5_5
            elif marker == "diamond9":
                sceneMarkers.markerIndex = coin.SoMarkerSet.DIAMOND_FILLED_9_9
            elif marker == "diamondhollow9":
                sceneMarkers.markerIndex = coin.SoMarkerSet.DIAMOND_LINE_9_9
            elif marker == "star7":
                sceneMarkers.markerIndex = coin.SoMarkerSet.STAR_7_7
            elif marker == "circlehollow9":
                sceneMarkers.markerIndex = coin.SoMarkerSet.CIRCLE_LINE_9_9
            elif marker == "square9":
                sceneMarkers.markerIndex = coin.SoMarkerSet.SQUARE_FILLED_9_9
            elif marker == "triangle9":
                sceneMarkers.markerIndex = coin.SoMarkerSet.TRIANGLE_FILLED_9_9
            elif marker == "circle9":
                sceneMarkers.markerIndex = coin.SoMarkerSet.CIRCLE_FILLED_9_9
            pointSep.addChild(sceneMarkers)

        self.pointCoordsList.append(novelSceneCoords)

    def removeShape(self, shapeName):
        """delete referenced object from FreeCAD model space"""
        if shapeName in self.shapeNameList:
            node = self.shapeNameList.index(shapeName)
            node = self.shapeNodeList[node]
            self.sceneRoot.removeChild(node)
            self.modelDoc.removeObject(shapeName)
            self.modelDoc.recompute()
            self.shapeNameList.remove(shapeName)
            self.shapeNodeList.remove(node)

    def saveIV(self, IVpathName):
        """write IV format"""
        splitPath = os.path.split(IVpathName)
        if not os.path.exists(splitPath[0]):
            print(self.name + " scene IV path failure: " + splitPath[0])
            return
        if os.path.exists(splitPath[1]):
            print(self.name + " scene IV path overwrite: " + splitPath[0])
        try:
            with open(IVpathName, "w") as outBuffer:
                F_GUI.exportSubgraph(self.sceneRoot, outBuffer, "IV")  # VRML, not IV
                outBuffer.close()
                print(self.name + " scene IV file written")
        except OSError as e:
            print(f"{type(e)}: {e}")
            print(self.name + " scene IV file write failure to " + IVpathName)

    def saveFCstd(self, pathName):
        """write FCstd format, saves geometry structure that can be read by FreeCAD"""
        splitPath = os.path.split(pathName)
        if not os.path.exists(splitPath[0]):
            print(self.name + " scene FCstd path failure: " + splitPath[0])
            return
        if os.path.exists(splitPath[1]):
            print(self.name + " scene FCstd path overwrite: " + splitPath[0])
        try:
            self.modelDoc.recompute()
            self.modelDoc.FileName = pathName
            self.modelDoc.save()
        except OSError as e:
            print(f"{type(e)}: {e}")
            print(self.name + " model document file write failure to " + pathName)

    def loadIV(self, IVpathName):
        """read IV format"""
        # filepath sanity check
        checkPath = glob.glob(IVpathName)
        if len(checkPath) != 1:
            raise Exception("pathname failure")
        else:
            readAgent = coin.SoInput()
            readAgent.openFile(IVpathName)
            self.sceneRoot = coin.SoDB.readAll(readAgent)
            readAgent.closeFile()


class STEPmodel(object):
    """class and methods to encapsulate objects and methods unique to FreeCAD environment"""

    def __init__(self):
        self.objectHandle = (
            ""  # pointer or string used to reference model in CAD model-space
        )
        self.filepath = ""
        self.name = ""
        self.generationTime = 0
        self.surfaceStatus = ""
        self.insertionPoint = Point(0.0, 0.0, 0.0)
        self.rotation = 0.0
        self.rotationAxis = None
        self.rotationMatrix = None
        self.scale = 1.0
        self.featureMaxPoints = []
        self.featureMinPoints = []  # arrange as a list of lists representing points
        self.surfacePoints = []
        self.centroid = None
        self.rotSymRidgePoints = []
        self.rotSymGroovePoints = []
        self.featureMaxCurveDisps = []
        self.featureMaxCentres = []
        self.featureMinCurveDisps = []
        self.featureMinCentres = []
        self.featureSphereDisps = []
        self.spherePoints = []
        self.centrePoints = []

    # noinspection PyTypeChecker
    def importSTEPmodel(
        self,
        filepath="",
        insertionPoint=Point(0.0, 0.0, 0.0),
        scale=1,
        rotation=0.0,
        rotationAxis=[],
    ):
        """
        FreeCAD dependent

        :param filepath: string
        :param insertionPoint: {x,y,z} Cartesian tuple, not equivalent to centroid
        :param scale: floating point scalar
        :param rotation: floating point scalar for Rhino degree value (0.0 <= rotation <= 360.0)
        :param rotationAxis:
        """

        # get default values from existing object or replace default initialisation values with given parameters
        if (filepath == "") and not (self.filepath == ""):
            filepath = self.filepath
        elif self.filepath == "":
            self.filepath = filepath

        if (insertionPoint == Point(0.0, 0.0, 0.0)) and not (
            self.insertionPoint == Point(0.0, 0.0, 0.0)
        ):
            insertionPoint = self.insertionPoint
        elif self.insertionPoint == Point(0.0, 0.0, 0.0):
            self.insertionPoint = insertionPoint

        if (scale == 1) and not (self.scale == 1):
            scale = self.scale
        elif self.scale == 1:
            self.scale = scale

        if (rotation == 0.0) and not (self.rotation == 0.0):
            rotation = self.rotation
        elif self.rotation == 0.0:
            self.rotation = rotation

        if (rotationAxis == []) and not (self.rotationAxis is None):
            rotationAxis = self.rotationAxis
        elif self.rotationAxis is None:
            self.rotationAxis = rotationAxis

        if self.name == "":
            self.name = os.path.basename(filepath)

        # def angleAxisTranslate2AffineMat(rotAxis, theta, disp):
        #     """
        #     Affine transformation matrix from 3x3 rotation matrix and displacement vector.
        #     3x3 rotation matrix derived from CCW angle theta around axis
        #     From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle & transform3d
        #
        #     :param rotAxis: axis of rotation vector axis emanating from origin
        #     :param theta: scalar angle of rotation
        #     :param disp: displacement from origin point, presumed (0,0,0)
        #     :return: 4x4 matrix representing affine transformation
        #     """
        #
        #     s = sin(theta); c = cos(theta); C = (1 - c)
        #
        #     xs = rotAxis[0] * s; ys = rotAxis[1] * s; zs = rotAxis[2] * s
        #
        #     xC = rotAxis[0] * C; yC = rotAxis[1] * C; zC = rotAxis[2] * C
        #
        #     xyC = rotAxis[0] * yC; yzC = rotAxis[1] * zC; zxC = rotAxis[2] * xC
        #
        #     xxC = rotAxis[0] * xC; yyC = rotAxis[1] * yC; zzC = rotAxis[2] * zC
        #
        #     return F_app.Matrix(xxC + c, xyC - zs, zxC + ys, disp[0],
        #                         xyC + zs, yyC + c, yzC - xs, disp[1],
        #                         zxC - ys, yzC + xs, zzC + c, disp[2],
        #                         0,        0,        0,       1      )

        try:
            # filepath sanity check
            checkPath = glob.glob(filepath)
            if len(checkPath) != 1:
                raise Exception("pathname failure")
            else:
                self.objectHandle = Part_app.read(filepath)

            if not self.objectHandle.isValid():
                raise Exception("imported STEP file fails geometry checks")

            if self.objectHandle.check():
                print("imported STEP file passes detailed geometry shape.check()")

            if not self.objectHandle.isClosed():
                raise Exception("imported STEP file fails closed geometry checks")

            # uniform scaling
            if scale - 1.0 >= eps:
                scalingMatrix = F_app.Matrix()
                scalingMatrix.scale(scale, scale, scale)
                self.objectHandle = self.objectHandle.transformGeometry(scalingMatrix)

            # shape rotation
            # if math.fabs(rotation) > eps:
            if not rotationAxis:
                # rotationAxis = F_app.Vector(0.0, 0.0, 1.0)  # Z-axis
                rotationAxis = [0.0, 0.0, 0.0]

            self.rotationMatrix = angleAxis2RotMat(rotationAxis, math.radians(rotation))
            self.objectHandle.Placement.Matrix = F_app.Matrix(
                self.rotationMatrix[0][0],
                self.rotationMatrix[0][1],
                self.rotationMatrix[0][2],
                self.insertionPoint.x,
                self.rotationMatrix[1][0],
                self.rotationMatrix[1][1],
                self.rotationMatrix[1][2],
                self.insertionPoint.y,
                self.rotationMatrix[2][0],
                self.rotationMatrix[2][1],
                self.rotationMatrix[2][2],
                self.insertionPoint.z,
                0,
                0,
                0,
                1,
            )

        except Exception as e:
            raise RuntimeError("Model insertion failure")

    def rayModelIntersection(self, V1, V2):
        """
        Returns the vertex that is at the intersection of the face
        and the line that passes through vertex1 and vertex2

        :param V1: origin of ray
        :param V2: direction of ray
        :return:
        """

        # Define a faraway extent that allows FreeCAD to approximate an infinite ray for intersection points.
        FreeCADEXTENTS = 100  # calculation speed sensitive to this scaling value
        # not required for rayModelIntersection3()

        # input variable check, vertices must be of Point type
        for v in [V1, V2]:
            if v.__class__.__name__ == "Shape":
                v = Point(v.Vertexes.Point.x, v.Vertexes.Point.y, v.Vertexes.Point.z)

        # offset V1 to cartesian origin, (0, 0, 0), move V2 accordingly
        _V2 = Point(V2.x - V1.x, V2.y - V1.y, V2.z - V1.z)
        lenV = sqrt(_V2.x**2 + _V2.y**2 + _V2.z**2)
        unitV = Point(_V2.x / lenV, _V2.y / lenV, _V2.z / lenV)
        farV2 = Point(
            FreeCADEXTENTS * unitV.x, FreeCADEXTENTS * unitV.y, FreeCADEXTENTS * unitV.z
        )
        farV1 = Point(-farV2.x, -farV2.y, -farV2.z)
        farV2 = Point(farV2.x + V1.x, farV2.y + V1.y, farV2.z + V1.z)
        farV1 = Point(farV1.x + V1.x, farV1.y + V1.y, farV1.z + V1.z)

        # """
        # Redefine second vertex as the second vertex projected to the surface
        # of a large sphere centered at the first vertex.
        #
        # :param V1: origin of ray
        # :param V2: direction of ray, to be moved to far model extents
        # :param sphereR: radius of far model extents
        # :return: redefined V2
        # """

        ray = Part_app.makeLine(
            (farV1.x, farV1.y, farV1.z), (farV2.x, farV2.y, farV2.z)
        )
        intersections = ray.common(self.objectHandle)
        # if intersections.Vertexes:
        #     [print(i.Point.x, i.Point.y, i.Point.z) for i in intersections.Vertexes]
        intersections = [
            Point(i.Point.x, i.Point.y, i.Point.z) for i in intersections.Vertexes
        ]
        # if vertex1 in intersections:
        #     intersections.remove(vertex1)
        if intersections:
            [
                self.surfacePoints.append(i)
                for i in intersections
                if i not in self.surfacePoints
            ]

        return intersections

    def rayModelIntersection2(self, V1, V2):
        """
        Returns the vertex that is at the intersection of the face
        and the line that passes through vertex1 and vertex2

        :param V1: origin of ray
        :param V2: direction of ray
        :return:
        """

        # Define a faraway extent that allows FreeCAD to approximate an infinite ray for intersection points.
        FreeCADEXTENTS = 300  # calculation speed sensitive to this scaling value
        # not required for rayModelIntersection3()

        # input variable check, vertices must be of Point type
        # for v in [V1, V2]:
        #     if v.__class__.__name__ == 'Shape':
        #         v = Point(v.Vertexes.Point.x, v.Vertexes.Point.y, v.Vertexes.Point.z)

        # offset V1 to cartesian origin, (0, 0, 0), move V2 accordingly
        _V2 = Point(V2.x - V1.x, V2.y - V1.y, V2.z - V1.z)
        lenV = sqrt(_V2.x**2 + _V2.y**2 + _V2.z**2)
        unitV = Point(_V2.x / lenV, _V2.y / lenV, _V2.z / lenV)
        farV2 = Point(
            FreeCADEXTENTS * unitV.x, FreeCADEXTENTS * unitV.y, FreeCADEXTENTS * unitV.z
        )
        # farV1 = Point(-farV2.x, -farV2.y, -farV2.z)
        farV2 = Point(farV2.x + V1.x, farV2.y + V1.y, farV2.z + V1.z)
        # farV1 = Point(farV1.x + V1.x, farV1.y + V1.y, farV1.z + V1.z)

        # """
        # Redefine second vertex as the second vertex projected to the surface
        # of a large sphere centered at the first vertex.
        #
        # :param V1: origin of ray
        # :param V2: direction of ray, to be moved to far model extents
        # :param sphereR: radius of far model extents
        # :return: redefined V2
        # """

        intersectSet = []
        ray = Part_app.makeLine((V1.x, V1.y, V1.z), (farV2.x, farV2.y, farV2.z))
        intersections = ray.common(self.objectHandle)
        if intersections.Vertexes:
            for iv in intersections.Vertexes:
                if not (
                    (abs(iv.Point.x - V1.x) < eps)
                    and (abs(iv.Point.y - V1.y) < eps)
                    and (abs(iv.Point.z - V1.z) < eps)
                ):
                    if (
                        (abs(iv.Point.x - farV2.x) < eps)
                        and (abs(iv.Point.y - farV2.y) < eps)
                        and (abs(iv.Point.z - farV2.z) < eps)
                    ):
                        print(
                            "WARNING: far model intersection constant likely to have been exceeded by model extents"
                        )
                    else:
                        intersectSet.append(Point(iv.Point.x, iv.Point.y, iv.Point.z))

            [
                self.surfacePoints.append(i)
                for i in intersectSet
                if i not in self.surfacePoints
            ]

        return intersectSet

        # face.Surface.intersect(edge.Curve)[0][0] should look like
        # face.Shape.Surface.intersect(edge.Curve)[0][0] if working with objects

    def rayModelIntersection3(self, V1, V2):
        """
        Returns the vertex that is at the intersection of the face
        and the line that passes through vertex1 and vertex2

        :param V1: origin of ray
        :param V2: direction of ray
        :return:
        """

        ray = Part_app.makeLine(V1, V2)
        # extract the curve from the ray, then test intersections with shape faces
        allFaceSets = [ray.Curve.intersect(f.Surface) for f in self.objectHandle.Faces]
        intersectPoints = [x for face in allFaceSets for x in face if x]
        intersectPoints = [x for face in intersectPoints for x in face]
        intersectPoints = [Point(p.X, p.Y, p.Z) for p in intersectPoints]
        [
            self.surfacePoints.append(i)
            for i in intersectPoints
            if i not in self.surfacePoints
        ]
        return intersectPoints

    def printModelAlignment(self):
        """
        Print insertion point and rotation of model to stdout.

        :return: stdout display
        """
        print(
            "Model insertion point: x: {:f}, y: {:f}, z: {:f}".format(
                self.insertionPoint.x, self.insertionPoint.y, self.insertionPoint.z
            )
        )
        print("Model insertion scale: {:f}".format(self.scale))
        print("Model insertion rotation (single axis?): {:f}".format(self.rotation))
        if self.rotationAxis:
            print(
                "Model insertion rotation axis: [{:f}, {:f}, {:f}]".format(
                    self.rotationAxis[0], self.rotationAxis[1], self.rotationAxis[2]
                )
            )
        else:
            print("Model insertion rotation axis: []")

    def featureClean(self):
        """
        remove spurious NaN values from feature values,
        (malformed tests)
        """

        def cleanPointNaN(P):
            Pclean = [
                p for p in P if not any([np.isnan(p.x), np.isnan(p.y), np.isnan(p.z)])
            ]
            if len(Pclean) != len(Pclean):
                print("NaN cleaned")
            return Pclean

        def cleanNaN(P):
            Pclean = [p for p in P if not np.isnan(p)]
            if len(Pclean) != len(Pclean):
                print("NaN cleaned")
            return Pclean

        # cleanPointNaN = lambda P: [p for p in P if not any([np.isnan(p.x), np.isnan(p.y), np.isnan(p.z)])]
        # cleanNaN = lambda P: [p for p in P if not np.isnan(p)]

        self.featureMaxPoints = cleanPointNaN(self.featureMaxPoints)
        self.featureMinPoints = cleanPointNaN(self.featureMinPoints)
        self.surfacePoints = cleanPointNaN(self.surfacePoints)
        self.rotSymRidgePoints = cleanPointNaN(self.rotSymRidgePoints)
        self.rotSymGroovePoints = cleanPointNaN(self.rotSymGroovePoints)
        self.featureMaxCurveDisps = cleanNaN(self.featureMaxCurveDisps)
        self.featureMaxCentres = cleanPointNaN(self.featureMaxCentres)
        self.featureMinCurveDisps = cleanNaN(self.featureMinCurveDisps)
        self.featureMinCentres = cleanPointNaN(self.featureMinCentres)
        self.featureSphereDisps = cleanNaN(self.featureSphereDisps)


def angleAxis2RotMat(rotAxis, theta):
    """
    Affine transformation matrix from 3x3 rotation matrix and displacement vector.
    3x3 rotation matrix derived from CCW angle theta around axis
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle & transform3d

    :param rotAxis: axis of rotation vector axis emanating from origin
    :param theta: scalar radian angle of rotation
    :return: 4x4 matrix representing affine transformation
    """

    # Point2XYZarray(rotAxis)

    # normalise to unit vector
    rotNorm = np.linalg.norm(rotAxis)
    if rotNorm > eps:
        rotAxis = rotAxis / rotNorm
        rotAxis = rotAxis.tolist()

    if rotAxis.__class__.__name__ == "list":
        rotAxis = Point(rotAxis[0], rotAxis[1], rotAxis[2])

    s = sin(theta)
    c = cos(theta)
    C = 1 - c
    xs = rotAxis.x * s
    ys = rotAxis.y * s
    zs = rotAxis.z * s
    xC = rotAxis.x * C
    yC = rotAxis.y * C
    zC = rotAxis.z * C
    xyC = rotAxis.x * yC
    yzC = rotAxis.y * zC
    zxC = rotAxis.z * xC
    xxC = rotAxis.x * xC
    yyC = rotAxis.y * yC
    zzC = rotAxis.z * zC

    return np.array(
        [
            [xxC + c, xyC - zs, zxC + ys],
            [xyC + zs, yyC + c, yzC - xs],
            [zxC - ys, yzC + xs, zzC + c],
        ]
    )


def printVerbose(text):
    if isVerbose:
        print(text)


def displayPointFC(P, colour, sceneInstance, marker=None):
    # noinspection GrazieInspection
    """
    create visible point, or set of points within coin3d display GUI
    FreeCAD, Pivy dependent
    Some of Open Inventor markers defined, see
    https://www.openinventor.com/reference-manuals/NewRefMan9912/RefManJava/com/openinventor/inventor/nodes/SoMarkerSet.MarkerTypes.html

    :param P: point reference
    :param colour: text or numeric colour reference
    :param sceneInstance: coin3d pivy graphical display instance
    :param marker: custom point marker can be specified,
    :return: None
    """

    # 16711935 magenta 16711680 blue 65280 green 32767 orange 65535 yellow 8947848 grey 8913032 brown 136 brown
    RhinoColours = {
        16711935: "magenta",
        16711680: "blue",
        65280: "green",
        32767: "orange",
        65535: "yellow",
        8421376: "olive",
        15790320: "dunno",
        0: "black",
    }
    if isinstance(colour, int):
        if colour in RhinoColours.keys():
            colour = RhinoColours[colour]
        else:
            colour = "black"

    sceneInstance.addPoints(P, colour, marker)


def projectVector(pPoint, modelObject, pVector):
    """
    Wrapper for CAD projection function RhinoInstance.ProjectPointToSurface
    FreeCAD dependent

    :param pPoint: cartesian point
    :param modelObject: CAD object reference
    :param pVector: vector defining point projection direction
    :return: intersected points array
    """
    return modelObject.rayModelIntersection2(pPoint, pVector)


def proj2object2(objectHandle, projVector, projPoint, rankType="nearest", scene=None):
    # noinspection GrazieInspection
    """
    Return array of intersections between vector(s) defined as a namedtuple Point vector origin
    and a Point defining unit vector(s)

    :param objectHandle: object reference
    :param projVector: vector defining point projection direction (from origin, or local origin)
    :param projPoint: cartesian point to be projected
    :param rankType:
    :param scene: coin3d pivy graphical display instance, point display for debug purposes
    :return: defines order of intersected points returned
    """

    # deal with single points per call rather than lists
    # unit vectors akin to CAD construction lines, infinite in length
    # returned points will have multiple intersections except at tangent
    # note projPoint generally corresponds to localCentroid, not required for shapes with centroid at cartesian origin

    surfaceIntersectPoints = projectVector(
        projPoint, objectHandle, projVector
    )  # update for point array
    if len(surfaceIntersectPoints) == 1:
        return surfaceIntersectPoints[0]

    if len(surfaceIntersectPoints) > 1:
        closestPoint = None
        closestPointIndex = None
        vectorPointDisp = localDisp(surfaceIntersectPoints, projVector)

        if rankType == "nearest":
            # find closest value to projVector - vectors are normalised to local centroid,
            # non-origin centroids must use localCentroid as projected Point
            closestPointIndex = vectorPointDisp.index(min(vectorPointDisp))
            # print("proj2object-nearest intersection number > 1: "+ str(len(surfaceIntersectPoints)))
        else:
            originPointDisp = localDisp(surfaceIntersectPoints, projPoint)
            # get the intersection points on the vector side of the projected point
            # any surface intersection that is closer to the displaced vector point than the point
            # projected is on the side of the vector
            nearsidePoints = [
                opdv
                for opdi, opdv in enumerate(originPointDisp)
                if vectorPointDisp[opdi] < opdv
            ]
            if nearsidePoints:
                if rankType == "localMax":
                    # find greatest absolute value along projVector
                    # must maintain sign of projVector
                    closestPointIndex = originPointDisp.index(max(nearsidePoints))
                elif rankType == "localMin":
                    # find smallest absolute value along projVector
                    # must maintain sign of projVector
                    closestPointIndex = originPointDisp.index(min(nearsidePoints))
            # else:
            #     dd=surfaceIntersectPoints+[offsetProjVector]+[projPoint]

        if closestPointIndex is not None:
            closestPoint = surfaceIntersectPoints[closestPointIndex]

            if scene:
                if (rankType == "localMax") or (rankType == "localMin"):
                    displayPointFC(closestPoint, "lime", scene)
                else:
                    displayPointFC(closestPoint, "blue", scene)

        return closestPoint
    else:
        return None


def TammesAngle(numberOfPoints=45):
    """
    Return radian angle between points on a Tammes sphere of numberOfPoints
    minimum angle for Thompson problem

    :param numberOfPoints: number of points
    :return:
    """

    # from math import pi

    TStable = [
        0,
        0,
        0,
        0,
        0,
        90.000,
        90.000,
        72.000,
        71.694,
        61.190,
        64.996,
        58.540,
        63.435,
        52.317,
        52.866,
        49.225,
        48.936,
        50.108,
        47.534,
        44.910,
        46.093,
        44.321,
        43.302,
        41.481,
        42.065,
        39.610,
        38.842,
        39.940,
        37.824,
        36.391,
        36.942,
        36.373,
        37.377,
        33.700,
        33.273,
        33.100,
        33.229,
        32.332,
        33.236,
        32.053,
        31.916,
        31.528,
        31.245,
        30.867,
        31.258,
        30.207,
        29.790,
        28.787,
        29.690,
        28.387,
        29.231,
    ]

    return TStable[numberOfPoints] / 180 * pi


def DesernoSphere(numberOfPoints=100, radRot=0.0, interleavedFlag=None):
    """
    One of two schemas that create equidistantly spaced points on a unit sphere centered on the Cartesian space origin.
    Return array of Point (namedtuple cartesian coordinates 'x, y, z') defining on a sphere of radius 1.0 from
    [0, 0, 0] origin.
    Deserno, Markus. "How to generate equidistributed points on the surface of a sphere." If Polymerforshung (Ed.)
    99.2 (2004).

    :param interleavedFlag:
    :param numberOfPoints: number of points
    :param radRot: angle parameter for rotation of point array around Z-axis
    :return: array of generated points
    """

    # For numberOfPoints points placed on circles of latitude at constant intervals of dtheta
    # Equidistant spacing on latitude circles of dphi such that dphi roughly equals dtheta

    global DesernoSphereCache

    numberOfPoints = 2 * (numberOfPoints // 2)  # floor to nearest even number

    # from math import sin, cos, pi, sqrt
    r = 1
    ptsOnSphere = []

    if numberOfPoints in DesernoSphereCache.keys():
        ptsOnSphere = DesernoSphereCache[numberOfPoints]
        # ignore the logistics of a previously saved model with a previously applied rotation
        ptsOnSphere = rotateCluster2(
            Point(0.0, 0.0, 0.0), Point(0.0, 0.0, 1.0), ptsOnSphere, radRot
        )

    else:
        # a = (4*pi*(r**2))/N # sphere area / number of divisions
        a = (4 * pi) / numberOfPoints  # sphere area / number of divisions
        d = sqrt(a)  # roughly dtheta * dphi
        Mtheta = round(pi / d)  # divisions of hemisphere
        # print("Mtheta: {}".format(pi/d))
        dtheta = pi / Mtheta
        dphi = a / dtheta
        for m in range(0, Mtheta):
            theta = pi * (m + 0.5) / Mtheta
            Mphi = round(2 * pi * sin(theta) / dphi)
            # print("Mphi: {}".format(2*pi*sin(theta)/dphi))
            for n in range(0, Mphi):
                _phi = 2 * pi * n / Mphi + radRot
                ptNew = Point(
                    r * sin(theta) * cos(_phi),
                    r * sin(theta) * sin(_phi),
                    r * cos(theta),
                )
                ptsOnSphere.append(ptNew)

    if numberOfPoints not in DesernoSphereCache.keys():
        DesernoSphereCache[numberOfPoints] = ptsOnSphere

    if interleavedFlag:
        # arrange order of Deserno sphere points so that opposing points within sphere are called in sequence
        # means round numberOfPoints down to even
        ptsIndex = [i for i in range(0, numberOfPoints - 1)]
        ptsIndex = [
            ptsIndex[0 : len(ptsIndex) // 2],
            ptsIndex[len(ptsIndex) // 2 :],
        ]
        ptsIndex = [x for t in zip(*ptsIndex) for x in t]
        ptsOnSphere = [ptsOnSphere[p] for p in ptsIndex]

    return ptsOnSphere


def DesernoAngle(numberOfPoints=100):
    """
    Return the radian angle between adjacent points on a Deserno sphere of N points

    :param numberOfPoints: number of points of said Deserno scheme sphere
    :return: radian angle
    """

    d = sqrt((4 * pi) / numberOfPoints)
    Mtheta = round(pi / d)
    dtheta = pi / Mtheta * 2
    return dtheta


def TammesSphere(numberOfPoints=100):
    """
    One of two schemas that create equidistantly spaced points on a unit sphere.
    Return array of Point (namedtuple cartesian coordinates 'x, y, z') defining on a sphere of radius 1.0
    from [0, 0, 0] origin. Derived from:
    http://web.archive.org/web/20120421191837/http://www.cgafaq.info/wiki/Evenly_distributed_points_on_sphere

    :param numberOfPoints: number of points defined
    :return: point array
    """

    # avoid duplicating Tammes sphere calculation using global dictionary structure
    global TammesSphereCache

    if numberOfPoints in TammesSphereCache.keys():
        ptsOnSphere = TammesSphereCache[numberOfPoints]

    else:
        # from math import cos, sin, pi, sqrt
        dlong = pi * (3.0 - sqrt(5.0))  # ~2.39996323
        dz = 2.0 / numberOfPoints
        long = 0.0
        z = 1.0 - dz / 2.0
        ptsOnSphere = []
        for k in range(0, numberOfPoints):
            r = sqrt(1.0 - z * z)
            ptNew = Point(cos(long) * r, sin(long) * r, z)
            ptsOnSphere.append(ptNew)
            # only positive ptNew are returned
            # if (ptNew.x >= 0.0)and(ptNew.y >= 0.0)and(ptNew.z >= 0.0):
            #    ptsOnSphere.append( ptNew )
            z = z - dz
            long = long + dlong

    if numberOfPoints not in TammesSphereCache.keys():
        TammesSphereCache[numberOfPoints] = ptsOnSphere

    return ptsOnSphere


def medianPoint(pointArray):
    """
    Returns centroid determined as the median of cartesian values in an input array of points

    :param pointArray: array of Cartesian points
    :return: point
    """

    # from statistics import median # apparently slow/accurate
    # from numpy import median
    if (None in pointArray) and (len(pointArray) < 2):
        raise RuntimeError("medianPoint() malformed input")

    xRange = [xyz.x for xyz in pointArray]
    xCentroid = max(xRange) - ((max(xRange) - min(xRange)) / 2)
    yRange = [xyz.y for xyz in pointArray]
    yCentroid = max(yRange) - ((max(yRange) - min(yRange)) / 2)
    zRange = [xyz.z for xyz in pointArray]
    zCentroid = max(zRange) - ((max(zRange) - min(zRange)) / 2)
    return Point(xCentroid, yCentroid, zCentroid)


def meanPoint(pointArray):
    """
    Returns centroid determined as the mean of cartesian values in an input array of points

    :param pointArray: array of Cartesian points
    :return: point
    """

    # from statistics import mean # apparently kinda slow/accurate
    # from numpy import mean
    if (None in pointArray) and (len(pointArray) < 2):
        raise RuntimeError("meanPoint() malformed input")

    xCentroid = [xyz.x for xyz in pointArray]
    yCentroid = [xyz.y for xyz in pointArray]
    zCentroid = [xyz.z for xyz in pointArray]

    xCentroid = np.mean(xCentroid)
    yCentroid = np.mean(yCentroid)
    zCentroid = np.mean(zCentroid)

    return Point(xCentroid, yCentroid, zCentroid)


def rotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.

    :param axis: axis defined as a vector point relative to cartesian origin
    :param theta: rotation angle in radians
    :return: rotation matrix
    """

    # Euler-Rodrigues formula
    # http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    # from numpy import asarray, array, dot
    # from math import sqrt, cos, sin
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    SDAA = sqrt(np.dot(axis, axis))
    if SDAA == 0.0:
        raise RuntimeError("rotationMatrix() zero div error (0,0,0)?")

    axis = axis / SDAA
    a = cos(theta / 2)
    b, c, d = -axis * sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def rotateCluster2(originOffset, pAxis, Cluster, radRot=0.0):
    """
    Anticlockwise rotation of points Cluster around pAxis by radian angle radRot
    Note that pAxis is defined wrt originOffset, not relative to the cartesian origin

    :param originOffset: displacement from Cartesian origin, defined as Point type
    :param pAxis: axis defined by pAxis through originOffset
    :param Cluster: points to rotate in array
    :param radRot: rotation in radians
    :return: rotated points in array
    """
    """anticlockwise rotation of Cluster around axis defined by p through origin by radian angle radRot"""
    # originOffset, pAxis are defined as Point type

    oCluster = [
        np.array([c.x - originOffset.x, c.y - originOffset.y, c.z - originOffset.z])
        for c in Cluster
    ]
    opAxis = np.array(
        [pAxis.x - originOffset.x, pAxis.y - originOffset.y, pAxis.z - originOffset.z]
    )
    pRotAxis = rotationMatrix(opAxis, radRot)
    oCluster = [np.dot(pRotAxis, c) for c in oCluster]
    Cluster = [
        Point(oc[0] + originOffset.x, oc[1] + originOffset.y, oc[2] + originOffset.z)
        for oc in oCluster
    ]

    return Cluster


def rotatedSurroundArray(originOffset, centralVector, radAperture, radRot=0.0):
    """
    Given an initial direction centralVector, return a surrounding array of 8 points arranged at 45 degree intervals
    around direction centralVector, similar to the cardinal points on a compass rosette.
    Surrounding array vectors are angled through radAperture from centralVector.
    The surrounding cluster of vectors are rotated around a cenntralVector axis by radRot

    :param originOffset: displacement of point on vector from cartesian origin
    :param centralVector: direction of vector relative to originOffset
    :param radAperture: angle between centreVector and surrounding vectors
    :param radRot: torsional rotation of vector cluster around centreVector
    :return: surrounding vector array
    """
    # from numpy import dot

    def arrayTypeCast(xyz):
        if type(xyz) == list:
            xyz = np.array([xyz[0], xyz[1], xyz[2]])
        elif type(xyz) == Point:
            xyz = np.array([xyz.x, xyz.y, xyz.z])
        elif type(xyz) == np.ndarray:
            pass
        else:
            raise RuntimeError("input type error")
        return xyz

    def orthoAxes(p):
        """
        Return the orthogonal axes to a vector from origin to xyz, also pi/4 axes

        :param p:  single numpy array
        :return: array of 4 vectors defining orthogonal and diagonal axes to p
        """

        p /= np.linalg.norm(p)
        if (abs(p[0]) < 5 * eps) and (abs(p[1]) < 5 * eps) and (abs(p[2]) < 5 * eps):
            raise RuntimeError("orthoAxes zero vector error")
        elif (abs(p[0]) < 5 * eps) and (abs(p[1]) < 5 * eps):
            kAxis = np.array([1.0, 1.0, 0.0])
        elif (abs(p[1]) < 5 * eps) and (abs(p[2]) < 5 * eps):
            kAxis = np.array([0.0, 1.0, 0.0])
        elif (abs(p[0]) < 5 * eps) and (abs(p[2]) < 5 * eps):
            kAxis = np.array([1.0, 0.0, 1.0])
        elif abs(p[2]) == 0.0:
            kAxis = np.array([0.0, 0.0, 1.0])
        else:
            kAxis = np.cross(p, np.array([-p[0], -p[1], p[2]]))
        if np.all(kAxis == 0.0):
            pass
        jAxis = np.cross(p, kAxis)
        jkAxis = jAxis + kAxis
        kjAxis = np.cross(p, jkAxis)
        return [kAxis, jkAxis, jAxis, kjAxis]

    o = arrayTypeCast(originOffset)
    v = arrayTypeCast(centralVector)

    # transpose centralVector back to absolute origin
    vO = [v[0] - o[0], v[1] - o[1], v[2] - o[2]]
    compassAxes = orthoAxes(vO)
    compassPos = [np.dot(rotationMatrix(ca, radAperture), vO) for ca in compassAxes]
    compassNeg = [np.dot(rotationMatrix(ca, -radAperture), vO) for ca in compassAxes]
    compass = compassPos + compassNeg
    rotAxis = rotationMatrix(vO, radRot)
    rotdCompass = [np.dot(rotAxis, c) for c in compass]
    return [Point(c[0] + o[0], c[1] + o[1], c[2] + o[2]) for c in rotdCompass]


# noinspection PyUnresolvedReferences
def localDisp(p1, p2):
    """
    Return displacement(s) between input points or list of points

    :param p1: single point
    :param p2: single or list of points
    :return: disp value or list of values
    """

    if not isinstance(p2, tuple):
        raise RuntimeError("localDisp second variable must be single point")
    if isinstance(p1, tuple):
        # noinspection PyUnresolvedReferences
        return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    elif isinstance(p1, list):
        return [
            sqrt((pp.x - p2.x) ** 2 + (pp.y - p2.y) ** 2 + (pp.z - p2.z) ** 2)
            for pp in p1
        ]
    else:
        raise RuntimeError("localDisp error")


def offsetPoints(p1, p2):
    """
    Displace point or list of points by displacement of point from origin

    :param p1: singlepoint or (x, y, z) tuple or list of points to be displaced
    :param p2: single point representing displacement vector from origin
    :return: displaced point or list of point
    """

    if not isinstance(p2, tuple):
        raise RuntimeError("offsetPoints second variable must be single point")
    if isinstance(p1, tuple):
        return Point(p1.x + p2.x, p1.y + p2.y, p1.z + p2.z)
    elif isinstance(p1, list):
        return [Point(pp.x + p2.x, pp.y + p2.y, pp.z + p2.z) for pp in p1]
    else:
        raise RuntimeError("offsetPoints error")


def pointIn3PointPlane(P1, P2, P3, Pcheck):
    """
    Return displacement of Pcheck from a plane defined by points P1, P2, P3

    :param P1: point defining plane
    :param P2: point defining plane
    :param P3: point defining plane
    :param Pcheck: point determined displacement from defined plane
    :return: orthogonal displacement from place
    """

    A = np.array([P1.x, P1.y, P1.z])
    B = np.array([P2.x, P2.y, P2.z])
    C = np.array([P3.x, P3.y, P3.z])

    # the cross product is a vector normal to the plane
    cp = np.cross(C - A, B - A)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, C)

    return (a * Pcheck.x + b * Pcheck.y + c * Pcheck.z - d) / np.sqrt(
        a * a + b * b + c * c
    )


def rotSymTest6(
    ShapeInstance, centrePoint, localCentroid, featureType, minAngle=0.01, scene=None
):
    """
    Test for rotational symmetry present in a ShapeInstance
    Project & test intersect displacement of several rays from localCentroid on Z-plane
    about finalPoint to check intersections have similar radial displacement

    If a ridge is identified at three points it becomes possible to calculate the ridge
    centre point (pointsToCircleRadiusCentre), The additional complexity of
    this method correspond to determination of three points that accurately intersect the ridge and are
    broadly spaced in order to return a centre point with reasonable accuracy (rotSymTest5).
    One subroutine uses an iterative search method similar to the rosette search to find points
    on ridge maxima that flank a discovered ridge point (getRadialEdges), a second
    subroutine refines the accuracy of the point centred on the discovered ridge
    (refineRadialEdgesMidpoint).
    Extra geometrical information is returned in maxState, the number of local maximum
    displacements measured between the local centroid and terminating neighbouring points.
    E.g. maxima/minima ellipsoid return zero -> 1, equals maxPoint or minPoint
    a ridge/groove not symmetric about an axis through a local shape centroid origin will return 2 maxima/minima
    theoretically a perfect 90 deg corner should return maxState of 3

    :param ShapeInstance: CAD shape object to be tested
    :param centrePoint: nucleating point on surface
    :param localCentroid: local centroid of shape reference, (geometric, not barycentre of points)
    :param featureType: "localMax", "localMin", convex and concave feature respectively
    :param scene: coin3d pivy graphical display instance, point display for debug purposes
    :param minAngle: halting condition for incremental search accuracy.
    :return: [boolean indicator of rotational symmetry,
             centre point of rotationally symmetric feature,
             refined initial centrePoint,
             edgeState/maxState number of flanking local maxima]

    """
    # using centrePoint input, get the max 2 alignment values with standard rosette using horizontal axis projection,

    # note hard-coded values sensitive to scaling
    rotSymSpacingAngleMin = None
    if featureType == "localMax":
        rotSymSpacingAngleMin = 0.15 / localDisp(centrePoint, localCentroid)
    elif featureType == "localMin":
        rotSymSpacingAngleMin = 0.15 / localDisp(centrePoint, localCentroid)
    rotSymSpacingAngleMax = pi / 4
    incrementAngle = pi / 8
    # minAngle = 0.002 #0.002 # minAngle should be proportional to disp.. circle radius

    # rotSymSpacingAngle is another fudge factor # gets scrambled once the
    # rot-sym feature radius < than localDisp(centrePoint, localCentroid)
    # wider angle => tighter tolerance
    # also relative to the radius of the circle
    # CW & CCW spacing should be approx same as apertureAngle

    def getRadialEdges(
        lCentroid, centrePoint, torsionalAngle, apertureAngle, scene=None
    ):
        """
        Given a point on a radial edge, find two flanking points separated by torsionalAngle from lCentroid,
        a local maximum search returns the accurate radial edge/corner
        Does this algorithm work for identifying rotationally symmetric features at end of thin-walled tubes?
        :param lCentroid: shape local centroid
        :param centrePoint: initial point on a radially-symmetric edge0
        :param torsionalAngle:
        :param apertureAngle: angle of separation between rays defined as point-centroid and flanking_point-centroid
        :param scene: coin3d pivy graphical display instance
        :return: [3 return points, 3 displacements from centroid, number of local maxima around centrepoint]
        """

        fullCluster = False

        while not fullCluster:  # probably should be a minimum limiting value
            rotSymVectors = rotatedSurroundArray(
                lCentroid, centrePoint, apertureAngle, radRot=0.0
            )
            firstCluster = []
            for rsv in rotSymVectors:  # update for point array
                nearbyPoint = proj2object2(ShapeInstance, rsv, lCentroid, "nearest")

                if nearbyPoint:
                    firstCluster.append(nearbyPoint)
                else:
                    print("intersection failure at tangent")

            # if scene:
            #     displayPointFC(firstCluster, 'white', scene)

            dispCentre = localDisp(centrePoint, lCentroid)
            disps = localDisp(firstCluster, lCentroid)

            # if intersections are not all within validSphere, reduce apertureAngle and retry
            # validSphere is a sphere centered on centrePoint of cluster
            validSphere = 1.5 * dispCentre * sin(apertureAngle)
            apertureDisps = localDisp(firstCluster, centrePoint)
            fullCluster = all([ad < validSphere for ad in apertureDisps])
            if apertureAngle < pi * minAngle:
                # extrema of ellipsoids appear here
                return None, None, 1
            apertureAngle *= 2 / 3

        maxVectors = []
        maxDisps = []

        if featureType == "localMax" and disps:
            for d in range(0, len(disps)):
                if (
                    disps[(d - 1) % len(disps)] < disps[d] > disps[(d + 1) % len(disps)]
                ):  # <= for two identical numbers, artifact?
                    maxVectors.append(
                        [rotSymVectors[(d - 1) % len(disps)], rotSymVectors[d]]
                    )
                    maxDisps.append([disps[(d - 1) % len(disps)], disps[d]])

        elif featureType == "localMin" and disps:
            for d in range(0, len(disps)):
                if disps[(d - 1) % len(disps)] > disps[d] < disps[(d + 1) % len(disps)]:
                    maxVectors.append(
                        [rotSymVectors[(d - 1) % len(disps)], rotSymVectors[d]]
                    )
                    maxDisps.append([disps[(d - 1) % len(disps)], disps[d]])

        if len(maxVectors) > 2:
            return None, None, 3  # 3 local maxima
        if len(maxVectors) < 2:
            return None, None, 1  # 1 local maxima

        _maxPoints = []
        outputVectors = []
        outputDisps = []
        outPoint = None
        outVector = None
        outDisp = None
        # for mv in maxVectors:
        for mv in range(0, len(maxVectors)):
            v_0 = maxVectors[mv][0]
            v_1 = maxVectors[mv][1]
            d_0 = maxDisps[mv][0]
            d_1 = maxDisps[mv][1]
            stepAngle = torsionalAngle
            lastMidPoints = None
            while stepAngle > minAngle:
                midVectors = rotateCluster2(
                    lCentroid, centrePoint, [v_0, v_1], stepAngle
                )  # anticlockwise rotation
                stepAngle /= 2
                midPoints = []
                for mmv in midVectors:
                    lastMidPoints = midPoints  # retain in the event that d_0 or d_1 have max disp on exit
                    midPoint = proj2object2(
                        ShapeInstance, mmv, lCentroid, featureType
                    )  # , _scene)

                    if midPoint:
                        midPoints.append(midPoint)
                    else:
                        if scene and (len(midPoints) > 1):
                            displayPointFC(midPoints, "green", scene, "circlehollow9")

                midDisps = localDisp(
                    midPoints, lCentroid
                )  # this fails on inadequate point intersections.
                # print(midDisps)
                if len(midDisps) != 2:
                    print("intersections missing at surface tangent")
                    return None, None, 2
                    # pass
                d_0a = midDisps[0]
                d_1a = midDisps[1]
                v_0a = midVectors[0]
                v_1a = midVectors[1]

                if featureType == "localMax":
                    if (d_0a > d_0) and (d_0a > d_1) and (d_0a > d_1a):
                        v_1 = v_0a  # v_0 remains v_0
                        d_1 = d_0a
                        outPoint = midPoints[0]
                        outVector = midVectors[0]
                        outDisp = d_0a
                    if (d_1 > d_0) and (d_1 > d_0a) and (d_1 > d_1a):
                        d_0 = d_0a
                        v_0 = v_0a
                        outPoint = lastMidPoints[1]
                        outVector = v_1
                        outDisp = d_1
                    if (d_1a > d_0) and (d_1a > d_0a) and (d_1a > d_1):
                        v_0 = v_1
                        d_0 = d_1
                        v_1 = v_1a
                        d_1 = d_1a
                        outPoint = midPoints[1]
                        outVector = midVectors[1]
                        outDisp = d_1a
                elif featureType == "localMin":
                    if (d_0a < d_0) and (d_0a < d_1) and (d_0a < d_1a):
                        v_1 = v_0a  # v_0 remains v_0
                        d_1 = d_0a
                        outPoint = midPoints[0]
                        outVector = midVectors[0]
                        outDisp = d_0a
                    if (d_1 < d_0) and (d_1 < d_0a) and (d_1 < d_1a):
                        d_0 = d_0a
                        v_0 = v_0a
                        outPoint = lastMidPoints[1]
                        outVector = midVectors[1]
                        outDisp = d_1
                    if (d_1a < d_0) and (d_1a < d_0a) and (d_1a < d_1):
                        v_0 = v_1
                        d_0 = d_1
                        v_1 = v_1a
                        d_1 = d_1a
                        outPoint = midPoints[1]
                        outVector = midVectors[1]
                        outDisp = d_1a

            if outPoint and outVector and outDisp:
                _maxPoints.append(outPoint)
                outputVectors.append(outVector)
                outputDisps.append(outDisp)

        if (len(outputDisps) < 2) or (len(_maxPoints) < 2) or (len(outputVectors) < 2):
            return None, None, 1

        return _maxPoints + [centrePoint], outputDisps + [dispCentre], 2

    def refineRadialEdgesMidpoint(lCentroid, flankingVectors, scene=None):
        """
        Correct position of midpoint on a radial edge.
        getRadialEdges() will iterate points (maxPoints) to each side of an initial point (centrePoint) on a
        radial edge, this function iterates the original point so that it is positioned on the edge with similar
        accuracy. The search rotates the two edge points 90 degrees, then tests the displacement of the median point
        between these rotated points for greatest displacement from shape local centroid.

        :param lCentroid: shape local centroid
        :param flankingVectors: points flanking central point (max disp)
        :param scene: coin3d pivy graphical display instance, point display for debug purposes
        :return: shifted centrepoint, centrePoint displacement from shape centroid
        """
        fullIntersection = False

        rotationPoint = medianPoint(flankingVectors)  # find midpoint of supplied Points
        flankingVectors = rotateCluster2(
            lCentroid, rotationPoint, flankingVectors, pi / 2
        )  # anticlockwise rotation
        # centreVectors must be rotated about an axis orthogonal defined by localCentroid, rotationPoint
        minStep = rotSymSpacingAngle * np.sin(minAngle)
        while not fullIntersection:  # rotate midVectors pi/2
            newCentrePoints = []
            for fv in flankingVectors:
                newCentrePoint = proj2object2(
                    ShapeInstance, fv, lCentroid, featureType
                )  # , scene)

                if newCentrePoint:
                    newCentrePoints.append(newCentrePoint)
                # else:
                #     if featureType == "localMax":
                #         newCentrePoints.append(rotationPoint)

            if scene:
                displayPointFC(newCentrePoints, "cyan", scene)

            if len(newCentrePoints) == 2:
                fullIntersection = True
            else:
                flankingVectors[0] = medianPoint([flankingVectors[0], rotationPoint])
                flankingVectors[1] = medianPoint([flankingVectors[1], rotationPoint])
                if scene:
                    displayPointFC(newCentrePoints, "red", scene, "star7")

        newCentreDisps = localDisp(newCentrePoints, lCentroid)
        dc_0 = newCentreDisps[0]
        dc_1 = newCentreDisps[1]
        cp_0 = newCentrePoints[0]
        cp_1 = newCentrePoints[1]
        cp_m = medianPoint(
            [cp_0, cp_1]
        )  # ------------------------------------------------ fallthru issue
        dc_m = localDisp(cp_m, lCentroid)

        while abs(localDisp(cp_0, cp_1)) > minStep:
            cv_m = medianPoint([cp_0, cp_1])
            cp_m = proj2object2(ShapeInstance, cv_m, lCentroid, "nearest")  # , _scene)

            if not cp_m:
                print("point skipped on centrePoint search")
                return None, None

            else:
                # if _scene:
                #     displayPointFC(cp_m, 'lime', _scene)

                dc_m = localDisp(cp_m, lCentroid)

                if featureType == "localMax":
                    if dc_0 < dc_1:
                        if dc_0 < dc_m:
                            cp_0 = cp_m
                            dc_0 = dc_m
                        elif dc_m < dc_0:
                            # easiest to bail out
                            return None, None
                            # pass
                    elif dc_1 < dc_0:
                        if dc_1 < dc_m:
                            cp_1 = cp_m
                            dc_1 = dc_m
                        elif dc_m < dc_1:
                            return None, None
                            # pass
                    elif dc_0 == dc_1:  # rare circumstance causing loop
                        centrePoint = cp_m
                        dispCentre = dc_m
                        return centrePoint, dispCentre

                elif featureType == "localMin":
                    if dc_1 < dc_0:
                        if dc_m < dc_0:
                            cp_0 = cp_m
                            dc_0 = dc_m
                        elif dc_0 < dc_m:
                            return None, None
                    if dc_0 < dc_1:
                        if dc_m < dc_1:
                            cp_1 = cp_m
                            dc_1 = dc_m
                        elif dc_1 < dc_m:
                            return None, None
                    if dc_0 == dc_1:  # rare circumstance causing loop
                        centrePoint = cp_m
                        dispCentre = dc_m
                        return centrePoint, dispCentre

        centrePoint = cp_m
        dispCentre = dc_m

        # if scene:
        #     displayPointFC(cp_m, 'red', scene)

        return centrePoint, dispCentre

    # increase rotSymSpacingAngle... import from edge search?
    # linear spacing increase about centre point

    # rotSymSpacingAngle = rotSymSpacingAngleMin
    withinTol = True
    rotSymFeature = False
    dispPlanarTols = None
    dispCentre = None

    CWwithinTol = True
    CCWwithinTol = True

    CCWdispCircularTol = 1
    CWdispCircularTol = 1
    CCWdispPlanarTol = 1
    CWdispPlanarTol = 1

    # advance search of ridge simultaneously clockwise and counterclockwise,
    # allows partial radially symmetric features to be identified

    flankingPointAngle = pi / 24
    rotSymSpacingAngle = flankingPointAngle

    edgePointTriad, edgeDisps, maxState = getRadialEdges(
        localCentroid, centrePoint, incrementAngle, rotSymSpacingAngle
    )

    if (edgePointTriad is None) or (edgeDisps is None) or (len(edgePointTriad) < 3):
        return False, None, None, maxState
    else:
        allEdgeDisps = edgeDisps
        allEdgePoints = edgePointTriad

        # equal radius sanity check
        printVerbose("allEdgeDisps std dev radius: " + str(np.std(allEdgeDisps)))

        # if scene:
        #     displayPointFC(edgePointTriad, 'blue', scene, 'star7')

        # use a faster trigonometric solution?
        centrePoint, dispCentre = refineRadialEdgesMidpoint(
            localCentroid, edgePointTriad[0:2]
        )

        if centrePoint is None:
            return False, None, None, 2
        edgePointTriad[2] = centrePoint
        CCWpoint = edgePointTriad[0]
        CWpoint = edgePointTriad[1]
        allEdgePoints.append(centrePoint)

    # purpose of following loop is to create points positioned on the rotationally-symmetric feature at
    # ever-widening angles from a central point identified by refineRadialEdgesMidpoint()
    # more points on rot-sym feature allows more accurate definition of a centre point

    while (
        (flankingPointAngle < rotSymSpacingAngleMax)
        and withinTol
        and (CCWwithinTol or CWwithinTol)
    ):
        # get the circle passing through centrePoint and two other maxima
        centreCircle, radiusCircle = pointsToCircleRadiusCentre(
            CCWpoint, centrePoint, CWpoint
        )

        # if scene:
        #     displayPointFC(centreCircle, 'red', scene, 'circlehollow9')
        # print("centreCircle: "+str(centreCircle))

        # rotate found centre and flanking points through axis of rotationally symmetrical feature
        if CCWwithinTol:
            CCWflankingVectors = rotateCluster2(
                localCentroid, centreCircle, edgePointTriad[:2], flankingPointAngle
            )  # anticlockwise rotation

            CCWflankingPoints = [
                proj2object2(ShapeInstance, ccwfv, localCentroid, featureType, scene)
                for ccwfv in CCWflankingVectors
            ]

            # if scene:
            #     displayPointFC(CCWflankingPoints, 'white', scene) # , 'star7')

            if None in CCWflankingPoints:
                CCWwithinTol = False
            else:
                testCCWpoint, testCCWpointDisp = refineRadialEdgesMidpoint(
                    localCentroid, CCWflankingPoints, scene
                )
                if testCCWpoint:
                    CCWdispCircularTol = abs(dispCentre - testCCWpointDisp)
                    # circle threshold can only flag radii *greater* than |localCentroid, centrePoint|
                    # because smaller radii exist at further displacement of circle centre from localCentroid
                    # but circle centre is deduced from the points under scrutiny
                    # won't catch corners, see allEdgeDisps

                    # coplanar check to prevent nearby ridges attracting radial edge searches, flat disc problem.
                    CCWdispPlanarTol = pointIn3PointPlane(
                        CCWpoint, CWpoint, centrePoint, testCCWpoint
                    )
                    CCWpoint = testCCWpoint
                    CCWpointDisp = testCCWpointDisp
                    allEdgeDisps.append(CCWpointDisp)
                    allEdgePoints.append(CCWpoint)
                    if scene:
                        displayPointFC(CCWpoint, "orange", scene, "star7")
                else:
                    withinTol = False

        if CWwithinTol:
            CWflankingVectors = rotateCluster2(
                localCentroid,
                centreCircle,
                edgePointTriad[:2],
                (2 * pi - flankingPointAngle),
            )

            CWflankingPoints = [
                proj2object2(ShapeInstance, cwfv, localCentroid, featureType, scene)
                for cwfv in CWflankingVectors
            ]

            # if scene:
            #     displayPointFC(CWflankingPoints, 'lime', scene) # , 'star7')

            if None in CWflankingPoints:
                CWwithinTol = False
            else:
                testCWpoint, testCWpointDisp = refineRadialEdgesMidpoint(
                    localCentroid, CWflankingPoints[:2], scene
                )
                if testCWpoint:
                    CWdispCircularTol = abs(dispCentre - testCWpointDisp)
                    CWdispPlanarTol = pointIn3PointPlane(
                        CCWpoint, CWpoint, centreCircle, testCWpoint
                    )
                    CWpoint = testCWpoint
                    CWpointDisp = testCWpointDisp
                    allEdgeDisps.append(CWpointDisp)
                    allEdgePoints.append(CWpoint)
                    if scene:
                        displayPointFC(CWpoint, "green", scene, "star7")
                else:
                    withinTol = False

        withinCircularTol = not (CCWdispCircularTol > 0.01 or CWdispCircularTol > 0.01)
        # centroid centering issue----------FUDGE FACTOR proportional to scale
        withinPlanarTol = not (
            (abs(CCWdispPlanarTol) > 0.1) or (abs(CWdispPlanarTol) > 0.1)
        )
        # centroid centering issue------------------------------FUDGE FACTOR

        flankingPointAngle += (
            pi / 24
        )  # if method fails after one+ successful round, return last good set?
        withinTol = withinCircularTol and withinPlanarTol
        if withinTol:
            rotSymFeature = True
        if maxState != 2:
            rotSymFeature = False

    # centreCircle, radiusCircle = pointsToCircleRadiusCentre(CCWpoint, centrePoint, CWpoint)

    # get deviations between allEdgeDisps + centrePoint (could also use export residuals or RMSE from sphereFit() ?)
    dispCircularTols = [abs(dispCentre - aed) for aed in allEdgeDisps]
    # dispCircularTols = [d/max(dispCircularTols) for d in dispCircularTols]
    # rotSymFeature = not any([d > 0.025 for d in dispCircularTols]) # was 0.005 --------------FUDGE FACTOR

    rotSymFeature = (
        np.mean(dispCircularTols) < 0.025
    )  # --------------------------------------FUDGE FACTOR

    # catch off-axis rot-sym point series (artifact of increasing localCentroid accuracy)
    if (dispCentre + 0.005 < max(allEdgeDisps)) and (featureType == "localMax"):
        # catch case of non-maximum point on ridge
        return False, None, None, 2

    if (dispCentre + 0.005 < max(allEdgeDisps)) and (featureType == "localMin"):
        # catch case of non-maximum point on groove
        pass

    if dispPlanarTols:  # case of dispPlanarTols at large divergence indicates a corner
        if max(dispPlanarTols) > 0.05:
            maxState = 3

    if scene:
        displayPointFC(centreCircle, "cyan", scene, "cross5")

    # move co-axial test to final pointsToCurves2() as cumulative localCentroid accuracy is sensitive to a low
    # intersection point distribution

    # test if identified symmetric feature does not run through an axis co-linear with local origin
    symmetricAxisDeviation = abs(
        localDisp(centreCircle, localCentroid) ** 2
        + radiusCircle**2
        - dispCentre**2
    )

    # if featureType == "localMin":
    #     printVerbose("localMin feature rotational symmetry test: " + str(symmetricAxisDeviation))

    if abs(symmetricAxisDeviation) > 0.05:
        if featureType == "localMin":  # case of minimum circle
            printVerbose(
                "localMin feature rotational symmetry test off-axis origin deviation: "
                + str(symmetricAxisDeviation / dispCentre**2)
            )
        else:
            printVerbose(
                "rotSymTest6 rotational symmetry off-axis origin deviation: "
                + str(symmetricAxisDeviation / dispCentre**2)
            )
    if (
        symmetricAxisDeviation / dispCentre**2
    ) > 0.1:  # ------------------SCALE/RADIUS SENSITIVE FUDGE FACTOR
        return False, None, centrePoint, maxState

    # a corner with the centrepoint on the origin shows up as a ridge
    if (
        localDisp(centrePoint, Point(0, 0, 0)) < 0.001
    ):  # -------------------------------FUDGE FACTOR
        return False, None, None, maxState

    # equal radius sanity check
    # note that circleFit() for localMin feature will return a wild centreCircle if radius not caught here
    radiusCheck = [localDisp(aep, centreCircle) for aep in allEdgePoints]
    # printVerbose("std dev radius: " + str(np.std(radiusCheck)))
    if np.std(radiusCheck) > 0.01:  # -------------------------------FUDGE FACTOR
        return False, None, None, maxState

    if scene:
        displayPointFC(centreCircle, "cyan", scene, "star7")
    RC = radiusCircle
    centreCircle, radiusCircle = circleFit(allEdgePoints)
    print("Radius point deviation: ", abs(RC - radiusCircle))

    # if featureType == "localMin":
    #     _ = 1

    # displayPointFC(centreCircle, 'cyan', scene, 'circlehollow9')
    if scene:
        scene.saveIV(scene.name)

    # weed out points that are on external or internal radially-symmetric features,
    # but have centre points that are distant from any axis passing through local shape centroid

    return rotSymFeature, centreCircle, centrePoint, maxState  # maxPoints[0]


def sphereOrPoint2(dispCentre, dispCluster):
    """
    Test for points deviating from the mean sphericity by more than a sphereCondition constant (number noise guess).

    :param dispCentre: local_centroid
    :param dispCluster: set of scalars representing point-local_centroid displacements
    :return: boolean
    """
    #
    sphereTolerance = (
        2e-7 * dispCentre
    )  # should be lower than point termination tolerance
    clusterExTolerance = any(
        [abs(d - dispCentre) > sphereTolerance for d in dispCluster]
    )
    return not clusterExTolerance


def sharpRadiusFeature(dispCentre, dispCluster):
    """
    Test dispCentre deviating from the mean dispCluster sphericity > sharpRadiusConst factor

    :param dispCentre: local_centroid
    :param dispCluster: set of scalars representing point-local_centroid displacements
    :return: boolean
    """

    sphereTolerance = 2e-7 * dispCentre
    clusterExTolerance = all(
        [abs(d - dispCentre) > sphereTolerance for d in dispCluster]
    )
    return clusterExTolerance


def getMaxMin(
    ShapeInstance,
    centrePoint,
    localCentroid,
    stepAngle,
    stopCondition=0.001,
    maxIter=15,
    featureType="localMax",
    scene=None,
):
    """
    Given a starting point, iterate to nearest maxima/minima based on a nearest neighbour algorithm.

    Thie method used to search for the most simple type of registration feature, a local
    maximum point is described here. The search algorithm is similar to a hill-climbing algorithm with
    adaptive step size. The search for a maximum registration feature starts from a seed point and
    progresses in the direction of steepest relative gradient. Unlike steepest gradient descent,
    the function describing the CAD model surface is unknown, therefore the surface derivative is also unknown.
    This means the relative gradient must be sampled from neighbouring points, these neighbouring points are
    a ring of eight points sampled around the initial point.
    The initial ray intersecting the seed point may be rotated in eight cardinal directions to
    create these new surface intersection points, similar to a compass rosette (neighbourGrid,
    rotationMatrix -> rotatedSurroundArray()). The initial angle is half that of the angle between vectors emanating
    from the Deserno projection, allowing each seed point to encompass the entire area of a local search region within
    two iterations (getModelFeatures2, searchFeatures, getMaxMin).

    This angle at which surrounding points are projected is halved at each iteration to reduce
    the diameter of the search pattern when the central point has a higher value than the
    surrounding rosette of points. Where one of the neighbouring points has the highest
    value, it is assigned as the next central point, but the angle is not subdivided for the next
    rosette of points. This strategy is combined with a rotation of the points rosette around an
    axis co-linear with the projected ray through the centre point on each rosette generation
    (rotateCluster -> rotatedSurroundArray()). The rosette is rotated by  an angle of clusterRotate,
    such that the rosette points always occupy a novel rotational angle on each iteration.
    This additional operation allows the search to progress along sharp edges and at corners where
    the limited resolution of eight surrounding points might otherwise miss the highest local gradient.
    These iterations continue until either a feature is located, or the angle between points is lower
    than a set minimum (stopCondition) or the number of iterations reaches an absolute limit (maxIter).

    :param ShapeInstance: CAD shape object to be tested
    :param centrePoint: nucleating point on surface
    :param localCentroid: local centroid of shape reference, (geometric, not barycentre of points)
    :param stepAngle: angle between centrePoint and neighbouring search points in rosette measured through localCentroid
    :param stopCondition: minimum value for stepAngle, function precision
    :param maxIter: maximum number of permissible iteration
    :param featureType: "localMax", "localMin", convex and concave feature respectively
    :param scene: coin3d pivy graphical display instance, point display for debug purposes
    :return: [centrePoint: refined 3D point identified with feature,
              featureIdentity: feature type,
              circleCentrePoint: centre of rotationally symmetrical feature identified by rotSymTestX()]
    """

    sphereCondition = (
        5e-4  # 0.00005 # minimum difference between height for more than one point
    )
    itern = 0
    peakness = 1
    clusterAngle = pi / 4
    lastDispCentre = 0.0
    checkDisp = 0.0

    # stepAngle & stopCondition should be related to scale?

    # if scene:
    #     displayPointFC(centrePoint, 16711680, scene)  # 16711680 blue

    centreMaxMin = (
        True  # determine is central point maximum or minimum disp for point validation
    )
    clusterRotate = 0
    featureIdentity = "unassigned"
    dispCentre = None
    dispCluster = None

    while (stepAngle > stopCondition) and (itern < maxIter) and centrePoint:
        clusterPoints = []
        itern += 1
        # Angle is subdivision of initial local search space sector over search iteration
        # For each search point, generate N neighbouring intersection points from
        # intersecting rays of step angle offset from current search point.

        surroundingClusterVectorsR = rotatedSurroundArray(
            localCentroid, centrePoint, stepAngle, clusterRotate
        )
        clusterRotate += phi / 8

        for scv in surroundingClusterVectorsR:  # update for point array
            nearbyPoint = proj2object2(ShapeInstance, scv, localCentroid, "nearest")
            if nearbyPoint:
                clusterPoints.append(nearbyPoint)

        if clusterPoints:
            pass
            # if scene:
            #     displayPointFC(centrePoint, 'red', scene)
            #     restOfClusterPoints = [rocp for rocp in clusterPoints if rocp is not centrePoint]
            #     displayPointFC(restOfClusterPoints, 'lime', scene)
        else:
            return None, "noCluster", featureIdentity

        dispCluster = localDisp(clusterPoints, localCentroid)
        dispCentre = localDisp(centrePoint, localCentroid)

        # if scene:
        #     displayPointFC(centrePoint, 'cyan', scene)  # 16711680 blue

        # Update current search point to highest centroid-boundary surface intersection value within current
        # search point and search neighbours. Continue to search if there is a single point with a greater quality
        # (disp from centroid, curvature, etc.) terminate search if more than one point in set shares greater quality
        # (use stopCondition or Gaussian noise?)
        # (note "peakness" conceptually equivalent to "edgeState" ("maxState" within rotSym6() )

        if featureType == "localMax":
            # If current value is maxima, stop search and store value as local feature point.
            dispClusterMax = max(dispCluster)
            if (abs(dispClusterMax - dispCentre) > 2 * eps) and (
                dispCentre > dispClusterMax
            ):
                # featureIdentity = "maxPoint"
                # case of initial point retaining a higher centroid displacement than neighbours
                # reduce step angle and repeat loop
                stepAngle = stepAngle / 2  # update step size as neighbouring arc angle
                centreMaxMin = True
            else:
                peakness = sum(
                    [(dispClusterMax - dc) < sphereCondition for dc in dispCluster]
                )
                # replace peakness measure with symmetry measure of clusterDisp? NO
                centreMaxMin = False
                if (peakness < 3) or (
                    itern == 1
                ):  # return 'None' if points are ambiguous
                    # If two or more point values are maxima, do not subtend arc step and repeat
                    # move centrePoint to cluster point with highest displacement
                    centrePoint = clusterPoints[dispCluster.index(dispClusterMax)]
                    if (peakness < 2) and (
                        abs(dispClusterMax - checkDisp) > sphereCondition
                    ):
                        # don't reduce stepAngle, continue to iterate to endpoint.
                        # track increasing radius displacement to detect rotationally symmetric ridge/groove
                        # featureIdentity = "maxPoint"

                        pass

                        # if scene:
                        #     displayPointFC(centrePoint, 'red', scene)
                        #     restOfClusterPoints = [rocp for rocp in clusterPoints if rocp is not centrePoint]
                        #     displayPointFC(restOfClusterPoints, 'green', scene)

                    else:  # peakness == 2
                        # has to be 3 in a row through the centrePoint?
                        # featureIdentity = "rotSymRidge" # is this redundant?
                        stepAngle = (
                            stepAngle / 2
                        )  # update step size as neighbouring arc angle, ridges have to time out

                        if scene:
                            displayPointFC(centrePoint, "orange", scene)
                            restOfClusterPoints = [
                                rocp
                                for rocp in clusterPoints
                                if rocp is not centrePoint
                            ]
                            displayPointFC(restOfClusterPoints, "lime", scene)

                else:  # if > 3 points within eps => sphere condition
                    # If neighbour value is single maxima, set current to neighbour
                    # featureIdentity = "sphere"
                    centreMaxMin = True
                    stepAngle = (
                        stepAngle / 2
                    )  # update step size as neighbouring arc angle
                    # no distinction

                    if scene:
                        displayPointFC(centrePoint, "teal", scene)
                        restOfClusterPoints = [
                            rocp for rocp in clusterPoints if rocp is not centrePoint
                        ]
                        displayPointFC(restOfClusterPoints, "white", scene)

        elif featureType == "localMin":
            # If current value is minima, stop search and store value as local feature point.
            dispClusterMin = min(dispCluster)
            if (abs(dispClusterMin - dispCentre) > 2 * eps) and (
                dispCentre < dispClusterMin
            ):
                # case of initial point retaining a lower centroid displacement than neighbours
                # reduce step angle and repeat loop
                featureIdentity = "minPoint"
                stepAngle = stepAngle / 2  # update step size as neighbouring arc angle
                centreMaxMin = True
            else:
                peakness = sum(
                    [(dc - dispClusterMin) < sphereCondition for dc in dispCluster]
                )
                # replace peakness measure with symmetry measure of clusterDisp?
                centreMaxMin = False
                if peakness < 3:  # return 'None' if points are ambiguous
                    # If two or more point values are maxima, do not subtend arc step and repeat
                    # move centrePoint to cluster point with highest displacement
                    centrePoint = clusterPoints[dispCluster.index(dispClusterMin)]
                    if (peakness < 2) and (
                        abs(dispClusterMin - checkDisp) > sphereCondition
                    ):
                        # don't reduce stepAngle, continue to iterate to endpoint.
                        # track increasing radius displacement to detect rotationally symmetric ridge/groove
                        featureIdentity = "minPoint"

                        # if scene:
                        #     displayPointFC(centrePoint, 16776960, scene)  # cyan

                    else:  # peakness == 2
                        # has to be 3 in a row through the centrePoint?
                        # print("ridge condition") # rotationally symmetrical ridge =>
                        #   flag and test number of points within eps before return
                        # ridgeCondition = True # keep track of disp once flagged?
                        featureIdentity = "rotSymGroove"
                        stepAngle = (
                            stepAngle / 2
                        )  # update step size as neighbouring arc angle, ridges have to time out
                        centrePoint = clusterPoints[dispCluster.index(dispClusterMin)]
                        # if scene:
                        #     displayPointFC(centrePoint, 255, scene)

                else:  # if > 3 points within eps => sphere condition
                    # If neighbour value is single maxima, set current to neighbour
                    # centrePoint = None # exit function with inconclusive search result
                    centreMaxMin = True  # only valid for planes on minimum search
                    centrePoint = clusterPoints[dispCluster.index(dispClusterMin)]
                    featureIdentity = "sphere"  # slope, little potential
                    # could also be a plane - no measure of concavity
                    # when stopCondition for stepAngle is set too small
                    # small distant curves become sphere slopes

            # peakness = sum([(dc - dispClusterMin) < sphereCondition for dc in dispCluster])

    if scene:
        scene.saveIV(scene.name)

    # determine if point or sphere
    sphereFeature = sphereOrPoint2(dispCentre, dispCluster)
    if not sphereFeature and featureIdentity == "sphere":
        featureIdentity = "unassigned"
    if sphereFeature:
        centreMaxMin = True
        circleCentrePoint = None
        featureIdentity = "sphere"
        if featureType == "localMin":
            print("searching for an inverted sphere? Get context (dimple?)")
        return centrePoint, featureIdentity, circleCentrePoint

    # if featureType == "localMin":
    #     if scene:
    #         displayPointFC(centrePoint, 'red', scene)  # 16711680 blue

    # should also return non-symmetrical ridge assessment
    rotSymFeature, circleCentrePoint, ridgePoint, edgeState = rotSymTest6(
        ShapeInstance, centrePoint, localCentroid, featureType, 0.01, scene
    )
    if circleCentrePoint and scene:
        displayPointFC(circleCentrePoint, "cyan", scene, "cross7")

    if rotSymFeature:
        if featureType == "localMax":
            return ridgePoint, "rotSymRidge", circleCentrePoint
        elif featureType == "localMin":
            return ridgePoint, "rotSymGroove", circleCentrePoint

    # edgeState (maxState in rotSymTest) is additional discriminating geometry data

    if edgeState == 2:
        if ridgePoint:
            # rotSymTest5 returns maxPoint on a ridge if not circular
            if featureType == "localMax":
                return ridgePoint, "maxPoint", None
            elif featureType == "localMin":
                return ridgePoint, "minPoint", None
        else:
            return (
                None,
                "ridge",
                None,
            )  # --------non-circular is not failed circular detection

    if edgeState == 3:  # and not rotSymFeature:
        if centreMaxMin:
            # rotSymTest5 returns maxPoint on a ridge if not circular
            if featureType == "localMax":
                return centrePoint, "maxPoint", None
            # groove behaviour less certain
            elif featureType == "localMin":
                return centrePoint, "minPoint", None
        else:
            return None, "ridge", None

    if edgeState == 1:
        # terminating at a local maximum or minimum, also maxima/minima of ellipsoid
        if (featureType == "localMax") and centreMaxMin:
            # print("single point (localMax) - spherical surface/zero-radius corner")
            # fails on quasi spherical surfaces, large torii
            # however rotated/translated zero radius corners also arrive here
            # if sharpRadiusFeature(dispCentre, dispCluster):
            #     print(centrePoint)
            return centrePoint, "maxPoint", None  # required for small radii
        elif (featureType == "localMin") and centreMaxMin:
            return centrePoint, "minPoint", None

    # weed out seedpoints that never terminated at a feature
    #  - SHOULD NOT OCCUR IF ridgeCondition SEARCHES DON'T REDUCE stepAngle
    # if (itern >= maxIter) or (centreMaxMin == False):
    if not centreMaxMin:
        # print("check case of localMin curve cluster not straddling minima curve")
        if centrePoint:  # and not ridgeCondition: #
            # if scene:
            #     displayPointFC(centrePoint, 'maroon', scene)  # brown

            # centrePoint = None # check here for sphere condition?
            return None, "unassigned", None

    print("fallthru")
    return None, "unassigned", None
    # return centrePoint, featureIdentity, circleCentrePoint


def centroidTranslation(ShapeInstance, giveUp=100, scene=None):
    """
    Initial surface detection and centroid translation estimation.
    Starting from the Cartesian origin, emit rays until an interception is detected.
    If no surface is
    Relocate to median point of surface interceptions and repeat.

    To identify a model centroid, the model must first be detected. As the methods available
    for model discovery are limited to point intersections with projected rays, an array of rays
    is projected in all directions from the origin to intersect the model surface.
    The Deserno regular method is used. This method is sensitive to the density of
    projected vectors and the scale of the geometry model. Any surface intersections indicate
    the presence of a model, the centroid of this model may then be estimated.
    If the Cartesian values of existing point intersections are averaged to form a mean point,
    this position forms a closer estimate to the model shape centroid, a similar spherical
    projection of rays is likely to intersect more of the model surface, generating a
    correspondingly more accurate estimate for a centroid. This process is iteratively repeated
    until the distance between centroid estimate adjustments falls below a set threshold value
    (centroidTranslation). Where the target model is a number of separate model
    geometries, the centroid will represent the mean points of intersection for all surfaces

    :param ShapeInstance: CAD shape object to be tested
    :param giveUp: finite search limit
    :param scene: coin3d pivy graphical display instance, point display for debug purposes
    :return: ShapeInstance.centroid value
    """

    # issues with unevenly distributed point intersections
    # Set estimated centroid to Cartesian space origin.

    centroid = Point(0.0, 0.0, 0.0)
    centroidStopCondition = 0.075
    # centroidStopCondition = 0.025
    # Deserno Sphere or Tammes Sphere do not work well on long or thin
    resolution = 40
    rotn = 0.0
    maxResolution = 80
    giveUp = resolution * 3
    pointList = []

    # Using an equal divison of an encompassing sphere (Tammes sphere),
    # emit rays from estimated centroid and test for intersections with object surfaces

    DS = None
    while (resolution < giveUp) and not pointList:
        DS = DesernoSphere(resolution)
        # DS = TammesSphere(resolution)

        # if intersections do not exist, repeat with higher density of rays until giveUp
        for ds in DS:
            p = projectVector(centroid, ShapeInstance, ds)
            if p:
                [pointList.append(pp) for pp in p if pp not in pointList]
                # scene.addLine(centroid, ds)

        resolution += 10

    # in cases of complex objects, e.g. toroids, repeat centroid search until
    # movement between updated is a fraction of initial displacement.
    # if intersections exist, update estimated centroid to mean value of intersections
    if pointList:
        centroid = medianPoint(ShapeInstance.surfacePoints)
        DS = offsetPoints(DS, centroid)
        # firstCentroidDisp = localDisp(centroid, Point(0.0,0.0,0.0))
        centroidDisp = 1.0  # firstCentroidDisp
        while centroidDisp > centroidStopCondition:
            # repeat TammesSphere/DesernoSphere projection distribution calculation to get reasonable centroid

            # distributedPoints = []
            # pointSpacing = np.array([])

            for ds in DS:
                pointList = projectVector(centroid, ShapeInstance, ds)
            #     if pointList:
            #         distributedPoints = distributedPoints + pointList
            #
            # # remove all points spaced less than the mean displacement apart
            # # this only works in one plane of the DesernoSphere, which has a helical point distribution.
            # for pi in range(1, len(distributedPoints) - 1):
            #     pointSpacing = np.append(pointSpacing, localDisp(distributedPoints[pi], distributedPoints[pi - 1]))
            # meanDisp = np.mean(pointSpacing)
            # pointSpacing = np.append(pointSpacing, meanDisp + 1)
            # distributedPoints = [distributedPoints[dp] for dp in np.where(pointSpacing[:] > meanDisp)[0]]
            # centroid = meanPoint(distributedPoints)

            lastCentroid = centroid
            # meanCentroid = meanPoint(ShapeInstance.surfacePoints)
            centroid = medianPoint(ShapeInstance.surfacePoints)
            printVerbose(
                "mean - median centroid: "
                + str(localDisp(centroid, meanPoint(ShapeInstance.surfacePoints)))
            )

            DS = DesernoSphere(resolution, rotn)
            # DS = TammesSphere(resolution)
            DS = offsetPoints(DS, centroid)
            # shift DS to a point midway between the greatest outlier and centroid?

            centroidDisp = localDisp(centroid, lastCentroid)
            resolution += 5
            rotn += phi

            if scene:
                displayPointFC(centroid, "lime", scene)

        medianCentroid = medianPoint(ShapeInstance.surfacePoints)
        printVerbose(
            "median centroid value: x: {:f}, y: {:f}, z: {:f}".format(
                medianCentroid.x, medianCentroid.y, medianCentroid.z
            )
        )
        ShapeInstance.centroid = centroid


def pointsToCircleRadiusCentre(P1, P2, P3):
    """
    Return centre coordinates and radius from three points on circle arc, barycentric coordinates of the circumcentre
    see https://stackoverflow.com/questions/20314306/find-arc-circle-equation-given-three-points-in-space-3d

    :param P1: point
    :param P2: point
    :param P3: point
    :return: [centre point, radius scalar]
    """

    A = np.array([P1.x, P1.y, P1.z])
    B = np.array([P2.x, P2.y, P2.z])
    C = np.array([P3.x, P3.y, P3.z])
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)
    s = (a + b + c) / 2
    radius = a * b * c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    b1 = a * a * (b * b + c * c - a * a)
    b2 = b * b * (a * a + c * c - b * b)
    b3 = c * c * (a * a + b * b - c * c)
    centre = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
    centre /= b1 + b2 + b3
    if np.shape(centre)[0] != 3:
        pass
    centre = Point(centre[0], centre[1], centre[2])
    return centre, radius


def circleFit(p, scene=None):
    """
    Least squares fit for a circle defined by points on periphery.
    3D sphere fitting based on Coope method, see
    https://ir.canterbury.ac.nz/handle/10092/11104, see also
    https://github.com/madphysicist/scikit-guess/blob/master/src/skg/nsphere.py
    Sphere centre is subsequently orthogonally projected to a plane defined
    through circle points.
    Note SVD speed observation in,
    https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points

    :param p: list of Cartesian Points type
    :param scene: coin3d pivy graphical display instance, point display for debug purposes
    :return: c2, circle centre point, r, circle radius
    """

    x = np.asfarray(p)
    x = x.reshape(-1, 3)

    B = np.empty((x.shape[0], 4), dtype=x.dtype)
    X = B[:, :-1]
    X[:] = x
    B[:, -1] = 1

    d = np.square(X).sum(axis=-1)
    y, *_ = np.linalg.lstsq(B, d, rcond=None)

    c = 0.5 * y[:-1]  # best fit sphere centre
    r = y[-1] + np.square(c).sum()  # squared radius value

    if scene:
        displayPointFC(Point(x=c[0], y=c[1], z=c[2]), "green", scene, "circlehollow9")

    # SVD circle plane estimation
    C = np.empty((x.shape[0], 3), dtype=x.dtype)
    C[:] = x
    centroid = C.mean(axis=0)
    m = C - centroid[None, :]
    U, S, Vt = np.linalg.svd(m.T @ m)  # plane eqn U, ax + by + cz + d = 0
    normal = U[:, -1]
    originDisp = normal @ centroid
    # RMSE = np.sqrt(S[-1] / len(C))
    cp = np.hstack([normal, -originDisp])
    n = cp[:-1]  # plane normal [a, b, c] from plane eqn ax + by + cz + d = 0
    t = (np.dot(c, n) + cp[-1]) / np.dot(n, n)
    c2 = c - n * t

    if scene:
        displayPointFC(Point(x=c2[0], y=c2[1], z=c2[2]), "red", scene, "circlehollow9")

    r = sqrt(abs(r - np.square(c2 - c).sum()))
    c2 = Point(x=c2[0], y=c2[1], z=c2[2])
    return c2, r


def pointsToBins(featurePointList, localCentroid, minCosResolution):
    """
    If a surface is subjected to multiple searches and the same features are discovered
    each time, then it is likely that an adequate number of searches have been conducted.
    But if the same number of searches discover different points each time, it is
    less likely that all registration features have been discovered.
    This may be described as a binomial probability distribution, e.g. two searches in a region with a
    single feature will always return the same feature, but two searches in a region with
    ten features each having an equal probability of discovery will only have a 0.1 chance of
    encountering the same feature twice in a row. In the implementation used for testing, a
    ratio of the number of features discovered multiple times against the total number of
    features found is used to evaluate a termination condition (pointsToBins2, pointsToCurves2).
    This approach is sensitive to the values of termination condition used, models may have
    features that are only revealed at a higher number of searches. When models are evaluated
    for registration features using differing parameters such as higher density of seed points
    or a higher confidence threshold, they may present an unmatched number of features that
    preventing simple matching. This issue is ameliorated by the use of preliminary searches
    that use both registration feature type and centroid displacement to detect a potential
    shape src.

    :param featurePointList: list of unsorted points indicating a feature
    :param localCentroid: CAD surface shape local centroid
    :param minCosResolution: minimum cosine distance between adjacent points distinguishing unique points
    :return: [featureCluster: list of features with duplicates removed
              foundRatio: average number of duplicates of found features]
    """

    # sort points into clusters dependent on cosine proximity and return list of averaged and unique points

    multiFoundFeature = []
    singleFoundFeature = []

    featurePointList = [fpl for fpl in featurePointList if type(fpl) == Point]

    for fpl_index, fpl_value in enumerate(featurePointList):
        # find feature points that have sufficiently low relative angular displacement as to be considered
        # as the same point
        featurePointCosines = getVectorCosine(
            featurePointList, fpl_value, localCentroid
        )

        # iterate through featurePointCosine values to find those smaller than minCosineResolution
        cosCompare = [
            fpc_index
            for fpc_index, fpc_value in enumerate(featurePointCosines)
            if fpc_value > minCosResolution
        ]  # minCosClusterResolution

        # if more than 1 instance within minCosineResolution, get value average (meanPoint) of instances
        # and store in separate list, multiFoundFeature
        if len(cosCompare) > 1:
            for cc in cosCompare:
                if cc not in dict(multiFoundFeature).keys():
                    multiFoundFeature.append(
                        (cc, cosCompare[0])
                    )  # append a unique tuple of (Point, ?)
        else:  # single instances to retain for singleFoundFeature
            singleFoundFeature.append(fpl_index)
    # if proportion of points in multiFoundFeatures lower that remainder of singleFoundFeatures repeat search
    # with higher Tammes Sphere value

    featureBins = list(dict(multiFoundFeature).values())

    # proportion of singleFoundFeature points? consider that multi point determination is for alignment,
    # also take stock of proportion of points searches that time out
    # if zero feature points appear, switch to searching for minima without increasing resolution
    # binning points should work for minima and maxima points as relatedness is based on angle difference
    # store mean of multiFoundFeature points in Shape object

    meanMultiFoundFeature = []
    binMultiFoundFeature = {}

    for k, v in dict(multiFoundFeature).items():
        binMultiFoundFeature.setdefault(v, []).append(k)  # featurePointList[v]?

    # change this to the order of most times discovered => most likely to be discovered in
    # other model, moot for symmetric models
    # disable 4 lines below and enable featureDisps.sort() to trial other histogram
    # binFoundTimes: elements of each bin, binFoundOrder: sort by bin size
    binFoundTimes = [len(bmff) for bmff in binMultiFoundFeature.values()]
    binFoundOrder = [
        i[0] for i in sorted(enumerate(binFoundTimes), key=lambda x: x[1], reverse=True)
    ]
    keyList = list(binMultiFoundFeature)
    freqSortedKeys = [keyList[i] for i in binFoundOrder]

    for fsk in freqSortedKeys:  # get a point average
        averagePointCluster = []
        [
            averagePointCluster.append(featurePointList[bmff])
            for bmff in binMultiFoundFeature[fsk]
        ]
        # meanMultiFoundFeature.append(meanPoint(averagePointCluster))
        meanMultiFoundFeature.append(
            meanPoint(dispTrim(averagePointCluster, meanPoint(averagePointCluster)))
        )

    # recalculate the centroid based on feature points that have been identified within a lower tolerance of
    # cosResolution (angular difference)
    featureCluster = meanMultiFoundFeature.copy()
    [featureCluster.append(featurePointList[sff]) for sff in singleFoundFeature]

    if len(featureBins) > 0.0:
        foundRatio = len(featureBins) / len(set(featureBins))
    else:
        foundRatio = 0.0

    return featureCluster, foundRatio


def getVectorCosine(listPoint, singlePoint, originOffset):
    """
    Return a list of scalar cosine values determined from angle between vectors defined by points in listPoint list
    and single point, singlePoint, common to a local shape centroid.

    :param listPoint: point list
    :param singlePoint: single point
    :param originOffset: origin offset
    :return: scalar cosine value list
    """

    if type(singlePoint) is not Point:
        pass
    vectorCosine = []

    v1 = np.array([singlePoint.x, singlePoint.y, singlePoint.z]) - np.array(
        [originOffset.x, originOffset.y, originOffset.z]
    )
    for lp in listPoint:
        v2 = np.array([lp.x, lp.y, lp.z]) - np.array(
            [originOffset.x, originOffset.y, originOffset.z]
        )
        vectorCosine.append(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    return vectorCosine


def pointsToBins2(pointList, localCentroid, minCosResolution, minDispResolution):
    """
    If a surface is subjected to multiple searches and the same features are discovered
    each time, then it is likely that an adequate number of searches have been conducted.
    But if the same number of searches discover different points each time, it is
    less likely that all registration features have been discovered.
    This may be described as a binomial probability distribution, e.g. two searches in a region with a
    single feature will always return the same feature, but two searches in a region with
    ten features each having an equal probability of discovery will only have a 0.1 chance of
    encountering the same feature twice in a row. In the implementation used for testing, a
    ratio of the number of features discovered multiple times against the total number of
    features found is used to evaluate a termination condition (pointsToBins2, pointsToCurves2).
    This approach is sensitive to the values of termination condition used, models may have
    features that are only revealed at a higher number of searches. When models are evaluated
    for registration features using differing parameters such as higher density of seed points
    or a higher confidence threshold, they may present an unmatched number of features that
    preventing simple matching. This issue is ameliorated by the use of preliminary searches
    that use both registration feature type and centroid displacement to detect a potential
    shape src.

    :param pointList: list of unsorted points indicating a feature
    :param localCentroid: CAD surface shape local centroid
    :param minCosResolution: minimum cosine distance between adjacent points distinguishing unique points
    :param minDispResolution: minimum distance between adjacent points distinguishing unique points
    :return: [featureCluster: list of features with duplicates removed
              foundRatio: average number of duplicates of found features]
    """

    """
    sort points into clusters dependent on cosine proximity and return list of averaged and unique points
    displacement from centroid is a second ordering factor
    """
    multiPoint = []
    singlePoint = []

    # strip out NANs etc
    pointList = [pl for pl in pointList if type(pl) == Point]

    for pl_index, pl_value in enumerate(pointList):
        # find feature points that have sufficiently low relative angular displacement as to be
        # considered as the same point
        pointCosines = getVectorCosine(pointList, pl_value, localCentroid)
        pointDisps = localDisp(pointList, pl_value)
        # pointCompare = []# list of lists containing points from same region

        pointCompare = [
            pl_index2
            for pl_index2, pl_value2 in enumerate(pointList)
            if (pointCosines[pl_index2] > minCosResolution)
            and (pointDisps[pl_index2] < minDispResolution * 20)
        ]

        if len(pointCompare) > 1:
            [
                multiPoint.append((cc, pointCompare[0]))
                for cc in pointCompare
                if cc not in dict(multiPoint).keys()
            ]
        else:  # single instances to retain for singleFoundFeature
            singlePoint.append(pl_index)

    pointBins = list(dict(multiPoint).values())

    # proportion of singleFoundFeature points? consider that multi point determination is for alignment,
    # also take stock of proportion of points searches that time out
    # if zero feature points appear, switch to searching for minima without increasing resolution
    # binning points should work for minima and maxima points as relatedness is based on angle difference
    # store mean of multiFoundFeature points in Shape object
    meanMultiPoint = []
    binMultiPoint = {}

    for k, v in dict(multiPoint).items():
        binMultiPoint.setdefault(v, []).append(k)

    # change this to the order of most times discovered => most likely to be discovered in other model,
    # moot for symmetric models
    # disable 4 lines below and enable featureDisps.sort() to trial other histogram
    # binFoundTimes: elements of each bin, binFoundOrder: sort by bin size
    binFoundTimes = [len(bmff) for bmff in binMultiPoint.values()]
    binFoundOrder = [
        i[0] for i in sorted(enumerate(binFoundTimes), key=lambda x: x[1], reverse=True)
    ]
    keyList = list(binMultiPoint)
    freqSortedKeys = [keyList[i] for i in binFoundOrder]

    for fsk in freqSortedKeys:  # get a point average
        averagePointCluster = []
        [averagePointCluster.append(pointList[bmp]) for bmp in binMultiPoint[fsk]]
        meanMultiPoint.append(meanPoint(averagePointCluster))

    # recalculate the centroid based on feature points that have been identified within a lower tolerance
    # of cosResolution (angular difference)
    pointCluster = meanMultiPoint.copy()
    [pointCluster.append(pointList[sff]) for sff in singlePoint]

    if len(pointBins) > 0.0:
        # foundRatio = min(binFoundTimes) # len(pointBins) / len(set(pointBins))
        foundRatio = (len(pointList) - len(singlePoint)) / len(pointList)

    else:
        foundRatio = 0.0

    return pointCluster, foundRatio


def dispsToBins(dispList, minResolution):
    """
    For each scalar value in dispList list, determine whether any other members of dispList are of less than
    minResolution absolute difference. Create separate lists of these values that fall within minResolution difference
    and get a mean value for these clustered groups.
    Return a list composed of the single unique displacement values and the mean displacement values of the
    clustered displacements.

    :param dispList: list of scalar displacement values
    :param minResolution: minimum absolute displacement for two displacements to be considered unique.
    :return: composite list of unique displacement values and median non-unique displacement values.
    """

    if len(dispList) == 1:
        return dispList

    multiInstanceDisp = []
    singleFoundDisp = []
    meanMultiInstanceDisp = []
    binMultiInstanceDisp = {}

    for dl_index, dl_value in enumerate(dispList):
        # iterate through dispList values to find those smaller than minResolution
        dispCompare = [
            dl_index2
            for dl_index2, dl_value2 in enumerate(dispList)
            if abs(dl_value2 - dl_value) < minResolution
        ]
        # if more than 1 instance within minResolution, get value average (meanDisp) of
        # instances and store in separate list, multiInstanceDisp
        if len(dispCompare) > 1:
            for dc in dispCompare:
                if dc not in dict(multiInstanceDisp).keys():
                    multiInstanceDisp.append(
                        (dc, dispCompare[0])
                    )  # append a unique tuple of (Point, ?)
        else:  # single instances to retain for singleFoundFeature
            singleFoundDisp.append(dl_index)

    for k, v in dict(multiInstanceDisp).items():
        binMultiInstanceDisp.setdefault(v, []).append(k)

    binFoundTimes = [len(bmff) for bmff in binMultiInstanceDisp.values()]
    binFoundOrder = [
        i[0] for i in sorted(enumerate(binFoundTimes), key=lambda x: x[1], reverse=True)
    ]
    keyList = list(binMultiInstanceDisp)
    freqSortedKeys = [keyList[i] for i in binFoundOrder]

    for fsk in freqSortedKeys:  # get a point average
        averageDispCluster = []
        [
            averageDispCluster.append(dispList[bmff])
            for bmff in binMultiInstanceDisp[fsk]
        ]
        # meanMultiInstanceDisp.append(meanPoint(averageDispCluster))
        meanMultiInstanceDisp.append(np.mean(scalarTrim(averageDispCluster)))

    featureCluster = meanMultiInstanceDisp.copy()
    [featureCluster.append(dispList[sfd]) for sfd in singleFoundDisp]

    return featureCluster


def dispTrim(pointList, referencePoint, percentile={0.05, 0.95}):
    """
    Remove outliers from a list of points based on percentile displacement of any point from a reference point value,

    :param pointList:
    :param referencePoint:
    :param percentile: quantile, percentage/100 {min, max}
    :return: trimmed point list
    """
    if len(pointList) < 3:
        return pointList
    disps = localDisp(pointList, referencePoint)
    lowerQuantile = np.quantile(disps, min(percentile))
    upperQuantile = np.quantile(disps, max(percentile))
    return [
        pointList[di]
        for di, dv in enumerate(disps)
        if (dv > lowerQuantile) and (dv < upperQuantile)
    ]


def scalarTrim(D, percentile={0.05, 0.95}):
    """
    Remove outliers from a list of scalar values based on percentile displacement of scalar list ,
    (not data 'Winsor' processing, which replaces expurged values)

    :param D: list of scalar values
    :param percentile: outlier boundary limits
    :return: trimmed scalar list
    """
    if len(D) < 3:
        return D
    Dcopy = D.copy()
    lowerQuantile = np.quantile(Dcopy, min(percentile))
    upperQuantile = np.quantile(Dcopy, max(percentile))
    return [
        D[di]
        for di, dv in enumerate(Dcopy)
        if (dv > lowerQuantile) and (dv < upperQuantile)
    ]


def pointsToCurves2(
    featureCurveList, featureCentreList, localCentroid, minDispResolution
):
    """
    As with pointsToBins2(), sort lists of symmetric features discovered into bins corresponding with features
    and centres of similar value. Substitute multiples values within bins with their mean values and create a return
    list of geometrically unique values according to minDispResolution criteria.
    Note that discrimination is based on a crude constant that is sensitive to scaling.

    :param featureCurveList: list of points determined to be on the periphery of rotationally symmetric features
    :param featureCentreList: list of points representing centres estimated for rotationally symmetric features
    :param localCentroid: local centroid point of CAD shape object
    :param minDispResolution: minimum displacement constant between unique features
    :return: [curveDisps: unique displacement of curve radius from curve centre,
              curveCentres: points representing unique feature centres,
              foundRatio: ratio of duplicated features normalised to total identified features]
    """

    multiFoundCentre = []
    multiFoundDisp = []
    singleFoundCentre = []
    singleFoundDisp = []
    centrePoints = []  # list of lists containing centrePoints from same curves
    edgeDisps = []  # list of lists containing disps from same curves

    # sort curves by proximity of centrePoints (featureCurveCentreList)
    # this allows distinction of curves of same ridge disp from centroid

    featureCurveDisps = localDisp(featureCurveList, localCentroid)
    # minDispResolution is scale dependent
    # minDispResolution *= np.mean(featureCurveDisps)
    # minDispResolution *= np.mean(localDisp(featureCentreList, localCentroid))

    # need to check both displacement of points from centroid along with centrePoint proximity
    for fcl_index, fcl_value in enumerate(featureCentreList):
        # find curve centre points that are relatively clustered within minDispResolution
        dispCompare = []
        for fcl_index2, fcl_value2 in enumerate(featureCentreList):
            centreDiff = (
                localDisp(fcl_value2, fcl_value) < minDispResolution * 15
            )  # seems to vary with groove & ridge
            # -----------SCALE DEPENDENT FUDGE FACTOR ALERT
            dispDiff = (
                abs(featureCurveDisps[fcl_index2] - featureCurveDisps[fcl_index])
                < minDispResolution
            )
            if centreDiff and dispDiff:
                dispCompare.append(fcl_index2)

        # dispCompare = [fcl_index2 for fcl_index2, fcl_value2 in enumerate(featureCentreList) if
        #               abs(fcl_value2 - fcl_value) < minDispResolution] # 0.005

        # if more than 1 instance within minDispResolution, get value average (meanPoint) of instances
        # and store in separate list, multiFoundFeature
        if len(dispCompare) > 1:
            for dc in dispCompare:
                if dc not in dict(multiFoundCentre).keys():
                    multiFoundCentre.append(
                        (dc, dispCompare[0])
                    )  # append a unique tuple of (Point, ?)
                    multiFoundDisp.append(
                        (dc, dispCompare[0])
                    )  # identical to multiFoundCentre => multiCentreDisp
        else:  # single instances to retain for singleFoundFeature
            singleFoundCentre.append(fcl_index)  # singleCentreDisp
            singleFoundDisp.append(fcl_index)

        pointList = [featureCentreList[i] for i in dispCompare]
        dispList = [featureCurveDisps[i] for i in dispCompare]

        if not centrePoints:
            centrePoints.append(pointList)
            edgeDisps.append(dispList)
        elif centrePoints[-1] != pointList:
            centrePoints.append(pointList)
            edgeDisps.append(dispList)

    curveBins = list(dict(multiFoundCentre).values())

    meanMultiFoundCentre = []
    binMultiFoundCentre = {}

    meanMultiFoundDisp = []
    # binMultiFoundDisp = {}

    for k, v in dict(multiFoundCentre).items():
        binMultiFoundCentre.setdefault(v, []).append(k)

    # for k, v in dict(multiFoundDisp).items():
    #     binMultiFoundDisp.setdefault(v, []).append(k)

    # change this to the order of most times discovered =>
    # most likely to be discovered in other model, moot for symmetric models
    # disable 4 lines below and enable featureDisps.sort() to trial other histogram
    # binFoundTimes: elements of each bin, binFoundOrder: sort by bin size
    binFoundTimes = [len(bmfc) for bmfc in binMultiFoundCentre.values()]
    binFoundOrder = [
        i[0] for i in sorted(enumerate(binFoundTimes), key=lambda x: x[1], reverse=True)
    ]
    keyList = list(binMultiFoundCentre)
    freqSortedKeys = [keyList[i] for i in binFoundOrder]

    # binFoundTimes = [len(bmfc) for bmfd in binMultiFoundDisp.values()]
    # binFoundOrder = [i[0] for i in sorted(enumerate(binFoundTimes), key=lambda x: x[1], reverse=True)]
    # keyList = list(binMultiFoundDisp)
    # freqSortedKeys = [keyList[i] for i in binFoundOrder]

    for fsk in freqSortedKeys:  # get a radius displacement average
        averageCentres = [featureCentreList[bmfc] for bmfc in binMultiFoundCentre[fsk]]
        averageDisps = [featureCurveDisps[bmfc] for bmfc in binMultiFoundCentre[fsk]]
        meanMultiFoundCentre.append(
            meanPoint(dispTrim(averageCentres, meanPoint(averageCentres)))
        )

        # meanMultiFoundCentre.append(medianPoint(averageCentres))
        meanMultiFoundDisp.append(np.mean(scalarTrim(averageDisps)))
        # meanMultiFoundDisp.append(np.mean(averageDisps))

    # recalculate the centroid based on feature points that have been identified within a lower tolerance
    # of cosResolution (angular difference)

    curveCentres = meanMultiFoundCentre.copy()
    # [curveCentres.append(featureCentreList[sff]) for sff in singleFoundCentre] #-----------------DROP SINGLE FOUND

    curveDisps = meanMultiFoundDisp.copy()
    # [curveDisps.append(featureCurveDisps[sff]) for sff in singleFoundDisp] #---------------------DROP SINGLE FOUND

    if len(curveBins) > 0.0:
        foundRatio = min(binFoundTimes)  # len(curveBins) / len(set(curveBins))
    else:
        foundRatio = 0.0

    return curveDisps, curveCentres, foundRatio


# TODO: make psuedoScale Shape class method
def psuedoScale(ShapeInstance):
    # get a scale feature to scale constants
    xVals = [p.x for p in ShapeInstance.surfacePoints]
    yVals = [p.y for p in ShapeInstance.surfacePoints]
    zVals = [p.z for p in ShapeInstance.surfacePoints]
    xRange = max(xVals) - min(xVals)
    yRange = max(yVals) - min(yVals)
    zRange = max(zVals) - min(zVals)
    # return sqrt(xRange*yRange*zRange)
    return xRange * yRange * zRange


def searchFeatures(
    ShapeInstance,
    seedPoints,
    stepAngle,
    minCosResolution,
    resolution=50,
    stopCondition=0.005,
    maxIter=25,
    searchFactor=0.9,
    scene=None,
):  # stopCondition=0.005,  # _____________________________fudge factor
    """
    Feature search operation,
    Repeat search for features until adequate features are identified. Given a centroid point,
    subdivide surface geometry into search regions defined by evenly distributed Tammes Sphere points
        Start search from random intersection from centroid to surface
        Identify features with extrema search getMaxMin()
        Test points for rotational symmetry rotSym6()
        Categorise identified features
        Using returned information on the number of times each feature is found, determine search termination.

    The "resolution" parameter is relatively sensitive to shape complexity, there is a trade-off between exhaustive
    search times and minimum levels of feature point recognition. The value is fixed for a proof-of-concept script
    over a finite test library, but setting optimal values remain an open problem.
    Some improvement is gained by recording the number of times the same feature point is discovered.

    :param ShapeInstance: CAD shape object
    :param seedPoints: initial surface points from where to initiate feature search,
    :param stepAngle: angle between centrePoint and neighbouring search points in rosette measured through localCentroid
    :param minCosResolution: minimum cosine distance between adjacent points distinguishing unique points,
    :param resolution: density of initial projected rays from local centroid
    :param stopCondition: minimum value for stepAngle, function precision
    :param maxIter: minimum number of discovered features before terminating feature search (disused)
    :param searchFactor: search termination constant based on proportion of unique features to total features found
    :param scene: coin3d pivy graphical display instance, point display for debug purposes
    :return: updates features in Shape object
    """

    # commence Feature Maxima with evenly distributed seed points from estimated centroid

    maxResolution = 50
    # from numpy import cos
    # minCosResolution = cos(8 * stopCondition)
    # minCosClusterResolution =  cos(32 * stopCondition)

    featureType = "localMax"
    # featureType = "localMin"
    repeatFind = 0
    lastRepeatFind = -1
    featurePointList = []
    featureCurveList = []
    featureSphereList = []
    clusteredFeatureTally = 0
    incidentPoints = ShapeInstance.surfacePoints
    localCentroid = ShapeInstance.centroid
    minDispResolution = 0.001
    inadequateFeatures = True
    inadequateResolution = True

    # if scene:
    #     displayPointFS(DS, 65280, scene)  # green

    while inadequateResolution or inadequateFeatures:
        if not seedPoints:  # not case of centroid updates
            # get distributed points
            seedPoints = randomDistIntersects(ShapeInstance, resolution, featureType)

            # if scene:
            #     scene.addPoints(seedPoints, 'grey')

            # rotate points by phi on each iteration?

        featureCluster = []
        curveDisps = []
        circleCentres = []
        unassignedCount = 0
        curveCentres = None

        for sp in seedPoints:
            featureObject, featureIdentity, circleCentre = getMaxMin(
                ShapeInstance,
                sp,
                localCentroid,
                stepAngle,
                stopCondition,
                35,
                featureType,
                scene=None,
            )

            if featureType == "localMax":
                printVerbose("Local maxima feature identified: " + featureIdentity)
            elif featureType == "localMin":
                printVerbose("Local minima feature identified: " + featureIdentity)

            if circleCentre and (circleCentre != "unassigned"):
                circleCentres.append(circleCentre)
            if featureObject:
                if (featureIdentity == "maxPoint") or (featureIdentity == "minPoint"):
                    featurePointList.append(featureObject)

                    if scene:
                        if featureIdentity == "maxPoint":
                            displayPointFC(featureObject, 65280, scene)  # green
                        else:
                            displayPointFC(featureObject, 1671168, scene)  # blue

                elif featureIdentity == "rotSymRidge" or (
                    featureIdentity == "rotSymGroove"
                ):
                    featureCurveList.append(featureObject)
                    # do nothing with this until comparison?
                    # would appear only to be of use wrt displacements

                    if scene:
                        displayPointFC(featureObject, "red", scene, "cross7")

                elif featureIdentity == "sphere":
                    featureSphereList.append(featureObject)
                    # ShapeInstance.spherePoints.append(featureObject)

                elif featureIdentity == "noCluster":
                    printVerbose("no point cluster returned")

                # refine location of local centroid, sensitive to broad point distribution
                localCentroid = medianPoint(ShapeInstance.surfacePoints)

                # shifting local centroid can cause rotSymTest() convergence failure in cylinders
                # print(medianPoint(ShapeInstance.surfacePoints))
                # if featurePointList:
                #     # case of cylinder tugging centroid around
                #     localCentroid = medianPoint(ShapeInstance.surfacePoints)
                #     #dummy = True
                # else:
                #     # unsure of instances with mix ridges + points
                #     # subsequently dropped for mean(circleCentre + featurePoint)
                #     localCentroid = meanPoint(ShapeInstance.surfacePoints)
                # # if medianPoint(ShapeInstance.surfacePoints) != meanPoint(ShapeInstance.surfacePoints)

            else:
                if (featureIdentity == "unassigned") or (featureIdentity == "ridge"):
                    # printVerbose("unassigned point")
                    unassignedCount += 1

        # feature rotationally symmetric curves differentiated between ridge and groove here.
        ShapeInstance.centrePoints.extend(circleCentres[:])
        if featureType == "localMax":
            ShapeInstance.rotSymRidgePoints.extend(featureCurveList[:])
        elif featureType == "localMin":
            # this should only occur if there are less than 3 localMax & curve features
            ShapeInstance.rotSymGroovePoints.extend(featureCurveList[:])

        if featurePointList:  # and (featureType == "localMax")
            pointDispResolution = (minDispResolution * 20) + (
                1.2e-7 * psuedoScale(ShapeInstance)
            )

            featureCluster, pointFoundRatio = pointsToBins2(
                featurePointList,
                localCentroid,
                cos(minCosResolution * 20),
                pointDispResolution,
            )
            printVerbose("\nPoint Found Ratio: " + str(pointFoundRatio))
            if featureCluster:
                # featureDisps = localDisp(featureCluster, localCentroid)

                if scene:
                    displayPointFC(featureCluster, "lime", scene, "cross7")

                # repeatFind = len(featureBins)/len(set(featureBins))
                repeatFind = pointFoundRatio  # should loop stop once  >3 feature points have been found multiple times?
                if (
                    np.abs(lastRepeatFind - repeatFind) < 0.01
                ):  # not having any effect, bail out, more fudge factor
                    inadequateResolution = False
                else:
                    lastRepeatFind = repeatFind
                    if (resolution < maxResolution) and (repeatFind < searchFactor):
                        resolution += 10
                        seedPoints = []  # ensure seedpoints are regenerated
                        stepAngle /= 2
                if (
                    (repeatFind > searchFactor)
                    and (unassignedCount < len(featurePointList) / 5)
                ) or (resolution >= maxResolution):
                    inadequateResolution = False
                else:
                    inadequateResolution = True

            # if len(circleCluster) == len(featureCluster): localCentroid = meanPoint(circleCluster + featureCluster)
            localCentroid = medianPoint(featureCluster)

            # pointCentroid = meanPoint(ShapeInstance.surfacePoints)
            # print('mean surface point - mean feature point centroid: ' + str(localDisp(localCentroid, pointCentroid)))

        if featureCurveList:
            if featureType == "localMax":
                # curveTol = 0.15 #-------0.025-------------------------------------------FUDGE FACTOR ALERT
                curveTol = 0.08 + 6e-6 * psuedoScale(ShapeInstance)
                # small surface geometry rotations lead to looser grouping
            else:
                # minimum curve features are that bit more flakey... can the whole concept be discarded?------------?
                # curveTol = 0.2
                curveTol = 2e-2 * psuedoScale(ShapeInstance)

            curveDisps, curveCentres, curveFoundRatio = pointsToCurves2(
                featureCurveList, circleCentres, localCentroid, curveTol
            )

            if len(curveCentres) > 0:
                # repeatFind = len(featureBins)/len(set(featureBins))
                repeatFind = curveFoundRatio
                # or should this loop stop once there are >3 feature points that have been found multiple times?
                if lastRepeatFind == repeatFind:  # not having any effect, bail out
                    inadequateResolution = False
                else:
                    lastRepeatFind = repeatFind
                    if (resolution < maxResolution) and (repeatFind < searchFactor):
                        resolution += 10
                        seedPoints = []  # ensure seedpoints are regenerated
                if (
                    (repeatFind > searchFactor)
                    and (unassignedCount < len(featureCurveList) / 5)
                ) or (
                    resolution >= maxResolution
                ):  # ----------------------------------FUDGE FACTOR ALERT
                    inadequateResolution = False
                else:
                    inadequateResolution = True

        # feature curves are sorted into their respective lengths - but
        # this will not distinguish between curves opposite centroid, e.g. cylinder

        if featureSphereList:
            sphereDisps = localDisp(featureSphereList, localCentroid)
            sphereDisps = dispsToBins(
                sphereDisps, 0.01
            )  # ------------------------------SCALE DEPENDENT FUDGE FACTOR
            if sphereDisps:
                ShapeInstance.featureSphereDisps = sphereDisps
                # sphereRadius = localDisp(ShapeInstance.spherePoints, localCentroid)
                # if sum([mean(sphereRadius) - sr for sr in sphereRadius]) > minDispResolution:
                #     ShapeInstance.spherePoint = []
                #     print("not a sphere")

                inadequateFeatures = (
                    False  # single sphere feature is a minimum identifiable feature
                )
                # There is a possibility of an object comprising several sphere sectors (not in test data)
                # However as a spherical surface feature cannot be sorted in bins, and ratios of previously
                # found feature points used to limit search cycles, limit maxResolution.
                maxResolution = 30
                resolution += 10

                # Assumed that all spherical features are determined within a fixed number of

        if featureCluster and not inadequateResolution:
            # generate sorted histogram signature from multiFoundFeature |mean points localCentroid| displacement
            # featureDisps.sort() # should the histogram be normalised? scale invariant?
            # in cases of spindles, aggregate single found features with similar length,
            # or create category of single found features where search ends with 3x in a row on search rosette
            # ShapeInstance.pointHistogram = featureDisps

            clusteredFeatureTally += len(featureCluster)

            if featureType == "localMax":
                ShapeInstance.featureMaxPoints = featureCluster
            elif featureType == "localMin":
                # this should only occur if there are less than 3 localMax & curve features
                ShapeInstance.featureMinPoints = featureCluster

        if curveDisps and not inadequateResolution:
            ShapeInstance.centrePoints = circleCentres
            clusteredFeatureTally += len(curveCentres)
            if featureType == "localMax":
                ShapeInstance.featureMaxCentres = curveCentres
                ShapeInstance.featureMaxCurveDisps = curveDisps
                ShapeInstance.rotSymRidgePoints = featureCurveList  # points retained for further centre identification?
            elif featureType == "localMin":
                # this should only occur if there are less than 3 localMax & curve features
                ShapeInstance.featureMinCentres = curveCentres
                ShapeInstance.featureMinCurveDisps = curveDisps
                ShapeInstance.rotSymGroovePoints = featureCurveList

        if featureType == "localMax":
            if (clusteredFeatureTally < 5) and (resolution <= maxResolution):
                # (clusteredFeatureTally < 5) is more than requirement for a 3-point SVD src,
                # but the reliability of minima features is less than that of maxima,
                # this should be taken into account during histogram comparison, e.g. differing weights
                # for maxima and minima features.

                # problem with "if (clusteredFeatureTally < 5) and (resolution <= maxResolution):" is that
                # objects with few features having a low chance of discovery (e.g. long cylinder minima end faces)
                # can return incompatible histograms
                # Seems feature discovery has to be exhaustive
                seedPoints = []
                if not inadequateResolution:
                    maxFeatures = []
                    if featureCluster:
                        maxFeatures = maxFeatures + featureCluster
                    if curveCentres:
                        maxFeatures = maxFeatures + curveCentres

                    featureCentroidCorr = medianCentroidCorrection(
                        ShapeInstance, tol=0.002, maximalPoints=maxFeatures
                    )
                    localCentroid = ShapeInstance.centroid
                    featureType = "localMin"  # ------------------------------------------------CLUSTERS
                    featureCurveList = []
                    featurePointList = []
                else:
                    resolution += 10
                    stepAngle /= 2
            elif featureIdentity == "sphere":
                if resolution > maxResolution:
                    inadequateResolution = False
            else:
                inadequateFeatures = False

        elif featureType == "localMin":
            inadequateFeatures = False  # stop looking for sufficient feature points
            if inadequateResolution:
                resolution += 5

    # localCentroid = meanPoint(circleCluster)
    ShapeInstance.centroid = localCentroid
    if clusteredFeatureTally < 3:
        print("inadequate definition warning")


def randomDistIntersects(ShapeInstance, resolution, featureType, rotation=0.0):
    """
    Intersect a given shape object with projections from shape centroid according to the distribution of
    points on a unit Deserno sphere of a given resolution.

    :param ShapeInstance: CAD shape object
    :param resolution: Deserno sphere point density
    :param featureType: intersection feature type, (always convex hull, "localMax")
    :param rotation: Deserno sphere rotation around Z-axis
    :return: intersected points
    """

    # DA = DesernoAngle(resolution)
    # DA = TammesAngle(resolution)

    DS = DesernoSphere(resolution, rotation, interleavedFlag=True)
    # DS = TammesSphere(resolution)
    initialPoints = []
    DS = offsetPoints(DS, ShapeInstance.centroid)

    for ds in DS:
        sp = proj2object2(ShapeInstance, ds, ShapeInstance.centroid, featureType)

        if sp:
            initialPoints.append(sp)

    if not initialPoints:
        print("no intersections found")

    return initialPoints


# find best src between a group of histograms or determine probability of src between 2 histograms
# histograms are sorted by |centroid  featureExtrema| displacement then normalised for scale
# get least mean squared difference between values

# https://github.com/scipy/scipy/blob/v0.19.0/scipy/spatial/distance.py#L408-L434
# scipy distance routines involve time-consuming array checking

# use minimum intersection distance because of relatively exact signature
# only normalise during comparison to retain scaling transform
# pad unequal length number signatures with zeros
# i.e. for each value in one signature, find the closest value in the other

# ranking on distance(Source.histogram list(Target[0].histogram.. Target[n].histogram) , method='dunno')
# from individual calls to opencvDistance & scipyDistance methods

# distanceRank(S.histogram, [t.histogram for t in T]) returns order ranked index of T.. target objects


def distanceRank(source, targetList, methodName="Intersection"):
    """
    Representing binned shape feature data as a one-dimensional histogram allows direct comparison of shape data
    independent of affine transformations between geometric shapes.
    DistanceRank() encapsulates several methods used to evaluate similarity or distance between histograms from
    OpenCV or SciPy libraries. Given a source array, return the normalised distance from the target array.
    Unequal arrays are padded with zeros to the size of the larger array
    Returned factor normalised to [0, 1] st 1 => source = target

    :param source: source histogram as list
    :param targetList: single or multiple target histograms lists within list
    :param methodName: method name
    :return: list of normalised distance rankings
    """

    rankList = []

    # def normList(L):
    #    """ normalise list floats to range [0, 1]"""
    #    s = sum(L)
    #    return [float(i) / s for i in L]

    def padList(L1, L2):
        """
        Pad shorter list with zeros to src longer list.

        :param L1: list
        :param L2: different list
        :return: L1, L2
        """

        if len(L1) > len(L2):
            [L2.append(0.0) for r in range(len(L2), len(L1))]
        elif len(L2) > len(L1):
            [L1.append(0.0) for r in range(len(L1), len(L2))]

        return L1, L2

    # import cv2

    OPENCV_METHODS = {
        "Correlation": cv2.HISTCMP_CORREL,
        "Chi-Squared": cv2.HISTCMP_CHISQR,
        "Intersection": cv2.HISTCMP_INTERSECT,
        "Hellinger": cv2.HISTCMP_BHATTACHARYYA,
    }

    # scipy distance metric methods
    # from scipy.spatial import distance as scipyDist
    SCIPY_METHODS = {
        "Euclidean": scipyDist.euclidean,
        "Manhattan": scipyDist.cityblock,
        "Chebysev": scipyDist.chebyshev,
    }

    source = normFloatList(source)
    if methodName in OPENCV_METHODS.keys():
        dNorm = cv2.compareHist(source, source, OPENCV_METHODS[methodName])

        for t in targetList:
            # compute distance between source & target histograms
            t = normFloatList(t)
            d = cv2.compareHist(padList(source, t), OPENCV_METHODS[methodName])
            rankList.append(d / dNorm)

    elif methodName in SCIPY_METHODS.keys():
        dNorm = SCIPY_METHODS[methodName](source, source)

        for t in targetList:
            # compute distance between source & target histograms
            t = normFloatList(t)
            d = SCIPY_METHODS[methodName](padList(source, t))
            rankList.append(d / dNorm)

    else:
        raise RuntimeError("distanceRank() unknown methodName")

    # correlation or intersection method => sort the results in reverse order
    # sort results by decreasing src
    rankList = sorted(
        rankList, reverse=methodName not in ("Correlation", "Intersection")
    )
    return rankList


# TODO: make medianCentroidCorrection Shape class method
def medianCentroidCorrection(shapeInstance, tol=0.001, maximalPoints=None):
    """
    Test for a difference between a mean and median centroid, indicating missing feature points,
    use median histogram as next best guess and update feature-point centroid displacements.

    :param maximalPoints:
    :param shapeInstance: CAD shape object
    :param tol: minimum value for consideration of separate median and mean centroid
    :return: boolean, also updates Shape instance values
    """

    centroidChanged = None

    if not maximalPoints:
        # featureMaxPoints, featureMaxCentres less sensitive to centroid misalignment
        maximalPoints = shapeInstance.featureMaxPoints + shapeInstance.featureMaxCentres
    if not maximalPoints:
        return centroidChanged

    medianCentroid = medianPoint(maximalPoints)
    meanCentroid = meanPoint(maximalPoints)
    scaledTol = tol * localDisp(max(maximalPoints), shapeInstance.centroid)
    centroidChanged = localDisp(meanCentroid, medianCentroid)
    if centroidChanged > scaledTol:
        shapeInstance.centroid = medianCentroid
        newDisps = localDisp(shapeInstance.featureMaxPoints, medianCentroid)
        revisedFeatureMaxPointOrder = sorted(
            enumerate(newDisps), key=lambda x: x[1], reverse=True
        )
        revisedFeatureMaxPoints = [
            shapeInstance.featureMaxPoints[i[0]] for i in revisedFeatureMaxPointOrder
        ]
        shapeInstance.featureMaxPoints = revisedFeatureMaxPoints
        newDisps = localDisp(shapeInstance.featureMaxCentres, medianCentroid)
        revisedFeatureMaxCentreOrder = sorted(
            enumerate(newDisps), key=lambda x: x[1], reverse=True
        )
        revisedFeatureMaxCentres = [
            shapeInstance.featureMaxCentres[i[0]] for i in revisedFeatureMaxCentreOrder
        ]
        shapeInstance.featureMaxCentres = revisedFeatureMaxCentres
        if centroidChanged:
            printVerbose(
                "median point intersection centroid - median feature centroid: "
                + str(centroidChanged)
            )
    return centroidChanged


def createDispSignature(shapeInstance):
    """
    This shape signature represents a geometric CAD surface as a minimal affine-invariant representation of features
    and their displacement from the shape centroid. To create a data representation suited to easy comparison against,
    features are ordered within respective category types (minima, maxima, rotationally-symmetric ridges,
    rotationally-symmetric grooves, spherical regions), then represented according to their respective
    displacement from the local shape centroid within these feature categories. Returned in dispSignature.
    A separate index orders respective features by displacement length within each feature category,
    (dispSignatureIndexMap).
    The number of distinct features are also represented as a list of tuples, creating a representation suited to
    rapid comparison (typeRangeMaxMin).

    :param shapeInstance:
    :return: [dispSignature: feature categories and disps,
              dispSignatureIndexMap: index map to feature disps, complementary index array
              typeRangeMaxMin: minimalist representation of number of distinct feature]
    """

    # artificial sorting of histogram by disp or ???discovered frequency???
    # both heuristics are sensitive to centroid accuracy
    # which is in turn sensitive to finding all features

    # weed out empty failed shape objects
    if (shapeInstance.surfaceStatus == "no surface returned") or (
        shapeInstance.surfacePoints == []
    ):
        print("Empty shape object passed to createDispSignature")
        return [[], [], [(), (), (), (), ()]]

    featureMaxDisps = localDisp(shapeInstance.featureMaxPoints, shapeInstance.centroid)
    featureMinDisps = localDisp(shapeInstance.featureMinPoints, shapeInstance.centroid)
    rotSymRidgeDisps = shapeInstance.featureMaxCurveDisps
    rotSymGrooveDisps = shapeInstance.featureMinCurveDisps
    sphereDisps = localDisp(shapeInstance.spherePoints, shapeInstance.centroid)

    FMaxPLen = len(featureMaxDisps)
    RSRidgePLen = len(rotSymRidgeDisps)
    FMinPLen = len(featureMinDisps)
    RSGroovePLen = len(rotSymGrooveDisps)
    SPLen = len(sphereDisps)

    dispSignature = (
        featureMaxDisps
        + rotSymRidgeDisps
        + featureMinDisps
        + rotSymGrooveDisps
        + sphereDisps
    )

    # typeRangeMaxMin is a compact format representing the number of different feature types ordered by
    # feature reliability

    # first integer in tuple is sum of (mixed) prior features, second is sum of prior features plus specific feature

    # [(0, numberOfPointMaxima), (numberOfPointMaxima, numberOfCurveMaxima)...(totalPriorFeatureIndex,  ]

    # form pairs of min, max values as ranges are contiguous
    typeRangeMaxMin = []
    if FMaxPLen > 0:
        typeRangeMaxMin.append(
            (0, FMaxPLen - 1),
        )
    else:
        typeRangeMaxMin.append(
            (),
        )

    if RSRidgePLen > 0:
        typeRangeMaxMin.append(
            (FMaxPLen, FMaxPLen + RSRidgePLen - 1),
        )
    else:
        typeRangeMaxMin.append(
            (),
        )

    if FMinPLen > 0:
        typeRangeMaxMin.append(
            (FMaxPLen + RSRidgePLen, FMaxPLen + RSRidgePLen + FMinPLen - 1),
        )
    else:
        typeRangeMaxMin.append(
            (),
        )

    if RSGroovePLen > 0:
        typeRangeMaxMin.append(
            (
                FMaxPLen + RSRidgePLen + FMinPLen,
                FMaxPLen + RSRidgePLen + FMinPLen + RSGroovePLen - 1,
            ),
        )
    else:
        typeRangeMaxMin.append(
            (),
        )

    if SPLen > 0:
        typeRangeMaxMin.append(
            (
                FMaxPLen + RSRidgePLen + FMinPLen + RSGroovePLen,
                FMaxPLen + RSRidgePLen + FMinPLen + RSGroovePLen + SPLen - 1,
            ),
        )
    else:
        typeRangeMaxMin.append(
            (),
        )

    dispSignatureIndexed = sorted(
        enumerate(dispSignature), key=lambda x: x[1], reverse=True
    )
    dispSignatureIndexMap = [i[0] for i in dispSignatureIndexed]
    dispSignature = [i[1] for i in dispSignatureIndexed]

    return [dispSignature, dispSignatureIndexMap, typeRangeMaxMin]


def normFloatList(L):
    """
    Normalise list floats to range [0, 1]

    :param L: list of floating point numbers
    :return: list of normalised floating point numbers
    """
    """ normalise list floats to range [0, 1]"""
    s = sum(L)
    if s > 0.0:
        return [float(i) / s for i in L]
    else:
        if all([i == 0.0 for i in L]):
            return L
        else:
            raise RuntimeError("normFloatList() div0")


def distanceMeasure(source, target, methodName="Correlation"):
    """
    Representing binned shape feature data as a one-dimensional histogram allows direct comparison of shape data
    independent of affine transformations between geometric shapes.
    distanceMeasure() encapsulates several methods used to evaluate similarity or distance between histograms from
    SciPy library. Given a source array, return the normalised distance from the target array.
    Unequal arrays are padded with zeros to the size of the larger array
    Returned factor normalised to [0, 1] st 1 => source = target

    This version mixes point and curve histograms, sorts by distance
    Then removes matches that do not have corresponding curve/point information

    :param source: source histogram as list
    :param target: single target histograms list
    :param methodName: method name
    :return: distance scalar
    """

    def padList(L1, L2):
        """
        Pad shorter list with zeros to src longer list
        Add values to map index
        """
        # values extending beyond initial indices will not be within mapped values => safe
        if len(L1) > len(L2):
            [L2.append(0.0) for r in range(len(L2), len(L1))]
            [L2.append(r) for r in range(len(L2), len(L1))]
        elif len(L2) > len(L1):
            [L1.append(0.0) for r in range(len(L1), len(L2))]
            [L1.append(r) for r in range(len(L1), len(L2))]

        return L1, L2

    # import cv2
    # OPENCV_METHODS = {
    #     "Correlation": cv2.HISTCMP_CORREL,
    #     "Chi-Squared": cv2.HISTCMP_CHISQR,
    #     "Intersection": cv2.HISTCMP_INTERSECT,
    #     "Hellinger": cv2.HISTCMP_BHATTACHARYYA}

    # scipy distance metric methods
    # from scipy.spatial import distance as scipyDist

    # dist.minkowski(u, v, p)
    #   {||u-v||}_p = (\\sum{|u_i - v_i|^p})^{1/p}.
    # p : int
    #    The order of the norm of the difference :math:`{||u-v||}_p`.

    SCIPY_METHODS = {
        "Euclidean": scipyDist.euclidean,
        "Manhattan": scipyDist.cityblock,
        "Chebyshev": scipyDist.chebyshev,
        "Braycurtis": scipyDist.braycurtis,
        "Correlation": scipyDist.correlation,
        "Sqeuclidean": scipyDist.sqeuclidean,
        "Cosine": scipyDist.cosine,
        "Cityblock": scipyDist.cityblock,
        "Canberra": scipyDist.canberra,
    }

    # source[0] = normFloatList(source[0])
    # if methodName in OPENCV_METHODS.keys():
    #     # compute distance between source & target histograms
    #     target[0] = normFloatList(target[0])
    #     normS, normT = padList(source, target)
    #     dist = cv2.compareHist(normS[0], normT[0], OPENCV_METHODS[methodName])

    if methodName in SCIPY_METHODS.keys():
        # compute distance between source & target histograms
        normS = normFloatList(source[0])
        normT = normFloatList(target[0])
        normS, normT = padList(normS, normT)
        if np.sum(np.array(normS) - np.array(normT)) < eps:
            return 0.0
        dist = SCIPY_METHODS[methodName](normS, normT)

    else:
        raise RuntimeError("distanceRank() unknown methodName")

    return dist


def distanceRank2(source, targetList, methodName="Correlation"):
    """
    Determine dissimilarity of shapes represented as histograms of features and their displacement from the shape
    local centroid. Source and Target lists of geometric shape features are compared in a 2 stage process.
    Features are represented as a histogram, and a series of tuples representing number of features.
    listTypeOrderMatch() weeds out shapes missing comparable features.
    distanceMeasure() then returns the distance derived from comparing histograms.

    :param source: source histogram as list
    :param targetList: single or multiple target histograms list
    :param methodName: method name
    :return: single distance scalar or list of target objects in decreasing order of similarity

    """
    """
    Given a source array, return the normalised distance from the target array.
    Unequal arrays are padded with zeros to the size of the larger array
    Returned factor normalised to [0, 1] st 1 => source = target

    This version mixes point and curve histograms, sorts by distance
    Then removes matches that do not have corresponding curve/point information
    """
    S = createDispSignature(source)
    distNorm = distanceMeasure(S, S, methodName)
    if methodName == "Correlation":
        distNorm = 1.0 - distNorm

    if type(targetList) != list:
        # case of single shape
        T = createDispSignature(targetList)
        # now check if the types under comparison have the same type ordering
        LTOM = listTypeOrderMatch(S, T)
        if LTOM == 1.0:
            dist = distanceMeasure(S, T, methodName)
            if methodName == "Correlation":
                dist = 1 - dist / distNorm
            return dist
        elif 1.0 > LTOM >= 0.5:
            print(" low disp src: minimum curve type?")
            # print("lower prob src: give it a whirl...")
            # dist = distanceMeasure(S, T, methodName)
            # if methodName=='Correlation':
            #     dist = 1 - dist/distNorm
            return LTOM
        else:
            return 0.0
    else:
        # correlation or intersection method => sort the results in reverse order
        # sort results by decreasing src
        rankList = []
        for tl in targetList:
            T = createDispSignature(tl)
            LTOM = listTypeOrderMatch(S, T)
            if LTOM == 1.0:
                dist = distanceMeasure(S, T, methodName)
                if methodName == "Correlation":
                    dist = 1 - dist / distNorm
                rankList.append(dist)
            elif 1.0 > LTOM >= 0.6:
                input("distanceRank weirdness?")
            else:
                rankList.append(0.0)  # check similarity = 1.0 or 0.0
        rankList = sorted(
            rankList, reverse=methodName not in ("Correlation", "Intersection")
        )
        return rankList


def listTypeOrderMatch(A, B):
    """
    Compare sets of features according to the typeRangeMaxMin() format,
    Return a score proportional to the number of categories with matching number of features.

    :param A: source feature list
    :param B: target feature list
    :return: src score scalar
    """

    # is L1/L2 needed? should it be an object?
    # if Ind1 exists as a max value, should it exist as a max value in Ind2?

    # for each "paired" value, starting from 0
    # if both share max/min type then contribute to max/min type bin
    # if both share point/curve type then contribute to point/curve type bin
    # max/min type bin summed over larger bin sum, multiplied by harsh factor
    # point/curve type bin summed over larger bin sum, multiplied by lenient factor reflecting less
    # discriminating point/curve identification

    PmaxA = 0.0
    CmaxA = 0.0
    PminA = 0.0
    CminA = 0.0
    SA = 0.0
    PmaxB = 0.0
    CmaxB = 0.0
    PminB = 0.0
    CminB = 0.0
    SB = 0.0

    if len(A[2][0]) > 0:
        PmaxA = A[2][0][1] - A[2][0][0] + 1  # extract source point maxima

    if len(A[2][1]) > 0:
        CmaxA = A[2][1][1] - A[2][1][0] + 1  # extract source curve maxima

    if len(A[2][2]) > 0:
        PminA = A[2][2][1] - A[2][2][0] + 1  # extract source point minima

    if len(A[2][3]) > 0:
        CminA = A[2][3][1] - A[2][3][0] + 1  # extract source curve minima

    if len(A[2][4]) > 0:
        SA = A[2][4][1] - A[2][4][0] + 1  # extract source spherical regions

    if len(B[2][0]) > 0:
        PmaxB = B[2][0][1] - B[2][0][0] + 1  # extract target point maxima

    if len(B[2][1]) > 0:
        CmaxB = B[2][1][1] - B[2][1][0] + 1  # extract target point minima

    if len(B[2][2]) > 0:
        PminB = B[2][2][1] - B[2][2][0] + 1  # extract target curve maxima

    if len(B[2][3]) > 0:
        CminB = B[2][3][1] - B[2][3][0] + 1  # extract target curve minima

    if len(B[2][4]) > 0:
        SB = B[2][4][1] - B[2][4][0] + 1  # extract target spherical regions

    # eliminate empty mins/ full maxs or empty maxs / full mins

    if ((PmaxA == 0) and (CmaxA == 0) and (SA == 0)) and (
        (PminB == 0) and (CminB == 0) and (SB == 0)
    ):
        return 0.0

    if ((PmaxB == 0) and (CmaxB == 0) and (SB == 0)) and (
        (PminB == 0) and (CminB == 0) and (SB == 0)
    ):
        return 0.0

    # top score: exact src over all categories
    if (
        (PmaxA == PmaxB)
        and (CmaxA == CmaxB)
        and (PminA == PminB)
        and (CminA == CminB)
        and (SA == SB)
    ):
        return 1.0

    # no mutual categories
    if (
        (min(PmaxA, PmaxB) == 0)
        and (min(CmaxA, CmaxB) == 0)
        and (min(PminA, PminB) == 0)
        and (min(CminA, CminB) == 0)
        and (min(SA, SB) == 0)
    ):
        return 0.0

    # if there is no exact src, then summation of percentages of max(curve, point)/total with
    # min(curve, point)/total .. sphere

    # assign weights according to credibility of feature methods
    # PointMaxWeight = 2/7
    # CurveMaxWeight = 2/7
    # PointMinWeight = 1/7
    # CurveMinWeight = 1/7
    # SphereWeight = 1/7

    individualDecision = 0.0
    decisionSum = 0

    if (PmaxA > 0) or (PmaxB > 0):
        individualDecision += min(PmaxA, PmaxB)
        decisionSum += max(PmaxA, PmaxB)

    if (CmaxA > 0) or (CmaxB > 0):
        individualDecision += min(CmaxA, CmaxB)
        decisionSum += max(CmaxA, CmaxB)

    if (PminA > 0) or (PminB > 0):
        individualDecision += min(PminA, PminB)
        decisionSum += max(PminA, PminB)

    if (CminA > 0) or (CminB > 0):
        individualDecision += min(CminA, CminB)
        decisionSum += max(CminA, CminB)

    if (SA > 0) and (SB > 0):
        individualDecision += min(SA, SB)
        decisionSum += max(SA, SB)

    individualDecision /= decisionSum
    decisionSum = 0

    combinesDecision = 0.0

    if max((PmaxA + CmaxA), (PmaxB + CmaxB)) > 0.0:
        combinesDecision += min((PmaxA + CmaxA), (PmaxB + CmaxB))
        decisionSum += max((PmaxA + CmaxA), (PmaxB + CmaxB))

    if max((PminA + CminA), (PminB + CminB)) > 0.0:
        combinesDecision += min((PminA + CminA), (PminB + CminB))
        decisionSum += max((PminA + CminA), (PminB + CminB))

    combinesDecision /= decisionSum

    # if (SA > 0) and (SB > 0):
    #     combinesDecision += min(SA, SB) / max(SA, SB)

    if individualDecision > combinesDecision:
        return individualDecision
    else:
        return combinesDecision


def listIntersection2(Ashape, Bshape, tol=1e-2):
    """
    Given two histograms, Ashape and Bshape, determine float values common to both histograms (within specified
    tolerance) and return equivalent histograms where unmatched histogram values are set to zero.
    Return the indices of the floats common to both input lists

    :param Ashape: CAD shape object
    :param Bshape: CAD shape object
    :param tol: tolerance for floating point subtraction
    :return: [Areturn: Ashape histogram less unmatched floats,
              Breturn: Bshape histogram less unmatched floats,
              scaling difference between histograms, scale is A/B
              maxDev: maximum difference between any normalised scalar pair]
    """
    # get source & target histogram intersection

    # import numpy as np

    # # recall createDispSignature() orders hostogram by disp, so mapping required
    # [A, orderA, _] = createDispSignature(Ashape)
    # [B, orderB, _] = createDispSignature(Bshape)

    DS = []
    for s in [Ashape, Bshape]:
        maxDisps = localDisp(s.featureMaxPoints, s.centroid)
        maxCentre = localDisp(s.featureMaxCentres, s.centroid)
        minDisps = localDisp(s.featureMinPoints, s.centroid)
        minCentre = localDisp(s.featureMinCentres, s.centroid)
        sphereDisps = s.featureSphereDisps
        DS.append(maxDisps + maxCentre + minDisps + minCentre + sphereDisps)

    A = np.array(DS[0])
    B = np.array(DS[1])

    if A.shape[0] >= B.shape[0]:
        HL = A
        HS = B
        subsetCombIndex = [
            i for i in combinations([j for j in range(0, A.shape[0])], B.shape[0])
        ]

    elif B.shape[0] > A.shape[0]:
        HL = B
        HS = A
        subsetCombIndex = [
            i for i in combinations([j for j in range(0, B.shape[0])], A.shape[0])
        ]
        # histogram ordered by disp, so permutations not required

    # elif A.shape[0] == B.shape[0]:
    #     HL = A
    #     HS = B
    #     subsetCombIndex = tuple(i for i in range(0, len(A)))
    #     subsetCombIndex = [subsetCombIndex,]

    # from itertools import combinations
    # subsetCombIndex = [i for i in combinations([j for j in range(0, maxIndex)], minIndex)]

    normDiff = []
    subsetIndex = []
    scaleList = []

    for sci in subsetCombIndex:
        subsetHL = HL[list(sci)]
        scaleDiff = np.abs(subsetHL / HS)
        # trim outliers that fall outside 2x standard deviations
        scaleDiffNorm = abs(scaleDiff - np.mean(scaleDiff))
        trimSubset = np.array(range(0, len(scaleDiff)))[
            scaleDiffNorm <= 2 * np.std(scaleDiff)
        ]
        # trimSubset = [di for di, dv in enumerate(scaleDiff) if (dv > np.quantile(scaleDiff, 0.05)) and (dv < np.quantile(scaleDiff, 0.95))]
        trimSubsetHL = [sci[ts] for ts in trimSubset]
        scale = np.quantile(scaleDiff[trimSubset], 0.5)
        diffSet = np.abs((HL[trimSubsetHL] / scale) - HS[trimSubset])
        normDiff.append(np.sum(diffSet) / diffSet.shape[0])
        subsetIndex.append(trimSubset)
        scaleList.append(scale)

    minDiff = normDiff.index(min(normDiff))
    # the index to the longer histogram taken from the combinational search
    subsetHLindex = np.array(subsetCombIndex[minDiff])
    # the subset of the combinational subset that returns the minimum scale variation
    subsubsetIndex = subsetIndex[minDiff].tolist()
    subsetHLindex = subsetHLindex[subsubsetIndex].tolist()
    scale = scaleList[minDiff]

    if len(A) >= len(B):
        return subsetHLindex, subsubsetIndex, scale, min(normDiff)
    elif len(B) > len(A):
        return subsubsetIndex, subsetHLindex, 1 / scale, min(normDiff)


def rigidTransform3D(A, centroidA, B, centroidB):
    """
    Given two sets of points, find the best rotation and translation to align them.
    See also [Kabsch, 1976] Kabsch, W. (1976). A solution for the best rotation to relate two sets of vectors.
    Acta. Crystal, 32A:922-923.
    [Kabsch, 1978] Kabsch, W. (1978). A discussion of the solution for the best rotation to related
    two sets of vectors. Acta. Crystal, 34A:827-828.

    Use Singular Value Decomposition to determine rotation matrix.
    Liberally borrowed from http://nghiaho.com/?page_id=671
    R = 3x3 rotation matrix
    t = 3x1 column vector

    :param A: set of 3D points
    :param centroidA: centroid of points within A
    :param B: set of 3D points
    :param centroidB: centroid of points within B
    :return: R: rotation matrix
             t: transpose vector
    """

    A = np.mat(A.T)
    B = np.mat(B.T)
    assert len(A) == len(B)
    N = A.shape[0]  # total points

    # should centroids be imported from Shape object?-----------------------
    centroid_A = np.mat(Point2XYZarray(centroidA).T)
    centroid_B = np.mat(Point2XYZarray(centroidB).T)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.transpose(AA) * BB  # cross-covariance matrix A = Pt * Q

    U, S, Vt = np.linalg.svd(H)

    dSign = np.linalg.det(Vt * U.T)

    if dSign:
        # S[-1] = -S[-1]
        # U[:, -1] = -U[:, -1]
        print("SVD transform reflection detected")
        # Vt[2, :] *= -1 #-----------------------------------Kabsh doesn't hold here?

    R = Vt.T * U.T
    t = -R * centroid_A.T + centroid_B.T
    return R, t


def Point2XYZarray(P):
    """
    Convert list of Point named tuple to np.array[[X], [Y], [Z]]

    :param P: list of Point of type named tuple
    :return: equivalent list of np.array[[X], [Y], [Z]]
    """

    if type(P) == list:
        return np.array([[p.x for p in P], [p.y for p in P], [p.z for p in P]])
    elif type(P) == Point:
        return np.array([[P.x], [P.y], [P.z]])


def XYZarray2Point(XYZ):
    """
    Convert list of np.array[[X], [Y], [Z]] to list of Point named-tuples

    :param XYZ: list of np.array[[X], [Y], [Z]]
    :return:  list of Point of type named tuple
    """

    P = []
    if 3 in XYZ.shape:
        for p in range(0, len(XYZ[0])):
            P.append(Point(XYZ[0][p], XYZ[1][p], XYZ[2][p]))
    else:
        P = Point(XYZ[0], XYZ[1], XYZ[2])
    return P


def transformPoints(P, R, t):
    """
    Translate and rotate points through translation vector t and rotation matrix R

    :param P: points to be operated on
    :param R: rotation matrix
    :param t: translation vector
    :return: translated & rotated points
    """

    XYZ = np.mat(Point2XYZarray(P))
    TXYZ = (R * XYZ) + np.tile(t, (1, XYZ.shape[1]))  # ".T" is self transpose
    return XYZarray2Point(np.array(TXYZ))


def matrix2vectorAngle(M):
    """
    Convert rotation matrix to single rotation axis vector and rotational angle
    :param M: 3x3 numpy rotation matrix
    :return: vector axis 3x1, rotation angle
    """
    angle = np.acos((M[0][0] + M[1][1] + M[2][2] - 1) / 2)
    x = (M[2][1] - M[1][2]) / sqrt(
        (M[2][1] - M[1][2]) ** 2 + (M[0][2] - M[2][0]) ** 2 + (M[1][0] - M[0][1]) ** 2
    )
    y = (M[0][2] - M[2][0]) / sqrt(
        (M[2][1] - M[1][2]) ** 2 + (M[0][2] - M[2][0]) ** 2 + (M[1][0] - M[0][1]) ** 2
    )
    z = (M[1][0] - M[0][1]) / sqrt(
        (M[2][1] - M[1][2]) ** 2 + (M[0][2] - M[2][0]) ** 2 + (M[1][0] - M[0][1]) ** 2
    )

    return Point(x, y, z), angle


def scaleSignature(sigIn, localCentroid, localScale):  # unused
    """
    The histogram (dispSignature) representing the displacements of feature categories from the shape local centroid
    is scaled.

    :param sigIn: input histogram
    :param localCentroid: local centroid Point
    :param localScale: scaling factor
    :return: sigOut: histogram with scaled displacement values
    """
    sigOut = []
    for si in sigIn:
        if type(si) == tuple:  # "Point" type
            pOut = Point(
                localScale * (si.x - localCentroid.x) + localCentroid.x,
                localScale * (si.y - localCentroid.y) + localCentroid.y,
                localScale * (si.z - localCentroid.z) + localCentroid.z,
            )
            sigOut.append(pOut)
        elif type(si) == float:
            # sigOut(si*localScale)
            sigOut.append(si * localScale)
    return sigOut


def scalePoints(pointList, localCentroid, localScale):
    """
    Given a list of Points, scale relative to a specified local centroid by a scale factor.

    :param pointList: list of Points to scale
    :param localCentroid: local centroid Point
    :param localScale: scale factor
    :return: list of scaled Points
    """
    xyz = Point2XYZarray(pointList)
    xyz[0, :] = np.multiply((xyz[0, :] - localCentroid.x), localScale) + localCentroid.x
    xyz[1, :] = np.multiply((xyz[1, :] - localCentroid.y), localScale) + localCentroid.y
    xyz[2, :] = np.multiply((xyz[2, :] - localCentroid.z), localScale) + localCentroid.z

    return XYZarray2Point(xyz)


def LHHelixOrder(pointList, startPoint, localCentroid):
    """
    See leftHandSpiralOrder2(), registration features are represented by an angular value and
    a displacement value from the start point, they are sorted by cylindrical coordinates to
    determine a sequence of features that lie in a helical path. If two points share the same
    Z-axis coordinate value, the point with the lower angular value takes precedence, otherwise
    the point with the lower Z-coordinate takes precedence in the sequence. Because the algorithm
    will generate identical sequences of registration points from asymmetric matching target and
    source models that share the same start point the sequential order must use a common chirality,
    or “handedness” of generative helix. This procedure generates a known order of feature
    registration points.

    From a given start point, create a Z-axis through the local centroid, then order all points
    around this axis according to,
        1. displacement from start point along Z-axis
        2. relative rotational axis around Z-axis

    Points are translated and rotated so that the start point is at the origin of a local cylindrical
    coordinate schema.

    :param pointList: list of feature points
    :param startPoint: identified start point
    :param localCentroid: local point centroid
    :return: index of points sorted to LH helical schema
    """

    # def clip(p):
    #     # round(p.z, 1) -> float("{0:.2f}".format(round(p.z, 1)))
    #     # see https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points
    #     # reduce accuracy to 2 decimal places
    #     return float("{0:.2f}".format(round(p, 1)))

    # number noise has a large effect on vectors near origin
    tol = 1e-10

    normLocalCentroid = np.array(
        [
            localCentroid.x - startPoint.x,
            localCentroid.y - startPoint.y,
            localCentroid.z - startPoint.z,
        ]
    )

    # edge case where localCentroid or startPoint lie close to the origin point

    # if scene:
    #     displayPointFC(Point(normLocalCentroid[0], normLocalCentroid[1], normLocalCentroid[2]), 16711935, scene)

    Z = np.array([0.0, 0.0, normLocalCentroid[2]])  # z vector

    # if scene:
    #     displayPointFC(Point(Z[0], Z[1], Z[2]), 65280, scene)

    # the cross product is a vector normal to the plane
    NLCaxis = np.cross(normLocalCentroid, Z)  # Z-vector
    # NLCaxis = np.cross(normLocalCentroid / np.linalg.norm(normLocalCentroid), Z / np.linalg.norm(Z))
    # this method fails on 180-degree rotation

    if all(NLCaxis[:] < tol):
        # case of selected point being exactly on Z-axis
        # implies that rotation axis can be any vector in XY plane
        NLCaxis = np.array([1.0, 1.0, 0.0])
        NLCtheta = np.pi
    else:
        NLCaxis /= np.linalg.norm(NLCaxis)

        if (
            (abs(NLCaxis[0]) < tol)
            and (abs(NLCaxis[1]) < tol)
            and (abs(NLCaxis[2]) < tol)
        ):
            # should be caught in if statement?
            raise RuntimeError("zero vector error")
        elif (abs(NLCaxis[0]) < tol) and (abs(NLCaxis[1]) < tol):
            NLCaxis = np.array([1.0, 1.0, 0.0])
        elif (abs(NLCaxis[1]) < tol) and (abs(NLCaxis[2]) < tol):
            NLCaxis = np.array([0.0, 1.0, 0.0])
        elif (abs(NLCaxis[0]) < tol) and (abs(NLCaxis[2]) < tol):
            NLCaxis = np.array([1.0, 0.0, 1.0])

        NLCdisp = sqrt(
            normLocalCentroid[0] ** 2
            + normLocalCentroid[1] ** 2
            + normLocalCentroid[2] ** 2
        )
        NLCtheta = np.arccos(normLocalCentroid[2] / NLCdisp)

    # move points to origin based at startpoint
    normPoints = [
        np.array([p.x - startPoint.x, p.y - startPoint.y, p.z - startPoint.z])
        for p in pointList
    ]

    # if scene:
    #     PnormPoints = [Point(np[0], np[1], np[2]) for np in normPoints]
    #     displayPointFC(PnormPoints, 15790320, scene)
    # 16711935 magenta 16711680 blue 65280 green 32767 orange 65535 yellow 16776960 15790320 986895

    pRotAxis = rotationMatrix(NLCaxis, np.sign(Z[2]) * NLCtheta)
    normPoints = [np.dot(pRotAxis, _np) for _np in normPoints]
    normPoints = [Point(_np[0], _np[1], _np[2]) for _np in normPoints]

    # if scene:
    #     displayPointFC(normPoints, 32767, scene)

    # cylCoords = [(np.arctan2(p.x, p.y), float("{0:.2f}".format(round(p.z, 1)))) for p in normPoints]

    cylCoords = [(np.arctan2(p.x, p.y) + np.pi, p.z) for p in normPoints]
    Zset = list(set([c[1] for c in cylCoords]))
    Zbins = sorted(dispsToBins(Zset, 0.1))
    Zgroups = []
    for zb in Zbins:
        zdisps = [np.abs(zb - cc[1]) for cc in cylCoords]
        constZgroup = [cylCoords[i] for i, zd in enumerate(zdisps) if zd < 0.05]
        Zgroups.append(sorted(constZgroup, key=itemgetter(0)))

    offsetZgroups = []  # [Zgroups[0][0]]
    lastMaxAngle = 0
    for z in range(0, len(Zgroups)):  # range(1, len(Zgroups)):
        rOrder = [
            (
                (r[0] - lastMaxAngle),
                r[1],
            )
            if (r[0] > lastMaxAngle)
            else (
                (r[0] - lastMaxAngle + 2 * np.pi),
                r[1],
            )
            for r in Zgroups[z]
        ]
        rIndex = [r[0] for r in sorted(enumerate(rOrder), key=itemgetter(1))]
        lastMaxAngle = Zgroups[z][rIndex[-1]][0]
        [offsetZgroups.append(Zgroups[z][ri]) for ri in rIndex]

    pointIndex = []
    for s in offsetZgroups:
        for i, c in enumerate(cylCoords):
            if s == c:
                pointIndex.append(i)

    # returnPoints = [pointList[p] for p in pointIndex]

    return pointIndex


def leftHandSpiralOrder2(
    sourcePointList, targetPointList, sourceCentroid, targetCentroid, tol, scene=None
):
    """
    Helical point sequencing: order all points on each model according to a helical arrangement,

        1. Determine a registration feature that has a unique point-to-centroid
    displacement value to serve as a starting point. Where there is no registration feature
    with a unique centroid displacement value, the task is then to determine the smallest
    set of registration feature displacements. Each of these sets must then be tested to
    identify a src. See multiTargetHelixIndex[] below.

        2. Once a start point is chosen from this set, registration features are translated so
    that this start point forms a new coordinate origin.

        3. Registration features are rotated so that the model centroid lies co-linear
    with the start point Z-axis. Accomplished via rotation through the axis generated by the
    cross product of a vertical unit vector and a normalised vector constructed from the
    starting point and the centroid. The rotation angle is determined from the arc-cosine of
    these same vectors, (exceptions for small values and start points located on the Z-axis).
    This translation and rotation allows the registration feature points to be converted to
    cylindrical coordinates using a signed arctangent function and Z-axis displacement.

        4. Once registration features are represented by an angular value and a displacement
    value from the start point, they are sorted by cylindrical coordinates to determine
    a sequence of features that lie in a helical path. If two points share the same Z-axis
    coordinate value, the point with the lower angular value takes precedence, otherwise
    the point with the lower Z-coordinate takes precedence in the sequence. This
    algorithm will generate identical sequences of registration points from asymmetric
    matching target and source models that share the same start point. It is necessary that
    both sequences share the same chirality, or “handedness” of generative helix. This
    procedure generates a known order of feature registration points. See LHHelixOrder().

    :param sourcePointList: source list of feature Points
    :param targetPointList: target list of feature Points
    :param sourceCentroid: local centroid of source feature Points
    :param targetCentroid: local centroid of target feature Points
    :param tol: unique value tolerance
    :param scene: coin3d pivy graphical display instance, point display for debug purposes
    :return:
    """

    if len(sourcePointList) != len(targetPointList):
        raise RuntimeError(
            "leftHandSpiralOrder2(): mismatched source and target point list"
        )

    # organise this spiral around an axis formed around a vector
    u = []

    # cheap solution first; find single minimum or maximum shared disp
    # or singular shared disp value
    sourceDisps = localDisp(sourcePointList, sourceCentroid)
    targetDisps = localDisp(targetPointList, targetCentroid)
    uniqueDisp = len(targetDisps)
    uniqueDispIndex = [0]

    # find value with minimum number of similar values, i.e. the most unique value
    # symmetric shapes will have local centroid feature displacements that fall into similar bands dictated
    # by number noise => use a constant scaled to the relative size of the shape,
    # but scaled factor will error once disp << 1, e.g. feature point at origin

    for td in targetDisps:
        # find value with minimum number of similar values
        u = np.where(np.abs(np.array(targetDisps) - td) < tol)[0]
        if len(u) < uniqueDisp:
            uniqueDisp = len(u)
            uniqueDispIndex = list(u)

    # select a source disp based on the first value in uniqueDispInd
    startDispIndex = np.where(
        np.abs(np.array(sourceDisps) - targetDisps[uniqueDispIndex[0]]) < tol
    )[0]

    if startDispIndex.size > 0:
        startDisp = sourceDisps[startDispIndex[0]]
        sourceStarts = np.where(np.abs(np.array(sourceDisps) - startDisp) < tol)[0]
        sourceStart = sourcePointList[sourceStarts[0]]
    else:  # case where there are no matches < tol value, use entire list to find minimum sum displacement src
        sourceStart = sourcePointList[0]
        uniqueDispIndex = [i for i in range(0, len(targetDisps))]

    sourceHelixIndex = LHHelixOrder(sourcePointList, sourceStart, sourceCentroid)
    sourceDispsRef = np.array([sourceDisps[shi] for shi in sourceHelixIndex])

    # if scene:
    #     displayPointFC(sourceStart, 16711935, scene)

    # create list of [T1, T2, etc.] for multiple target configurations
    multiTargetHelixIndex = []
    returnList = [
        sourceHelixIndex,
    ]  # [S, T1, T2, etc]

    # necessary to find all helix combinations from least common start points
    # as several may share equivalent displacements, see CubePyramidPyramid.stp model
    for udi in uniqueDispIndex:
        targetStartPoint = targetPointList[udi]  # ---------------------

        if scene:
            displayPointFC(targetStartPoint, 16711935, scene)

        targetHelixIndex = LHHelixOrder(
            targetPointList, targetStartPoint, targetCentroid
        )
        multiTargetHelixIndex.append(targetHelixIndex)

    # test each target helix generated from equivalent disp start points for matching disp sequence
    minDispError = []
    for mthi in multiTargetHelixIndex:
        # find equivalent point in target set displacements.
        dispError = np.sum(
            np.array([targetDisps[thi] for thi in mthi]) - sourceDispsRef
        )
        minDispError.append(dispError)

        # if dispError < (len(targetDisps)*tol): # too fussy
        #     returnList.append(mthi)

    minIndex = minDispError.index(min(minDispError))
    returnList.append(multiTargetHelixIndex[minIndex])
    if len(returnList) == 1:
        # nothing was found
        return []
    else:
        return returnList


def SVDerror(
    source,
    target,
    sourceCP,
    scaledTargetCP,
    targetCP,
    STscale,
    numberOfPoints,
    scene=None,
):
    """
    Using a Singular Value Decomposition method, translate & rotate points from a source model to target model and take
    displacement from translated & rotated point to equivalent point on target model as displacement error.
    Select points to test from feature list, in the event that numberOfPoints requested > feature points available,
    use randomly selected surface points stored from previous shape object operations.
    To calculate the error of a sampled tranformed point, a ray is projected from the local target centroid, through
    the transformed point and intersects with teh target object surface. The error is given as displacement between this
    surface ray intersection and the transformed point.

    :param source: source CAD geometric object
    :param target: target CAD geometric object
    :param sourceCP: list of source check points
    :param scaledTargetCP: list of scaled target check points
    :param targetCP: list of target check points
    :param STscale: source/target scale
    :param numberOfPoints: number of points for error test requested
    :param scene: coin3d pivy graphical display instance, point display for debug purposes
    :return: list of individual error displacements
    """

    #     for p in [0, 1, 2, 3]:
    #         displayPointFC(sourceCPs[p], 255)

    # for p in [0, 1, 2, 3]:
    #     displayPointFC(scaledTargetCPs[p], 65280)

    # test for case of torus with single point.
    if (
        (len(sourceCP) < 3)
        and (len(scaledTargetCP) < 3)
        and (sum([len(source.rotSymRidgePoints), len(source.rotSymGroovePoints)]) > 0)
        and (sum([len(target.rotSymRidgePoints), len(target.rotSymGroovePoints)]) > 0)
    ):
        torusFlag = 1  # torii are represented with 2 or 3 featurePoints, 1 minimum centre, 1 to 2 max centre
        minCommonPoints = min(
            [len(source.rotSymRidgePoints), len(target.rotSymRidgePoints)]
        )
        # minCommonPoints = min([len(source.featureMaxCentres) + len(source.featureMinCentres), len(target.featureMaxCentres) + len(target.featureMinCentres)])
        if minCommonPoints == 0:
            return math.inf, None, None, None
        else:
            # featureMaxCentre, featureMinCentre?
            pointSelect = sample(source.rotSymRidgePoints, minCommonPoints)
            [sourceCP.append(ps) for ps in pointSelect]
            pointSelect = sample(target.rotSymRidgePoints, minCommonPoints)
            pointSelect = scalePoints(pointSelect, target.centroid, 1 / STscale)
            [scaledTargetCP.append(ps) for ps in pointSelect]
    else:
        torusFlag = 0

    rotationMat, translateVector = rigidTransform3D(
        Point2XYZarray(sourceCP),
        source.centroid,
        Point2XYZarray(scaledTargetCP),
        target.centroid,
    )
    printVerbose("\n SVD derived rotation matrix: ")
    printVerbose(rotationMat)

    printVerbose("\n SVD derived translation vector:")
    printVerbose([i[0] for i in translateVector.tolist()])

    printVerbose("\n Scale difference:" + str(STscale) + "\n")

    # note that curve centres are not surface points, expunge from list
    modelErrors = []
    sourceUniquePoints = [
        scp
        for scp in sourceCP
        if (scp not in source.featureMaxCentres)
        and (scp not in source.featureMinCentres)
    ]
    if len(sourceUniquePoints) > 0:
        sourceUniquePoints = scalePoints(sourceUniquePoints, source.centroid, STscale)
        checkSourcePoints = transformPoints(
            sourceUniquePoints, rotationMat, translateVector
        )

        if scene:
            colorList = [
                "red",
                "green",
                "lime",
                "blue",
                "yellow",
                "purple",
                "cyan",
                "white",
                "black",
                "grey",
                "maroon",
                "navy",
                "olive",
                "silver",
                "teal",
                "orange",
                "lightblue",
            ]
            printVerbose(
                "SVD debug source: " + source.name + " & target: " + target.name
            )
            printVerbose(" SVD feature point offset________________________")

        # for pi, pv in enumerate(sourceUniquePoints):
        #       displayPointFC(pv, colorList[pi % len(colorList)], scene, 'circle9')
        #       scene.addText(pv, str(pi))
        #       displayPointFC(checkSourcePoints[pi], colorList[pi % len(colorList)], scene, 'triangle9')
        #       scene.addText(checkSourcePoints[pi], str(checkSourcePoints[pi]))
        #       scene.saveIV(scene.name)

        # use each of these translated points to find a correspondence on the target model
        if not torusFlag:
            # for csi in range(0, len(checkSourceVectors)):
            for csi in range(0, len(checkSourcePoints)):
                # printVerbose("________________________")
                # printVerbose("predictionTest proj2object vars")
                # printVerbose(target.filepath)
                # printVerbose(checkSourcePoints[csi])

                nearbyPoint = proj2object2(
                    target, checkSourcePoints[csi], target.centroid, "nearest"
                )
                if nearbyPoint:
                    if scene:
                        modelError = localDisp(nearbyPoint, checkSourcePoints[csi])
                        # if type(modelError) == type(list()):
                        #     modelError = min(modelError)
                        printVerbose("Feature point error: " + str(modelError))
                        # printVerbose("Feature point error: " + str(min(localDisp(scaledTargetCP, nearbyPoint))))
                    # print(modelError)

                    # if scene:
                    #     scene.addLine(checkSourcePoints[csi], nearbyPoint, 'proj', 'lime')
                    #     displayPointFC(checkSourcePoints[csi], colorList[csi % len(colorList)], scene, 'triangle9')
                    #     displayPointFC(nearbyPoint, colorList[csi % len(colorList)], scene, 'circle9')
                    #     scene.saveIV(scene.name)

                    modelErrors.append(min(localDisp(targetCP, nearbyPoint)))

    # repeat process to attain full point complement
    # append unique feature points from source.surfacePoints if value < numberOfPoints
    shortfall = numberOfPoints
    if not torusFlag:
        shortfall -= len(sourceUniquePoints)
    if shortfall > 0:
        # randomly select points
        sourceUniquePoints = [choice(source.surfacePoints) for _ in range(shortfall)]
        sourceUniquePoints = scalePoints(sourceUniquePoints, source.centroid, STscale)
        checkSourcePoints = transformPoints(
            sourceUniquePoints, rotationMat, translateVector
        )

        for csi in range(0, len(checkSourcePoints)):
            # printVerbose("predictionTest proj2object vars")
            # printVerbose(target.filepath)
            # printVerbose(checkSourcePoints[csi])

            nearbyPoint = proj2object2(
                target, checkSourcePoints[csi], target.centroid, "nearest", scene
            )

            if nearbyPoint:
                modelError = localDisp(nearbyPoint, checkSourcePoints[csi])

                if scene:
                    scene.addLine(
                        checkSourcePoints[csi], nearbyPoint, "proj", colour="lime"
                    )
                    displayPointFC(
                        checkSourcePoints[csi],
                        colorList[csi % len(colorList)],
                        scene,
                        "circlehollow9",
                    )
                    displayPointFC(
                        nearbyPoint, colorList[csi % len(colorList)], scene, "circle9"
                    )
                    # scene.saveIV(scene.name)

                printVerbose("Random point error: " + str(modelError))
                modelErrors.append(modelError)
    if scene:
        printVerbose("________________________")
        scene.saveIV(scene.name)

    return (
        modelErrors,
        rotationMat,
        [x for xs in translateVector.tolist() for x in xs],
        STscale,
    )


def predictionTest5(source, target, numberOfPoints=20, DistThreshold=1e-2, scene=None):
    """
    Use identified shape features to determine geometric shape matching between source & target.
    - re-revaluate local centroid around feature points (medianCentroidCorrection())
    - get shapes with subset of common feature points (listIntersection())
    - scale target histogram to source histogram (scalePoints())
    - find common lH helical ordering of feature points
    - return median RMS error of transposed source-target points

    :param scene:
    :param source: source CAD geometrical object
    :param target: target CAD geometrical object
    :param numberOfPoints: number of points used to establish similarity
    :param DistThreshold: ?? should be tol ?? absolute displacement indicating unique points
    :return: error value
    """
    # use ranked histogram to calculate affine transform, rotation/scaling/translation
    # get histogram intersection & difference

    sourceCentroidChanged = medianCentroidCorrection(source, tol=0.01)
    targetCentroidChanged = medianCentroidCorrection(target, tol=0.01)

    # degenerate case of two spheres
    sourceIsSphere = 0
    targetIsSphere = 0
    if (
        not source.featureMaxPoints
        and not source.featureMaxCurveDisps
        and not source.featureMinPoints
        and not source.featureMinCurveDisps
        and source.spherePoints
    ):
        sourceIsSphere = True

    if (
        not target.featureMaxPoints
        and not target.featureMaxCurveDisps
        and not target.featureMinPoints
        and not target.featureMinCurveDisps
        and target.spherePoints
    ):
        targetIsSphere = True

    # method ignores spheres comprised of multiple sector radii, another edge case
    if sourceIsSphere and targetIsSphere:
        return (
            1.0,
            None,
            localDisp(source.centroid, target.centroid),
            np.mean(source.featureSphereDisps) / np.mean(target.featureSphereDisps),
        )
    elif (sourceIsSphere and not targetIsSphere) or (
        not sourceIsSphere and targetIsSphere
    ):
        return math.inf, None, None, None

    # SnT, TnS, STscale = listIntersection(source, target, tol=1e-2)  # yet another sensitive constant
    SnT, TnS, STscale, hError = listIntersection2(
        source, target, tol=1e-2
    )  # yet another sensitive constant

    printVerbose("scalar histogram comparison max error: " + str(hError))

    if (len(SnT) < 2) or (len(TnS) < 2) or (len(SnT) != len(TnS)):
        print("Inadequate common points between source and target for comparison")
        return np.Infinity, None, None, None

    # get Source.featurePoints and Target.featurePoints from SnT, TnS intersection, order is important (src listIntersection())
    sourcePointSignature = (
        source.featureMaxPoints
        + source.featureMaxCentres
        + source.featureMinPoints
        + source.featureMinCentres
        + source.spherePoints
    )
    # source.sphereCentre == 0,0,0 => sphere displacements within listIntersection are definitive

    targetPointSignature = (
        target.featureMaxPoints
        + target.featureMaxCentres
        + target.featureMinPoints
        + target.featureMinCentres
        + target.spherePoints
    )

    # dispSignatureIndexed = sorted(enumerate(dispSignature), key=lambda x: x[1], reverse=True)
    # dispSignatureIndexMap = [i[0] for i in dispSignatureIndexed]
    # dispSignature = [i[1] for i in dispSignatureIndexed]

    sourceCommonPoints = [sourcePointSignature[s] for s in SnT]
    targetCommonPoints = [targetPointSignature[t] for t in TnS]

    # points have to be offset by scale from centroid
    scaledTargetCommonPoints = scalePoints(targetCommonPoints, target.centroid, STscale)

    sourceTargetIndex = leftHandSpiralOrder2(
        sourceCommonPoints,
        scaledTargetCommonPoints,
        source.centroid,
        target.centroid,
        tol=5e-2,
    )

    if not sourceTargetIndex:
        return np.Infinity, None, None, None

    sourceCommonPoints = [sourceCommonPoints[si] for si in sourceTargetIndex[0]]

    MeanErrorScores = []
    for targetPointsIndexInstance in sourceTargetIndex[1:]:
        # find the lowest error matrix rotation
        scaledTargetCommonPointTrial = [
            scaledTargetCommonPoints[ti] for ti in targetPointsIndexInstance
        ]

        # for p in [0, 1, 2, 3]:
        #     displayPointFC(scaledTargetCommonPointTrial[p], 255, )
        #     RhinoInstance.AddText(str(p), scaledTargetCommonPointTrial[p])
        #     pass

        ErrorDisps, rotMat, transVector, scaleDifference = SVDerror(
            source,
            target,
            sourceCommonPoints,
            scaledTargetCommonPointTrial,
            targetCommonPoints,
            1 / STscale,
            numberOfPoints,
            scene,
        )
        if ErrorDisps:
            MeanErrorScores.append(RMSError(ErrorDisps))
        else:
            return np.Infinity, None, None, None

        return min(MeanErrorScores), rotMat, transVector, STscale


def RMSError(err):
    """
    Return root mean square error of a list.

    :param err: list of values
    :return: Root Mean Square values
    """

    return np.sqrt((np.array(err) ** 2).mean())


def getModelFeatures2(
    modelFilePath,
    insertionPoint,
    scaleFactor,
    rotationDegrees,
    rotationAxis,
    scene=None,
):
    """
    Import a CAD STEP model file from given file path into CAD model space, also viewing scene space
    extract model features via searchFeatures(),
    return model

    :param modelFilePath: CAD file path
    :param insertionPoint: insertion point within 3D CAD model space
    :param scaleFactor: scale factor
    :param rotationDegrees: initial rotation of model within CAD space in degrees
    :param rotationAxis: defined rotation axis
    :param scene: coin3d pivy graphical display instance, point display for debug purposes
    :return: CAD model object
    """
    # initiate CAD model from file path
    resolution = 35  # TammesSphere(resolution)
    # DA = TammesAngle(resolution)
    DA = DesernoAngle(resolution)
    stopCondition = 0.002
    minCosResolution = cos(8 * stopCondition)
    cosClusterResolution = 32  # cos(32 * stopCondition) # starting value that terminates at minCosResolution

    # return furthest initial intersection for ts (centrePoint) if "localMax" type featureMaxPoints
    # return nearest initial intersection for ts if "localMin" type featureMinPoints

    # call feature search with a relatively low feature point cluster resolution
    # then iteratively reduce for centroid convergence

    # Model = Shape(modelFilePath, insertionPoint, scaleFactor, rotationDegrees, rotationAxis)

    STEPtestModel = STEPmodel()
    STEPtestModel.importSTEPmodel(
        modelFilePath, insertionPoint, scaleFactor, rotationDegrees, rotationAxis
    )

    # insert into CAD model space
    # Model.placeModel()
    if scene:
        scene.addShape(STEPtestModel.objectHandle, "STEPtestModel")

    # detect model
    centroidTranslation(STEPtestModel, 100, scene=None)

    if STEPtestModel.centroid and (
        STEPtestModel.surfaceStatus != "no surface returned"
    ):
        # ElapsedTime = time.clock()  # windows peculiarity ~ time() -- outmoded >Python 3.3
        ElapsedTime = time.process_time()
        # get the initial feature points and cluster from the Tammes sphere seed points

        searchFeatures(
            STEPtestModel,
            [],
            DA / 2,
            (cosClusterResolution * stopCondition),
            resolution,
            0.001,
            10,
            0.9,
            scene,
        )  # 0.001,  # ------------------stopCondition=SCALE-DEPENDENT?

        # genTiming = (time.clock() - ElapsedTime)
        genTiming = time.process_time() - ElapsedTime
        printVerbose(" Similarity matrix generation: {:f} seconds".format(genTiming))
        STEPtestModel.generationTime = genTiming
        printVerbose(
            "median centroid value: x: {:f}, y: {:f}, z: {:f}".format(
                STEPtestModel.centroid.x,
                STEPtestModel.centroid.y,
                STEPtestModel.centroid.z,
            )
        )
        printVerbose(
            "feature max centres: {:d}".format(len(STEPtestModel.featureMaxCentres))
        )
        printVerbose(
            "feature min centres: {:d}".format(len(STEPtestModel.featureMinCentres))
        )
        printVerbose(
            "feature max points: {:d}".format(len(STEPtestModel.featureMaxPoints))
        )
        printVerbose(
            "feature min points: {:d}".format(len(STEPtestModel.featureMinPoints))
        )
        printVerbose(
            "feature sphere disps: {:d}".format(len(STEPtestModel.featureSphereDisps))
        )

        if scene:
            displayPointFC(STEPtestModel.centroid, "lime", scene)

            # featureCentroidIteration(Model)
        # searchFeatures(Model, ancillaryPoints, localCentroid)
    else:
        print("no centroid")
        return None
    # get extrema features from 'S' model
    return STEPtestModel


def printModelAlignment2(insertionPoint, scale, rotation, rotationAxis):
    """
    Print specified insertion point, scale and rotation.

    :param insertionPoint:
    :param scale:
    :param rotation:
    :param rotationAxis:
    :return: stdout display
    """
    print(
        "Model insertion point: x: {:f}, y: {:f}, z: {:f}".format(
            insertionPoint.x, insertionPoint.y, insertionPoint.z
        )
    )
    print("Model insertion scale: {:f}".format(scale))
    print("Model insertion rotation (single axis?): {:f}".format(rotation))
    if rotationAxis:
        print(
            "Model insertion rotation axis: [{:f}, {:f}, {:f}]".format(
                rotationAxis[0], rotationAxis[1], rotationAxis[2]
            )
        )
    else:
        print("Model insertion rotation axis: []")


def shape2pickle(ShapeInstance, CacheDir):
    # script flakiness => save each model separately in a folder
    # n = datetime.now()
    # fName = ShapeInstance.name[:-4] + '_' + str(n.month) + '_' + str(n.day) + '_' + str(n.hour) + '_' + str(n.minute) + '.obj'
    # shapeCacheFilePath = os.path.join(CacheDir, fName)
    shapeCacheFilePath = fileNameTimeStamp(ShapeInstance.name[:-4], "obj", CacheDir)
    try:
        with open(shapeCacheFilePath, "wb") as _:
            pickle.dump(ShapeInstance, _)
        printVerbose(
            ShapeInstance.name[:-4] + " shape written to " + shapeCacheFilePath
        )
    except Exception as e:
        print(
            ShapeInstance.name[:-4]
            + " pickle file write failure to "
            + shapeCacheFilePath
        )


def fileNameTimeStamp(fileNameHead, MSidentifier=None, fDir=None):
    """
    Generate a complete file path based on a text string head, with a generated timestamp body and an
    optional three character Microsoft-style file identifier. Complete pathname optional with supplied
    directory path

    :param fileNameHead: text string file body
    :param MSidentifier: three-character file type identifier
    :param fDir: optional directory pathname
    :return: file path string
    """
    n = dt.now()
    # datetime.now(timezone.utc)
    fName = (
        fileNameHead
        + "_"
        + str(n.month)
        + "_"
        + str(n.day)
        + "_"
        + str(n.hour)
        + "_"
        + str(n.minute)
    )
    if MSidentifier:
        fName = fName + "." + MSidentifier
    if fDir:
        fName = os.path.join(fDir, fName)
    return fName


def singleShapePRtest(localCacheDir, shapeDir, file_S=None, file_N=None, file_P=None):
    """
    Tests to determine identification of similar models, along with rejection of dissimilar models, saved to CSV file.
    Three models are used,
        testModel_S: source model,
        testModel_P: positive src model, which is a transformed version of source model testModel_S
        testModel_N: negative src model, which is an untransformed model that is not testModel_S

    :param localCacheDir: directory to save pre-processed pickle-format shape object files.
    :param shapeDir: directory containing geometry shape models
    :param file_S: pre-calculated source model object, full pathname
    :param file_P: pre-calculated positive src model, which is a transformed version of source model testModel_S
    :param file_N: pre-calculated negative src model, which is an untransformed model that is not testModel_S
    """

    def findExistingShapeData(filepath):
        """
        Test shape object for pre-existing pickle file data or max/min feature point data
        :param filepath: queried shape object file path
        :return: shapeObject
        """
        # test whether shape object already holds shape data before searching for pickled file
        # shapeCacheFilePath = os.path.join(localCacheDir, filepath)
        # if (shapeObject.surfacePoints == []) and (shapeObject.centroid == None):
        #     if len(glob.glob(shapeCacheFilePath)) < 1:  # test file exists
        #         print("Cannot locate testModel_S: " + shapeCacheFilePath)
        #         return None
        #     else:

        try:
            # with open(os.path.join(localCacheDir, shapeObject.filepath), 'rb') as _f:
            with open(filepath, "rb") as _:
                shapeObject = pickle.load(_)  # retrieve STEPmodel object
                if (shapeObject.surfacePoints == []) and (shapeObject.centroid is None):
                    if len(glob.glob(shapeCacheFilePath)) < 1:  # test file exists
                        print("testModel_S object pickle file empty")
                        return None
                # import model into FreeCAD model space
                shapeObject.importSTEPmodel(
                    shapeObject.filepath,
                    shapeObject.insertionPoint,
                    shapeObject.scale,
                    shapeObject.rotation,
                    shapeObject.rotationAxis,
                )
                printVerbose("shape data retrieved from: " + shapeObject.name[:-4])
                return shapeObject
        except Exception as e:
            print("Pickle file load failure from " + shapeCacheFilePath)
            return None

    # create a list of test files
    shapeFiles = []
    for root, dirs, files in os.walk(shapeDir):
        for file in files:
            if file.endswith(".stp"):
                shapeFiles.append(os.path.join(root, file))

    # issue with unit_sphere_x8.0.stp from test-set, non-closed. Same issue disappears with rotated unit_sphere_z8.0.stp
    for file in shapeFiles:
        if "unit_sphere_x8.0.stp" in file:
            shapeFiles.remove(file)

    # Create Source test shape (S), default origin position, rotation, scale
    # test whether shape object already holds shape data before searching for pickled file

    testModel_S = None
    testModel_N = None
    testModel_P = None

    if file_S:
        # shapeCacheFilePath = os.path.join(localCacheDir, file_S) # full path provided
        shapeCacheFilePath = file_S
        testModel_S = findExistingShapeData(shapeCacheFilePath)
        if testModel_S:
            printVerbose("Source model insertion: " + shapeCacheFilePath)
            shapeInsertionPoint = testModel_S.insertionPoint
            shapeScale = testModel_S.scale
            shapeRotation = testModel_S.rotation
            shapeRotationAxis = testModel_S.rotationAxis

    if not testModel_S:
        shapeInsertionPoint = Point(uniform(0, 10), uniform(0, 10), uniform(0, 10))
        shapeScale = uniform(5, 15)
        shapeRotation = uniform(0, 360)
        shapeRotationAxis = [random(), random(), random()]

        while testModel_S is None:
            file_S = shapeFiles[randint(0, len(shapeFiles) - 1)]
            # file_S = r"E:\PY_DELL2\SWKS_Rhino_compare_circle\primitives\Cylinder\unit_cyl_y2.0_blend.03.stp"
            printVerbose("Generating source model: " + file_S)

            if generativeSceneFlag:
                generativeScene_S = (
                    ViewScene()
                )  # add shape comparison scene details by default
                generativeScene_S.name = fileNameTimeStamp(
                    os.path.split(file_S)[1][:-4], "iv", localCacheDir
                )
            else:
                generativeScene_S = None

            testModel_S = getModelFeatures2(
                file_S,
                shapeInsertionPoint,
                shapeScale,
                shapeRotation,
                shapeRotationAxis,
                generativeScene_S,
            )
            shape2pickle(testModel_S, localCacheDir)
            if generativeScene_S:
                generativeScene_S.addPoints(
                    [
                        Point(0.0, 0.0, 0.0),
                    ],
                    "red",
                )  # show origin point
                generativeScene_S.addText(
                    (-15.0, -15.0, 0.0),
                    "Source: " + os.path.split(testModel_S.filepath)[1],
                )
                generativeScene_S.saveIV(generativeScene_S.name)
            playsound(880, 500)

    printModelAlignment2(
        shapeInsertionPoint, shapeScale, shapeRotation, shapeRotationAxis
    )

    # Create a second scaled, rotated, translated model that is a negative src (N)

    if file_N:
        # shapeCacheFilePath = os.path.join(localCacheDir, file_N) # full path provided
        shapeCacheFilePath = file_N
        testModel_N = findExistingShapeData(shapeCacheFilePath)
        if testModel_N:
            printVerbose("Negative model insertion: " + shapeCacheFilePath)
            shapeInsertionPoint = testModel_N.insertionPoint
            shapeScale = testModel_N.scale
            shapeRotation = testModel_N.rotation
            shapeRotationAxis = testModel_N.rotationAxis

    if not testModel_N:
        shapeInsertionPoint = Point(uniform(0, 10), uniform(0, 10), uniform(0, 10))
        shapeScale = uniform(5, 15)
        shapeRotation = uniform(0, 360)
        shapeRotationAxis = [random(), random(), random()]

        while not testModel_N:
            remainingFiles = [os.path.split(sf)[1] for sf in shapeFiles]
            Sfilename = os.path.split(testModel_S.filepath)[1]
            if Sfilename in remainingFiles:
                remainingFiles.remove(Sfilename)
            file_N = remainingFiles[randint(0, len(remainingFiles) - 1)]
            remainingFiles.remove(file_N)
            shapeCacheFilePath = [s for s in shapeFiles if file_N in s][0]
            if shapeCacheFilePath:
                printVerbose("\n\nNegative model generation: " + shapeCacheFilePath)

                if generativeSceneFlag:
                    generativeScene_N = (
                        ViewScene()
                    )  # add shape comparison scene details by default
                    generativeScene_N.name = fileNameTimeStamp(
                        file_N[:-4], "iv", localCacheDir
                    )
                else:
                    generativeScene_N = None

                testModel_N = getModelFeatures2(
                    shapeCacheFilePath,
                    shapeInsertionPoint,
                    shapeScale,
                    shapeRotation,
                    shapeRotationAxis,
                    generativeScene_N,
                )
                shape2pickle(testModel_N, localCacheDir)
                if generativeScene_N:
                    generativeScene_N.addPoints(
                        [
                            Point(0.0, 0.0, 0.0),
                        ],
                        "red",
                    )  # show origin point
                    generativeScene_N.addText(
                        (-15.0, -15.0, 0.0),
                        "Negative: " + os.path.split(testModel_N.filepath)[1],
                    )
                    generativeScene_N.saveIV(generativeScene_N.name)
                playsound(880, 500)

    printModelAlignment2(
        shapeInsertionPoint, shapeScale, shapeRotation, shapeRotationAxis
    )

    # printVerbose("Source model insertion: " + shapeCacheFilePath)

    # Create a transformed version of the source model shape (P).
    if file_P:
        # shapeCacheFilePath = os.path.join(localCacheDir, file_P) # full path provided
        shapeCacheFilePath = file_P
        testModel_P = findExistingShapeData(shapeCacheFilePath)
        if testModel_P:
            printVerbose("Positive model insertion: " + shapeCacheFilePath)
            shapeInsertionPoint = testModel_P.insertionPoint
            shapeScale = testModel_P.scale
            shapeRotation = testModel_P.rotation
            shapeRotationAxis = testModel_P.rotationAxis

    if not testModel_P:
        shapeInsertionPoint = Point(uniform(0, 10), uniform(0, 10), uniform(0, 10))
        shapeScale = uniform(5, 15)
        shapeRotation = uniform(0, 360)
        shapeRotationAxis = [random(), random(), random()]

        printVerbose("\n\nPositive model generation: " + testModel_S.filepath)

        if generativeSceneFlag:
            generativeScene_P = (
                ViewScene()
            )  # add shape comparison scene details by default
            generativeScene_P.name = fileNameTimeStamp(
                testModel_S.name[:-4], "iv", localCacheDir
            )
        else:
            generativeScene_P = None

        testModel_P = getModelFeatures2(
            testModel_S.filepath,
            shapeInsertionPoint,
            shapeScale,
            shapeRotation,
            shapeRotationAxis,
            generativeScene_P,
        )
        if testModel_P is None:
            print("Positive model generation error")
            return None
        shape2pickle(testModel_P, localCacheDir)
        if generativeScene_P:
            generativeScene_P.addPoints(
                [
                    Point(0.0, 0.0, 0.0),
                ],
                "red",
            )  # show origin point
            generativeScene_P.addText(
                (-15.0, -15.0, 0.0),
                "Positive: " + os.path.split(testModel_P.filepath)[1],
            )
            generativeScene_P.saveIV(generativeScene_P.name)
        playsound(880, 500)

    printModelAlignment2(
        shapeInsertionPoint, shapeScale, shapeRotation, shapeRotationAxis
    )

    # create standalone MDI coin3D viewer for each comparison instance
    if not viewerFlag:
        SSPRscene = None
    if viewerFlag:
        SSPRscene = ViewScene()
        SSPRscene.name = fileNameTimeStamp("SSPRcompare", "iv", localCacheDir)
        SSPRscene.addShape(
            testModel_S.objectHandle,
            "testModel_S",
            ambColor="green",
            diffColor="",
            specColor="",
            emColor="",
            transp=0.5,
        )

        if testModel_S.featureMaxPoints:
            SSPRscene.addPoints(testModel_S.featureMaxPoints, "lime", "cross7")
        if testModel_S.featureMinPoints:
            SSPRscene.addPoints(testModel_S.featureMinPoints, "green", "cross7")
        if testModel_S.featureMaxCentres:
            SSPRscene.addPoints(testModel_S.featureMaxCentres, "yellow", "cross7")
        if testModel_S.featureMinCentres:
            SSPRscene.addPoints(testModel_S.featureMinCentres, "white", "cross7")
        # SSPRscene.saveIV(SSPRscene.name)

        SSPRscene.addShape(
            testModel_N.objectHandle,
            "testModel_N",
            ambColor="red",
            diffColor="",
            specColor="",
            emColor="",
            transp=0.5,
        )
        if testModel_N.featureMaxPoints:
            SSPRscene.addPoints(testModel_N.featureMaxPoints, "red", "diamond5")
        if testModel_N.featureMinPoints:
            SSPRscene.addPoints(testModel_N.featureMinPoints, "green", "diamond5")
        if testModel_N.featureMaxCentres:
            SSPRscene.addPoints(testModel_N.featureMaxCentres, "yellow", "diamond5")
        if testModel_N.featureMinCentres:
            SSPRscene.addPoints(testModel_N.featureMinCentres, "white", "diamond5")

        SSPRscene.addShape(
            testModel_P.objectHandle,
            "testModel_P",
            ambColor="green",
            diffColor="",
            specColor="",
            emColor="",
            transp=0.5,
        )
        if testModel_P.featureMaxPoints:
            SSPRscene.addPoints(testModel_P.featureMaxPoints, "blue")
        if testModel_P.featureMinPoints:
            SSPRscene.addPoints(testModel_P.featureMinPoints, "green")
        if testModel_P.featureMaxCentres:
            SSPRscene.addPoints(testModel_P.featureMaxCentres, "yellow")
        if testModel_P.featureMinCentres:
            SSPRscene.addPoints(testModel_P.featureMinCentres, "white")
        SSPRscene.saveIV(SSPRscene.name)

    printVerbose(
        "Corrolation distance measure histogram tests (1 is zero distance, 0 is infinite distance)\n"
    )
    # determine a sensitivity and specificity metric via comparison of true and false targets with source
    SN_histogramRank = distanceRank2(testModel_S, testModel_N, methodName="Correlation")
    printVerbose(
        "\nNegative trial "
        + testModel_S.name
        + " & "
        + testModel_N.name
        + " histogram comparison output: {:.4f}".format(SN_histogramRank)
        + "\n"
    )

    SN_RMSE = np.Infinity
    if SN_histogramRank > 0.5:
        SN_RMSE, SN_rotMat, SN_translation, SN_scale = predictionTest5(
            testModel_S,
            testModel_N,
            numberOfPoints=20,
            DistThreshold=1e-2,
            scene=SSPRscene,
        )  # -------------
        if not SN_RMSE:
            SN_RMSE = np.Infinity

    printVerbose(
        "Negative trial "
        + testModel_S.name
        + " & "
        + testModel_N.name
        + " Root Mean Square Error: {:.6f}".format(SN_RMSE)
    )

    SP_histogramRank = distanceRank2(testModel_S, testModel_P, methodName="Correlation")
    printVerbose(
        "\nPositive trial "
        + testModel_S.name
        + " & "
        + testModel_P.name
        + " histogram comparison output: {:.6f}".format(SP_histogramRank)
        + "\n"
    )

    SP_RMSE = np.Infinity
    if SP_histogramRank > 0.5:
        SP_RMSE, SP_rotMat, SP_translation, SP_scale = predictionTest5(
            testModel_S,
            testModel_P,
            numberOfPoints=20,
            DistThreshold=1e-2,
            scene=SSPRscene,
        )
        # publish testModel_N rotation matrix

        if not SP_RMSE:
            SP_RMSE = np.Infinity

    printVerbose(
        "\nPositive trial "
        + testModel_S.name
        + " & "
        + testModel_P.name
        + " Root Mean Square Error: {:.6f}".format(SP_RMSE)
    )

    GeomTestParams = {
        "source name": testModel_S.name,
        "source gen time": testModel_S.generationTime,
        "source insertion point": testModel_S.insertionPoint,
        "source scale": testModel_S.scale,
        "source rotation": testModel_S.rotation,
        "+ve target name": testModel_P.name,
        "+ve target gen time": testModel_P.generationTime,
        "+ve target insertion point": testModel_P.insertionPoint,
        "+ve target scale": testModel_P.scale,
        "+ve target rotation": testModel_P.rotation,
        "+ve target RMSE src": SP_RMSE,
        "-ve target name": testModel_N.name,
        "-ve target gen time": testModel_N.generationTime,
        "-ve target insertion point": testModel_N.insertionPoint,
        "-ve target scale": testModel_N.scale,
        "-ve target rotation": testModel_N.rotation,
        "-ve target RMSE src": SN_RMSE,
    }

    return GeomTestParams, SSPRscene


def SNPdata2CSV(SNPdata, GeomTestCSVFilePath):
    """
    Open CSV file at GeomTestCSVFilePath and add dict() structured SNP comparison data.

    :param SNPdata:
    :param GeomTestCSVFilePath:
    :return:
    """

    GeomTestCSVheader = [
        "source name",
        "source gen time",
        "source insertion point",
        "source scale",
        "source rotation",
        "+ve target name",
        "+ve target gen time",
        "+ve target insertion point",
        "+ve target scale",
        "+ve target rotation",
        "+ve target RMSE src",
        "-ve target name",
        "-ve target gen time",
        "-ve target insertion point",
        "-ve target scale",
        "-ve target rotation",
        "-ve target RMSE src",
    ]

    # initialise a CSV file to capture values
    if len(glob.glob(GeomTestCSVFilePath)) < 1:  # check if .csv already exists
        # print("Append data to existing Geometry test file [line:4230]")
        # os.remove(GeomTestCSVFilePath)

        # test file exists
        try:
            with open(
                GeomTestCSVFilePath, "x", encoding="utf-8"
            ) as GeomTestCSVFile:  # add the header row
                Twriter = csv.DictWriter(
                    GeomTestCSVFile, lineterminator="\n", fieldnames=GeomTestCSVheader
                )
                Twriter.writeheader()
        except Exception as e:
            print("CSV file init failure: " + GeomTestCSVFilePath)

    try:
        with open(GeomTestCSVFilePath, "a", encoding="utf-8") as GeomTestFile:
            Twriter = csv.DictWriter(
                GeomTestFile, lineterminator="\n", fieldnames=GeomTestCSVheader
            )
            Twriter.writerow(
                {
                    "source name": SNPdata["source name"],
                    "source gen time": SNPdata["source gen time"],
                    "source insertion point": SNPdata["source insertion point"],
                    "source scale": SNPdata["source scale"],
                    "source rotation": SNPdata["source rotation"],
                    "+ve target name": SNPdata["+ve target name"],
                    "+ve target gen time": SNPdata["+ve target gen time"],
                    "+ve target insertion point": SNPdata["+ve target insertion point"],
                    "+ve target scale": SNPdata["+ve target scale"],
                    "+ve target rotation": SNPdata["+ve target rotation"],
                    "+ve target RMSE src": SNPdata["+ve target RMSE src"],
                    "-ve target name": SNPdata["-ve target name"],
                    "-ve target gen time": SNPdata["-ve target gen time"],
                    "-ve target insertion point": SNPdata["-ve target insertion point"],
                    "-ve target scale": SNPdata["-ve target scale"],
                    "-ve target rotation": SNPdata["-ve target rotation"],
                    "-ve target RMSE src": SNPdata["-ve target RMSE src"],
                }
            )
            print("CSV file update: " + GeomTestCSVFilePath)
    except Exception as e:
        print("CSV file open/write failure: " + GeomTestCSVFilePath)


def multiShapePRtest(
    GeomTestCSVFilePath, rankedTestFilePath, shapeCacheDir, debugSceneFlag=0
):
    """
    Using previously saved CAD geometry models from singleShapePRtest(), conduct an exhaustive test between
    all models, deriving the Root Mean Square Error value via predictionTest5() for models with histograms over
    a 0.9 similarity threshold via distanceRank2().

    Creating a new scene for viewing each iteration is too computationally intensive to be practicable except
    during debug. When re-inserting test files into CAD space models for point tests, the original rotation/translation
    is used, rather than the calculated rotation/translation derived from the feature search.

    :param GeomTestCSVFilePath: file path for saving CSV formatted test results
    :param rankedTestFilePath: test output file path
    :param shapeCacheDir: directory to get pre-processed pickle-format shape object files.
    :param debugSceneFlag: create individual scenes of shape comparison for debug purposes
    :return: -
    """

    # Test comparing all previously generated shape files
    # get the list of object histogram files
    shapeFiles = []
    for root, dirs, files in os.walk(shapeCacheDir):
        for file in files:
            if file.endswith(".obj"):
                shapeFiles.append(os.path.join(root, file))

    # sort list alphabetically
    shapeFiles = sorted(shapeFiles, key=str.lower)

    # create dict structure {sourceFilename0:{targetFilename0:RMSE, targetFilename1:RMSE ..} ..}
    rankedTest = dict()
    # targetDict = dict()
    lastSourceName = ""

    for sourceFile in shapeFiles:  # compare all targets
        try:
            print("\nsourceFile: " + str(sourceFile))
            with open(sourceFile, "rb") as _f:
                source = pickle.load(_f)
                sourceName = os.path.split(sourceFile)[-1][:-4]
                # sourceName = source.name[:-4]
                if (sourceName == lastSourceName) or (
                    source.surfaceStatus == "no surface returned"
                ):
                    source = None
                else:
                    lastSourceName = sourceName
        except Exception as e:
            source = None
            print("Pickle file load failure from " + sourceFile)

        if source:
            source.importSTEPmodel()
            source.featureClean()
            # sourceName = os.path.split(sourceFile)[-1][:-4]
            # sourceName = source.name[:-4]
            rankedTest[sourceName] = {"FILES": sourceName}

            for targetFile in shapeFiles:
                targetName = os.path.split(targetFile)[-1][:-4]
                if (targetName in rankedTest) and (
                    sourceName in rankedTest[targetName]
                ):
                    rankedTest[sourceName][targetName] = "-"
                    printVerbose("duplicate")
                else:
                    print("\ntargetFile: " + str(targetFile))

                    try:
                        with open(targetFile, "rb") as _f:
                            target = pickle.load(_f)
                    except Exception as e:
                        target = None
                        print("Pickle file load failure from " + targetFile)

                    if target:
                        target.importSTEPmodel()
                        target.featureClean()
                        # targetName = target.name[:-4]
                        # targetName = os.path.split(targetFile)[-1][:-4]

                        histogramRank = distanceRank2(
                            source, target, methodName="Correlation"
                        )
                        printVerbose(
                            source.name
                            + " & "
                            + target.name
                            + " histogram comparison output: {:.6f}".format(
                                histogramRank
                            )
                        )

                        if not debugSceneFlag:
                            debugScene = None
                        if debugSceneFlag:
                            debugScene = ViewScene()
                            # debugScene.name = fileNameTimeStamp(
                            #     "debugScene", "iv", shapeCacheDir
                            # )
                            debugScene.name = (
                                shapeCacheDir
                                + os.sep
                                + "debugScene_"
                                + str(random())[2:6]
                                + ".iv"
                            )

                            debugScene.addShape(
                                source.objectHandle,
                                "source",
                                ambColor="green",
                                diffColor="",
                                specColor="",
                                emColor="",
                                transp=0.5,
                            )
                            debugScene.addShape(
                                target.objectHandle,
                                "target",
                                ambColor="red",
                                diffColor="",
                                specColor="",
                                emColor="",
                                transp=0.5,
                            )

                        RMSE = np.Infinity
                        if histogramRank > 0.9:
                            RMSE, rotMat, transScalar, scaleRatio = predictionTest5(
                                target,
                                source,
                                numberOfPoints=20,
                                DistThreshold=1e-2,
                                scene=debugScene,
                            )
                            if RMSE:
                                printVerbose(
                                    source.name
                                    + " & "
                                    + target.name
                                    + " Root Mean Square Error: {:.6f}".format(RMSE)
                                )
                            else:
                                RMSE = np.Infinity

                        if debugSceneFlag:
                            debugScene.saveIV(debugScene.name)

                        rankedTest[sourceName][targetName] = RMSE
                        # if targetName + ".0" not in rankedTest[sourceName].keys():
                        #     rankedTest[sourceName][targetName + ".0"] = RMSE
                        # else:
                        #     targetNameList = rankedTest[sourceName].keys()
                        #     targetNameList = [
                        #         tnl for tnl in targetNameList if tnl[:-2] == targetName
                        #     ]
                        #     rankedTest[sourceName][
                        #         targetName + "." + str(len(targetNameList))
                        #     ] = RMSE

    shapeSortedFilenames = sorted(rankedTest.keys(), key=str.lower)

    # initialise a CSV file to capture values

    try:
        # check to see if GeomTestCSVFilePath already exists
        if os.path.exists(GeomTestCSVFilePath):
            os.remove(GeomTestCSVFilePath)
        with open(GeomTestCSVFilePath, "x", encoding="utf-8") as GeomTestCSVFile:
            Twriter = csv.DictWriter(
                GeomTestCSVFile,
                lineterminator="\n",
                fieldnames=[" "] + shapeSortedFilenames,
            )
            Twriter.writeheader()
    except Exception as e:
        print("CSV file init failure: " + GeomTestCSVFilePath)

    try:
        with open(GeomTestCSVFilePath, "a", encoding="utf-8") as GeomTestFile:
            Twriter = csv.DictWriter(
                GeomTestFile,
                lineterminator="\n",
                fieldnames=["FILES"] + shapeSortedFilenames,
            )
            for ssf in shapeSortedFilenames:
                Twriter.writerow(rankedTest[ssf])

            print("CSV file update: " + GeomTestCSVFilePath)
    except Exception as e:
        print("CSV file open/write failure: " + GeomTestCSVFilePath)

    if os.path.exists(rankedTestFilePath):
        os.remove(rankedTestFilePath)
    try:
        with open(rankedTestFilePath, "wb") as _f:
            pickle.dump(rankedTest, _f)
    except Exception as e:
        print("Pickle file dump failure to " + rankedTestFilePath)


def interactiveScript():
    """
    Overall script for,
    1. converting STEP models to feature point representation
    2. testing random rotated, scaled, translated CAD models for similarity
    3. cross-testing previously generated feature point represented models
    Was used for PhD data generation
    """

    # Repository version copies folders over
    # This script version merely prompts for pathnames and assumes populated working directories and shape repository

    # BaseDir = os.path.normpath(r'C:\Users\JaneDoe\Desktop\GeometrySimilarityTest')
    # primitivesDir = os.path.normpath(r'C:\Users\JaneDoe\Desktop\GeometrySimilarityTest\primitives')
    # localCacheDir = os.path.normpath(r'C:\Users\JaneDoe\Desktop\GeometrySimilarityTest\ShapeCache')

    # BaseDir = input('Specify local directory where test output data is saved: ')

    # local machine
    if platform.system() == "Windows":
        BaseDir = os.path.normpath(r"E:\PY_DELL2\SWKS_Rhino_compare_circle")
        primitivesDir = BaseDir + os.sep + r"primitives"
        localCacheDir = BaseDir + os.sep + r"ShapeCache"

    elif platform.system() == "Linux":
        BaseDir = os.path.normpath(
            r"/home/foobert/PycharmProjects/GeometrySimilarityPub"
        )
        primitivesDir = os.path.normpath(
            r"/media/foobert/Dell2HDD/PY_DELL2/SWKS_Rhino_compare_circle/primitives"
        )
        localCacheDir = BaseDir + os.sep + r"ShapeCache"

    for checkDir in [BaseDir, primitivesDir, localCacheDir]:
        if os.path.exists(checkDir):
            print(checkDir + " directory found")
        else:
            print(checkDir + " directory does NOT exist")

    # Tammes Sphere and Deserno Sphere are two variants of equidistant point distribution on a unit sphere.
    # load from pickle files if already generated and available
    # TammesSphereFilePath = os.path.join(BaseDir, "TammesSphereCache.obj")
    # if os.path.exists(TammesSphereFilePath):  # check to see if TammesSphereCache.obj exists
    #     with open(TammesSphereFilePath, "rb") as f:
    #         TammesSphereCache = pickle.load(f)

    DesernoSphereFilePath = os.path.join(BaseDir, "DesernoSphereCache.obj")
    if os.path.exists(
        DesernoSphereFilePath
    ):  # check to see if TammesSphereCache.obj exists
        with open(DesernoSphereFilePath, "rb") as f:
            DesernoSphereCache = pickle.load(f)

    # Precision & recall of models taken from National Design Repository (primitive model classification). Bespalov et al.
    # STEP model test files are contained within 'primitives' subdirectory

    # ShapeCache subdirectory contains pre-processed geometric model objects with feature points from
    # GeometrySimilarityTestA(), to be re-used in GeometrySimilarityTestB()

    F_GUI.setupWithoutGUI()

    # create & view scene of source, positive & negative test models together.
    # single instance per run of singleShapePRtest()
    viewerFlag = 1
    # create & save individual scenes that includes higher detail of model feature detection
    generativeSceneFlag = 1

    # testShape_Source = r'unit_cyl_y2.0_blend.03_2_25_17_26.obj'
    # testShape_Negative = r'unit_cube_xz1.5_blend.01_2_25_17_54.obj'
    # testShape_Positive = r'unit_cyl_y2.0_blend.03_2_25_18_39.obj'

    # testShape_Source = os.path.join(localCacheDir, testShape_Source)
    # testShape_Negative = os.path.join(localCacheDir, testShape_Negative)
    # testShape_Positive = os.path.join(localCacheDir, testShape_Positive)

    # testShape_Source = None
    # testShape_Negative = None
    # testShape_Positive = None
    #
    # SNPtest, SNPscene = singleShapePRtest(
    #     localCacheDir,
    #     primitivesDir,
    #     testShape_Source,
    #     testShape_Negative,
    #     testShape_Positive,
    # )

    doSingleTest = input(
        "Test for single positive and negative src among sampled CAD models? [y / ^y]"
    )
    while doSingleTest.lower() in ["y", "yes"]:
        SNPtest, SNPscene = singleShapePRtest(localCacheDir, primitivesDir)
        SNPdata2CSV(SNPtest, os.path.join(BaseDir, r"PRshapeSimilarity_feb23.csv"))
        doSingleTest = input(
            "Test again for single positive and negative src among sampled CAD models? [y / ^y]"
        )

    doMultiTest = input(
        "Test for similarity matches among multiple processed feature representations? [y / ^y]"
    )
    if doMultiTest.lower() in ["y", "yes"]:
        # with sufficient test model histograms from shapePRtest3models(), run overall comparison
        # fileNameTimeStamp("MultiShapeSimilarity", "csv", localCacheDir)
        # fileNameTimeStamp("rankedTest_A", "obj", primitivesDir)
        multiShapePRtest(
            os.path.join(BaseDir, r"MultiShapeSimilarity_feb23.csv"),
            os.path.join(localCacheDir, r"MultiShapeSimilarity_feb23.obj"),
            localCacheDir,
            debugSceneFlag=0,
        )

    # update TammesSphereCache before exit
    # if TammesSphereCache:
    #     if len(glob.glob(TammesSphereFilePath)) == 1:
    #         os.remove(TammesSphereFilePath)
    #     with open(TammesSphereFilePath, "wb") as _f:
    #         pickle.dump(TammesSphereCache, _f)
    #     printVerbose("  Tammes Sphere Cache written to " + TammesSphereFilePath)

    # update DesernoSphereCache before exit
    if DesernoSphereCache:
        if len(glob.glob(DesernoSphereFilePath)) == 1:
            os.remove(DesernoSphereFilePath)
        with open(DesernoSphereFilePath, "wb") as _f:
            pickle.dump(DesernoSphereCache, _f)
        printVerbose("  Deserno Sphere Cache written to " + DesernoSphereFilePath)

    sys.exit(0)


def checkDirPath(dirPath):
    if os.path.isdir(dirPath) and os.access(dirPath, os.R_OK):
        return dirPath
    else:
        raise Exception("{0} is not a readable dir".format(dirPath))


def checkFilePath(filePath):
    if os.path.isfile(filePath) and os.access(filePath, os.R_OK):
        return filePath
    else:
        raise Exception("{0} is not a readable filepath".format(filePath))


if __name__ == "__main__":
    # CLItest.py v1

    CLIparser = argparse.ArgumentParser(prog="src", formatter_class=argparse.RawTextHelpFormatter)

    CLIsubparsers_help = (
        "singlePRtest is a precision-recall tests performed between similar and dissimilar STEP models "
        + "converted to feature point representation"
        + "\nmultiPRtest is a set of precision-recall performed over all existing feature point model "
        + "representations"
    )

    CLIsubparsers = CLIparser.add_subparsers(
        dest="command", title="Test options", help=CLIsubparsers_help
    )

    singlePR_CLIparserHelp = (
        "Feature point model and shape similarity calculation for STEP input files"
        + "Model file pathnames may be individually entered (source_filename, positive_filename, negative_filename)"
        + "or may be selected at random from a directory of STEP format models (input_directory_path)"
    )

    singlePR_CLIparser = CLIsubparsers.add_parser(
        "singlePRtest",
        usage="convert STEP shape files to feature point representation and perform similarity test",
        description=singlePR_CLIparserHelp,
        fromfile_prefix_chars="@",
    )

    singlePR_CLIparser.add_argument(
        "-I",
        "--input_dir_path",
        type=checkDirPath,
        # default='/tmp/non_existent_dir',
        nargs="?",
        help="STEP test file directory pathname (random selection)",
    )

    singlePR_CLIparser.add_argument(
        "-O",
        "--output_dir_path",
        type=checkDirPath,
        nargs="?",
        help="output directory pathname",
    )

    singlePR_CLIparser.add_argument(
        "-of", "--csv_file", nargs="?", help="output file directory pathname"
    )

    singlePR_CLIparser.add_argument(
        "-sf",
        "--source_file",
        type=checkFilePath,
        nargs="?",
        help="source single STEP file pathname (dual model selection)",
    )

    singlePR_CLIparser.add_argument(
        "-nf",
        "--negative_file",
        type=checkFilePath,
        nargs="?",
        help="negative single STEP file pathname (dual model selection)",
    )

    singlePR_CLIparser.add_argument(
        "-pf",
        "--positive_file",
        type=checkFilePath,
        nargs="?",
        help="optional positive single STEP file pathname (dual model selection)",
    )

    singlePR_CLIparser.add_argument(
        "-s",
        "--scenegraph",
        help="output scenegraph file showing source, positive, negative shapes ",
        action="store_true",
    )  # on/off flag

    singlePR_CLIparser.add_argument(
        "-s2",
        "--model_scene",
        help="output scenegraph file showing individual shapes and feature points",
        action="store_true",
    )  # on/off flag

    singlePR_CLIparser.add_argument(
        "-v", "--verbose", action="store_true"
    )  # on/off flag

    # singlePR_CLIparser.add_argument('--version', action='version', version='%(prog)s 0.x')

    multiPR_CLIparserHelp = (
        "Multiple shape discrimination using precalculated feature point shape representation"
        + " selected from a directory of feature point format models (input_directory_path)"
    )

    multiPR_CLIparser = CLIsubparsers.add_parser(
        "multiPRtest",
        usage="test multiple feature point representation files for similarity",
        description=multiPR_CLIparserHelp,
        fromfile_prefix_chars="@",
    )

    multiPR_CLIparser.add_argument(
        "-I",
        "--input_dir_path",
        type=checkDirPath,
        # default='/tmp/non_existent_dir',
        nargs="?",
        help="shape object file directory pathname",
    )

    multiPR_CLIparser.add_argument(
        "-O",
        "--output_dir_path",
        type=checkDirPath,
        nargs="?",
        help="output directory pathname",
    )

    multiPR_CLIparser.add_argument(
        "-of",
        "--csv_file",
        nargs="?",
        help="CSV output file directory pathname",
    )

    multiPR_CLIparser.add_argument(
        "-obf",
        "--obj_file",
        nargs="?",
        help="python object output file directory pathname",
    )

    multiPR_CLIparser.add_argument(
        "-v", "--verbose", action="store_true"
    )  # on/off flag

    multiPR_CLIparser.add_argument(
        "-s",
        "--debug_scenegraph",
        help="output debug scenegraph file showing source, positive, negative shapes and transform points",
        action="store_true",
    )  # on/off flag

    # multiPR_CLIparser.add_argument('--version', action='version', version='%(prog)s 0.2')

    CLIargs = CLIparser.parse_args()

    if CLIargs.command == "singlePRtest":
        isVerbose = CLIargs.verbose
        # create & view scene of source, positive & negative test models together.
        # single instance per run of singleShapePRtest()
        viewerFlag = CLIargs.scenegraph
        # create & save individual scenes that includes higher detail of model feature detection
        generativeSceneFlag = CLIargs.model_scene

        if viewerFlag or generativeSceneFlag:
            F_GUI.setupWithoutGUI()

        # ShapeCache subdirectory contains pre-processed geometric model objects with feature points from
        # singleShapePRtest(), to be re-used in shapePRtestNmodels()
        if CLIargs.output_directory_path:
            localCacheDir = CLIargs.output_directory_path
        else:
            localCacheDir = os.getcwd() + ".." + os.sep +  + r"data" + os.sep + r"shapeCache"
            os.makedirs(localCacheDir, exist_ok=True)

        testShape_Source = CLIargs.source_filename
        testShape_Negative = CLIargs.negative_filename
        testShape_Positive = CLIargs.positive_filename

        if (
            testShape_Positive is None
            and testShape_Negative is None
            and testShape_Source is not None
        ):
            testShape_Source = testShape_Positive
            testShape_Positive = None

        nonRandomTest = True
        if (testShape_Source is None and testShape_Negative is None) or (
            testShape_Source is None
            and testShape_Negative is None
            and testShape_Positive is None
        ):
            nonRandomTest = False

        # Precision & recall of models taken from National Design Repository (primitive models). Bespalov et al.
        # STEP model test files are contained within 'primitives' subdirectory
        if CLIargs.input_directory_path:
            primitiveShapeDir = CLIargs.input_directory_path
            if not os.path.isdir(primitiveShapeDir):
                guessPrimitiveShapeDir = os.getcwd() + ".." + os.sep + r"data"  + os.sep + r"primitives"
                if not os.path.isdir(guessPrimitiveShapeDir):
                    if not nonRandomTest:
                        print("No input files identified")
                        sys.exit(1)
                else:
                    primitiveShapeDir = guessPrimitiveShapeDir

        if CLIargs.csv_filename:
            if not CLIargs.csv_filename[-4:] == ".csv":
                CLIargs.csv_filename = CLIargs.csv_filename + ".csv"
            if not os.path.isdir(CLIargs.csv_filename):
                CSVoutput = os.path.join(localCacheDir, CLIargs.csv_filename)
        else:
            CSVoutput = os.path.join(localCacheDir, r"PRshapeSimilarity.csv")

        SNPtest, SNPscene = singleShapePRtest(
            localCacheDir,
            primitiveShapeDir,
            testShape_Source,
            testShape_Negative,
            testShape_Positive,
        )
        SNPdata2CSV(SNPtest, CSVoutput)

    if CLIargs.command == "multiPRtest":
        isVerbose = CLIargs.verbose
        # create & view scene of source, positive & negative test models together.
        # multi instance per run of multiShapePRtest()
        debugSceneFlag = CLIargs.debug_scenegraph

        if debugSceneFlag:
            F_GUI.setupWithoutGUI()

        # ShapeCache subdirectory contains pre-processed geometric model objects with feature points from
        # multiShapePRtest(), to be re-used in shapePRtestNmodels()
        if CLIargs.input_directory_path:
            localCacheDir = CLIargs.input_directory_path
            if not os.path.isdir(localCacheDir):
                guessLocalCacheDir = os.getcwd() + ".." + os.sep +  + r"data" + os.sep + r"shapeCache"
                if os.path.isdir(guessLocalCacheDir):
                    localCacheDir = guessLocalCacheDir
                else:
                    print("No input files identified")
                    sys.exit(1)

        if CLIargs.output_directory_path:
            localOutputDir = CLIargs.output_directory_path
        else:
            localOutputDir = os.getcwd()

        if CLIargs.csv_filename:
            if not CLIargs.csv_filename[-4:] == ".csv":
                CLIargs.csv_filename = CLIargs.csv_filename + ".csv"
            if not os.path.isdir(CLIargs.csv_filename):
                CSVoutput = os.path.join(localOutputDir, CLIargs.csv_filename)
        else:
            CSVoutput = fileNameTimeStamp(
                "multiShapeTest", MSidentifier="csv", fDir=localOutputDir
            )

        if CLIargs.obj_filename:
            if not CLIargs.obj_filename[-4:] == ".obj":
                CLIargs.obj_filename = CLIargs.obj_filename + ".obj"
            if not os.path.isdir(CLIargs.obj_filename):
                objOutput = os.path.join(localOutputDir, CLIargs.obj_filename)
        else:
            objOutput = fileNameTimeStamp(
                "multiShapeTest", MSidentifier="obj", fDir=localOutputDir
            )

        multiShapePRtest(
            CSVoutput,
            objOutput,
            localCacheDir,
            debugSceneFlag,
        )

# TODO: PEP8 lowercase_variable rename

# TODO: automated determination of the various fudge factors ? using decision trees, e.g ?
# from sklearn.neighbors import KDTree
# tree = KDTree(np.array(xyz), leaf_size=2)

# TODO: more robust handling of scaling related parameters

# TODO: substitute ridge crawling for search timing out
#  e.g. using getRadialEdges()
