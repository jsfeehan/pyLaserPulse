#! /usr/bin/env python3
# -*-encoding: utf-8 -*-

from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np  # required for some fmt commands
import multiprocessing as mp
import time

from pyLaserPulse.single_plot_window.mplwidget import MplWidget
from pyLaserPulse.single_plot_window.plot_gallery import Ui_MainWindow
from pyLaserPulse.data import paths
import pyLaserPulse.sys_info as si
import pyLaserPulse.single_plot_window.single_matplotlib_window as smw

# Taskbar icon in windows
if si.OS == "Windows":
    import ctypes
    myappid = 'MPLGallery'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


def launch_plot(plot_dicts):
    """
    Launch the MatplotlibGallery.

    plot_dicts: list.
        List of plot dictionaries produced by optical assembly classes.
    """
    app = QtWidgets.QApplication(['MPLGallery'])
    app.setQuitOnLastWindowClosed(True)
    # dpi = app.screens()[0].physicalDotsPerInch()
    dpi = 96
    plotWindow = MatplotlibGallery(
        plot_dicts=plot_dicts, windowTitle='pyLaserPulse simulation gallery',
        dpi=dpi)
    plotWindow.show()
    app.exec()


def concatenate_plot_dicts(dict_list):
    """
    Take all dictionaries in list dict_list and turn them into one dictionary.

    dict_list: list of dictionaries.
    """
    dictionary = {}
    for d in dict_list:
        dictionary.update(d)
    return dictionary


def savePlotFunc(plotInfo, plotName, directory):
    """
    Almost the same as MatplotlibGaller.makePlot.

    Redefining AGAIN because MplWidget objects aren't pickleable.
    Need to replot everything to use multiprocessing library. This ends up
    being faster in the long run.

    I don't think it's possible to define the thumbnails or the pop-out plots
    from MatplotlibGallery.onMplWidgetClick using this function. Returning
    an smw.MatplotlibWindow object doesn't seem to work ('wrapper to C/C++
    object has been deleted').

    Args:
        plotInfo: tuple.
            (axis object, list of formatting commands as strings).
        plotName: string. Name given to the plot.
        directory: string. Absolute directory to which plots are to be saved.
    """

    window = smw.MatplotlibWindow(windowTitle=plotName)
    window.ui.plotWidget.canvas.axes.clear()
    ax = plotInfo[0]
    fmt = plotInfo[1]

    fontsize = 13
    labelsize = 12
    titlesize = 14
    legendsize = 11

    if len(ax.lines) > 0:  # line plot
        # Set line data
        for l in ax.lines:
            data = l.get_data()
            d = l.__dict__
            if ax.get_yscale() == 'log':
                window.ui.plotWidget.canvas.axes.semilogy(
                    data[0], data[1], d['_marker']._marker,
                    ls=d['_linestyle'], lw=d['_linewidth'],
                    alpha=d['_alpha'], c=d['_color'],
                    markeredgecolor=d['_markeredgecolor'],
                    markersize=d['_markersize'],
                    markerfacecolor=d['_markerfacecolor'],
                    markeredgewidth=d['_markeredgewidth'])
            else:
                window.ui.plotWidget.canvas.axes.plot(
                    data[0], data[1], d['_marker']._marker,
                    ls=d['_linestyle'], lw=d['_linewidth'],
                    alpha=d['_alpha'], c=d['_color'],
                    markeredgecolor=d['_markeredgecolor'],
                    markersize=d['_markersize'],
                    markerfacecolor=d['_markerfacecolor'],
                    markeredgewidth=d['_markeredgewidth'])
    else:
        # hack to get pcolormesh working.
        # p made member of ax in optical assembly class.
        X = ax.p._coordinates.T[0, 0:-1, 0] \
            + np.diff(ax.p._coordinates.T[0, :, 0]) / 2
        Y = ax.p._coordinates.T[1, 0, 0:-1] \
            + np.diff(ax.p._coordinates.T[1, 0, :]) / 2
        data = ax.p.get_array()
        data = data.reshape(len(Y), len(X))
        im = window.ui.plotWidget.canvas.axes.pcolormesh(
            X, Y, data, cmap=ax.p.__dict__['cmap'],
            norm=ax.p.__dict__['_norm'])
        cb = window.ui.plotWidget.canvas.figure.colorbar(im)
        cb.set_label(label=ax.colorbar_label, size=labelsize)
        cb.ax.tick_params(labelsize=labelsize)
    window.ui.plotWidget.canvas.axes.set_position(ax.get_position())
    window.ui.plotWidget.canvas.axes.set_xlabel(
        ax.get_xlabel(), fontsize=fontsize)
    window.ui.plotWidget.canvas.axes.set_ylabel(
        ax.get_ylabel(), fontsize=fontsize)
    window.ui.plotWidget.canvas.axes.tick_params(
        axis='both', labelsize=labelsize)
    window.ui.plotWidget.canvas.axes.set_ylim(ax.get_ylim())
    window.ui.plotWidget.canvas.axes.set_xlim(ax.get_xlim())
    window.ui.plotWidget.canvas.axes.set_title(
        ax.get_title(), fontsize=titlesize)

    if 'axes.legend' not in fmt:
        hndl, lbl = ax.get_legend_handles_labels()
        window.ui.plotWidget.canvas.axes.legend(
            hndl, lbl, frameon=False, fontsize=legendsize)

    # Execute additional formatting commands
    for cmd in fmt:
        if 'self.' in cmd:
            cmd = cmd.replace('self.', 'plotWidget.canvas.')
        cmd = cmd.replace('plotWidget', 'window.ui.plotWidget')
        if not cmd.startswith('window.ui.plotWidget.canvas.'):
            cmd = 'window.ui.plotWidget.canvas.' + cmd
        exec(cmd)
    window.ui.plotWidget.canvas.figure.tight_layout()
    window.ui.plotWidget.canvas.draw()
    figFile = (directory + '/' + plotName + '.png')
    window.ui.plotWidget.canvas.figure.savefig(figFile, dpi=180)
    time.sleep(0.5)
    window.close()


class MatplotlibGallery(QtWidgets.QMainWindow):
    class signals(QtCore.QObject):
        saveAllPlots = QtCore.pyqtSignal(list, list, str, list)

    signals = signals()

    def __init__(self, plot_dicts=[], windowTitle=None, dpi=96):
        """
        Application that puts all plots in a single window.
        Plots can be opened in their own large window by clicking on the
        thumnails.

        plot_dicts: list.
            List containing all dictionaries of plot information from the
            amplifier objects.
            The following naming convention MUST be followed:
            "amplifier name: plot name"
        windowTitle: str.
            Title of the window.
        dpi: int
            Screen dpi.

        James Feehan, 17/5/2022
        """
        super(MatplotlibGallery, self).__init__()
        self.setWindowIcon(QtGui.QIcon(paths.single_plot_window.icon))
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle(windowTitle)
        self.dpi = dpi
        self.cols = 3  # number of plot columns
        self.thumbnailSize = 350
        self.galleryLayout = QtWidgets.QVBoxLayout()
        self.ui.groupBox_gallery.setLayout(self.galleryLayout)
        self.plotDicts = plot_dicts

        self.changeStatusLabel("Initializing.")

        self.plotNames = []
        for d in self.plotDicts:
            pn = list(d.keys())
            pn.sort()
            self.plotNames.append(pn)

        self.groupBoxLabelsFromDict()
        self.makeGalleryGroupBoxes()
        self.populateGalleryGroupBoxes()

        self.ui.actionSave_all_plots.triggered.connect(
            self.saveAllPlots)

        self.saveAllPlotsThread = QtCore.QThread()

        self.pps = parallelPlotSaver()
        self.signals.saveAllPlots.connect(self.pps.saveAllPlots)
        self.pps.signals.finished.connect(self.changeStatusLabel)
        self.pps.moveToThread(self.saveAllPlotsThread)

        self.saveAllPlotsThread.start()

        self.changeStatusLabel("Ready.")

    def groupBoxLabelsFromDict(self):
        """
        Creates a list of labels for the amplifier group boxes using
        self.plot_dicts.keys(). The following naming convention MUST be
        followed:
        "GROUPNAME: plot title".
        """
        self.groupBoxLabels = []
        for d in self.plotDicts:
            self.groupBoxLabels.extend(list(d.keys()))
        self.groupBoxLabels = [l.split(':')[0] for l in self.groupBoxLabels]
        self.groupBoxLabels = list(dict.fromkeys(self.groupBoxLabels))

    def makeGalleryGroupBoxes(self):
        """
        Populate self.ui.groupBox_gallery with one group box for each
        amplifier simulated.

        The groupBoxes are enumerated and labelled according to
        self.groupBoxLabels, so self.groupBoxLabelsFromDict() must be called
        first and the naming convention must be followed.
        """
        self.galleryGroupBoxList = []
        self.galleryGroupBoxLayouts = []
        for l in self.groupBoxLabels:
            lo = QtWidgets.QGridLayout()
            self.galleryGroupBoxLayouts.append(lo)
            gb = QtWidgets.QGroupBox(l)
            gb.setMinimumSize(360, 360)
            gb.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Expanding)
            gb.setLayout(lo)
            self.galleryGroupBoxList.append(gb)
            self.galleryLayout.addWidget(gb)

    def populateGalleryGroupBoxes(self):
        """
        Add MplWidgets to all groupboxes in self.galleryGroupBoxList.
        """
        for i, l in enumerate(self.groupBoxLabels):
            num = len(self.plotDicts[i].keys())
            row = 0
            col = 0
            for j in range(num):
                if (j % self.cols) == 0:
                    col = 0
                    row += 1
                mplw = MplWidget()
                mplw.setFixedSize(self.thumbnailSize, self.thumbnailSize)
                mplw.name = self.plotNames[i][j]
                self.makePlot(mplw, self.plotDicts[i][mplw.name])
                mplw.canvas.mpl_connect(
                    'button_press_event',
                    lambda state, x=self.plotDicts[i][mplw.name],
                    y=mplw.name: self.onMplWidgetClick(x, y))
                self.galleryGroupBoxLayouts[i].addWidget(mplw, row, col)
                col += 1
            self.galleryGroupBoxList[i].setMinimumSize(
                int(self.cols * self.thumbnailSize),
                int(1.05 * row * self.thumbnailSize))

    def onMplWidgetClick(self, plotInfo, plotName):
        """
        Launch new window containing only an MplWidget and plot the data in
        plotdict.

        Args:
            plotInfo: tuple.
                (axis object, list of formatting commands as strings).
            plotName: string.
            groupBoxLabel: None or string. Name of the groupBox that the plots
                are put in. This is prepended to the file name when saving all
                plots.
        """
        window = smw.MatplotlibWindow(self, windowTitle=plotName)
        self.makePlot(window.ui.plotWidget, plotInfo, thumbnail=False)
        window.show()

    def makePlot(self, plotWidget, plotInfo, thumbnail=True):
        """
        Update plotWidget.

        Args:
            plotWidget: MplWidget object.
            plotInfo: tuple.
                (axis object, list of formatting commands as strings).
            thumbnail: bool (default True).
                Misses font formatting, legend, etc assuming that the resulting
                plot will be too small to read these features.
        """
        plotWidget.canvas.axes.clear()
        ax = plotInfo[0]
        fmt = plotInfo[1]

        if thumbnail:
            fontsize = 8
            labelsize = 8
            titlesize = 9
            legendsize = 8
        else:
            fontsize = 13
            labelsize = 12
            titlesize = 14
            legendsize = 11

        if len(ax.lines) > 0:  # line plot
            # Set line data
            for l in ax.lines:
                data = l.get_data()
                d = l.__dict__
                if ax.get_yscale() == 'log':
                    plotWidget.canvas.axes.semilogy(
                        data[0], data[1], d['_marker']._marker,
                        ls=d['_linestyle'], lw=d['_linewidth'],
                        alpha=d['_alpha'], c=d['_color'],
                        markeredgecolor=d['_markeredgecolor'],
                        markersize=d['_markersize'],
                        markerfacecolor=d['_markerfacecolor'],
                        markeredgewidth=d['_markeredgewidth'])
                else:
                    plotWidget.canvas.axes.plot(
                        data[0], data[1], d['_marker']._marker,
                        ls=d['_linestyle'], lw=d['_linewidth'],
                        alpha=d['_alpha'], c=d['_color'],
                        markeredgecolor=d['_markeredgecolor'],
                        markersize=d['_markersize'],
                        markerfacecolor=d['_markerfacecolor'],
                        markeredgewidth=d['_markeredgewidth'])
        else:
            # hack to get pcolormesh working.
            # p made member of ax in optical assembly class.
            X = ax.p._coordinates.T[0, 0:-1, 0] \
                + np.diff(ax.p._coordinates.T[0, :, 0]) / 2
            Y = ax.p._coordinates.T[1, 0, 0:-1] \
                + np.diff(ax.p._coordinates.T[1, 0, :]) / 2
            data = ax.p.get_array()
            data = data.reshape(len(Y), len(X))
            im = plotWidget.canvas.axes.pcolormesh(
                X, Y, data, cmap=ax.p.__dict__['cmap'],
                norm=ax.p.__dict__['_norm'])
            cb = plotWidget.canvas.figure.colorbar(im)
            cb.set_label(label=ax.colorbar_label, size=labelsize)
            cb.ax.tick_params(labelsize=labelsize)
        plotWidget.canvas.axes.set_position(ax.get_position())
        plotWidget.canvas.axes.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
        plotWidget.canvas.axes.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
        plotWidget.canvas.axes.tick_params(axis='both', labelsize=labelsize)
        plotWidget.canvas.axes.set_ylim(ax.get_ylim())
        plotWidget.canvas.axes.set_xlim(ax.get_xlim())
        plotWidget.canvas.axes.set_title(ax.get_title(), fontsize=titlesize)

        # if not thumbnail:
        if 'axes.legend' not in fmt:
            hndl, lbl = ax.get_legend_handles_labels()
            plotWidget.canvas.axes.legend(
                hndl, lbl, frameon=False, fontsize=legendsize)

        # Execute additional formatting commands
        for cmd in fmt:
            if 'plotWidget.canvas' not in cmd:
                cmd = 'plotWidget.canvas.' + cmd
            exec(cmd)
        if thumbnail:
            plotWidget.canvas.figure.set_size_inches(
                self.thumbnailSize/self.dpi, self.thumbnailSize/self.dpi)
        plotWidget.canvas.figure.tight_layout()
        plotWidget.canvas.draw()

    def saveAllPlots(self):
        """
        Make and save, but do not display, all plots in the gallery.

        A dialogue box opens so that that user can select the directory where
        the plots are to be saved.
        """
        self.saveAllPlotsDir = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select folder')
        self.changeStatusLabel("Saving all plots to %s" % self.saveAllPlotsDir)
        self.signals.saveAllPlots.emit(
            self.plotDicts, self.plotNames, self.saveAllPlotsDir,
            self.groupBoxLabels)

    def changeStatusLabel(self, statString):
        """
        Change the status label test.
        """
        fullStr = "Status: " + statString
        self.ui.label_status.setText(fullStr)


class parallelPlotSaver(QtCore.QObject):
    """
    Worker class for saving all plots in parallel
    """
    class signals(QtCore.QObject):
        finished = QtCore.pyqtSignal(str)

    signals = signals()

    @QtCore.pyqtSlot(list, list, str, list)
    def saveAllPlots(self, plotDicts, plotNames, saveDir, groupBoxLabels):
        if si.OS == 'Windows':  # multiprocessing not working on Windows?
            for i, l in enumerate(groupBoxLabels):
                for n in iter(plotNames[i]):
                    n2 = n.replace(':', ' -')  # Windows-friendly name
                    savePlotFunc(plotDicts[i][n], n2, saveDir)
            self.signals.finished.emit("Ready.")
        else:
            args = []
            for i, l in enumerate(groupBoxLabels):
                for n in iter(plotNames[i]):
                    n2 = n.replace(':', ' -')  # Windows-friendly name
                    args.append((plotDicts[i][n], n2, saveDir))
            with mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1) as p:
                r = p.starmap(savePlotFunc, iter(args))
            if r:
                self.signals.finished.emit("Ready.")
