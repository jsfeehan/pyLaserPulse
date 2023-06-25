#! /usr/bin/env python3
# -*-encoding: utf-8 -*-

from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavBar
from pyLaserPulse.single_plot_window.plot_widget import Ui_MainWindow


class MatplotlibWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, windowTitle=None):
        """
        Application that puts all plots in a single window.
        Plots can be selected from a comboBox drop-down menu.
        The combobox is populated from plot_dict.keys().

        James Feehan, 14/5/2022
        """
        super(MatplotlibWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle(windowTitle)
        self.addToolBar(NavBar(self.ui.plotWidget.canvas, self))
