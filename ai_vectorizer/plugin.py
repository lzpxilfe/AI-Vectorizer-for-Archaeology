# -*- coding: utf-8 -*-
"""
AI Vectorizer for Archaeology - Main Plugin Class

Copyright (C) 2026 nuri9
GNU General Public License v2
"""

import os
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QToolBar

from qgis.core import QgsProject

from .ui.main_dialog import AIVectorizerDialog


class AIVectorizer:
    """Main QGIS Plugin class for AI Vectorizer."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance for QGIS.
        :type iface: QgsInterface
        """
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = '&ArchaeoTrace'
        self.toolbar = None
        self.dialog = None

    def tr(self, message):
        """Get the translation for a string."""
        return QCoreApplication.translate('ArchaeoTrace', message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None
    ):
        """Add a toolbar icon / menu item."""
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            if self.toolbar is None:
                self.toolbar = self.iface.addToolBar('ArchaeoTrace')
                self.toolbar.setObjectName('ArchaeoTraceToolbar')
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToVectorMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        
        # Main dialog action
        self.add_action(
            icon_path,
            text=self.tr('ArchaeoTrace'),
            callback=self.run,
            parent=self.iface.mainWindow(),
            status_tip=self.tr('Open ArchaeoTrace dialog')
        )

    def unload(self):
        """Remove the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginVectorMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)
        
        if self.toolbar is not None:
            del self.toolbar

    def run(self):
        """Open the main plugin as a docked panel on the left."""
        from qgis.PyQt.QtCore import Qt
        
        if self.dialog is None:
            self.dialog = AIVectorizerDialog(self.iface, parent=self.iface.mainWindow())
            # Add as dock widget to left side
            self.iface.addDockWidget(Qt.LeftDockWidgetArea, self.dialog)
        
        self.dialog.show()
        self.dialog.raise_()
