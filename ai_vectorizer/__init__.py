# -*- coding: utf-8 -*-
"""
AI Vectorizer for Archaeology
QGIS Plugin for AI-assisted contour digitizing from historical maps.

Copyright (C) 2026 nuri9

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.
"""


def classFactory(iface):
    """Load AIVectorizer class from file ai_vectorizer.plugin.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .plugin import AIVectorizer
    return AIVectorizer(iface)
